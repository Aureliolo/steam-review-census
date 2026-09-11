//! The window.
//!
//! Everything here is a thin front for the same core the pipeline runs. A command in this
//! module may read the library, start a stage, and report what happened; it may not decide
//! anything a number depends on. Where the window and the terminal disagree about a figure,
//! one of them is calling the wrong function.

use std::path::{Path, PathBuf};

use census_core::{
    CrawlOptions, DEFAULT_PACE, DEFAULT_SHARD_TARGET, SteamClient, ReviewQuery, embed, report,
};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};

/// Shards fetched at once. Pacing is global, so this reorders work rather than leaning
/// harder on Valve, and matches what the pipeline uses when nobody says otherwise.
const SHARDS_AT_ONCE: usize = 4;

/// Where corpora live when nobody has said.
///
/// The pipeline defaults to `data` beside the working directory, which is right for a
/// terminal and wrong for an icon: an application opened from a menu has no working
/// directory worth writing gigabytes into.
fn library_dir(app: &AppHandle) -> PathBuf {
    if let Some(chosen) = std::env::var_os("CENSUS_DATA") {
        return PathBuf::from(chosen);
    }
    app.path()
        .app_data_dir()
        .map_or_else(|_| PathBuf::from("data"), |dir| dir.join("data"))
}

fn text(error: impl std::fmt::Display) -> String {
    error.to_string()
}

/// One game as the library lists it.
#[derive(Debug, Clone, Serialize)]
struct Game {
    app_id: u32,
    name: String,
    reviews: u64,
    valve_total: u64,
    coverage: f64,
    verdict: String,
    snapshot: i64,
    stage: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct Shelf {
    path: String,
    games: Vec<Game>,
}

/// What Valve says about an app before a single review has been downloaded.
#[derive(Debug, Clone, Serialize)]
struct Found {
    app_id: u32,
    name: String,
    reviews: u64,
    positive: u64,
    negative: u64,
    verdict: String,
    /// Whether this game is already in the library, so the window can offer to continue a
    /// crawl rather than silently starting one that resumes.
    held: bool,
}

/// How far a game has been taken. Named after what exists on disk rather than what was
/// asked for, because an interrupted stage leaves the previous one intact.
fn stage_of(dir: &Path, app_id: u32) -> &'static str {
    let Ok(snapshot) = embed::latest_snapshot(dir, app_id) else {
        return "crawled";
    };
    if snapshot.join("classification.json").exists() {
        "classified"
    } else if snapshot.join("embeddings.parquet").exists() {
        "embedded"
    } else {
        "crawled"
    }
}

fn shelf(dir: &Path) -> Shelf {
    let mut games = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let Some(app_id) = entry
                .file_name()
                .to_str()
                .and_then(|name| name.strip_prefix("appid="))
                .and_then(|digits| digits.parse::<u32>().ok())
            else {
                continue;
            };
            let Ok(facts) = report::crawl_facts(dir, app_id) else {
                continue;
            };
            games.push(Game {
                app_id,
                name: facts.title(),
                reviews: facts.rows_unique,
                valve_total: facts.valve_total_reviews,
                coverage: facts.coverage,
                verdict: facts.review_score_desc.clone(),
                snapshot: facts.snapshot_unix,
                stage: stage_of(dir, app_id),
            });
        }
    }
    games.sort_by(|left, right| left.name.to_lowercase().cmp(&right.name.to_lowercase()));
    Shelf {
        path: dir.display().to_string(),
        games,
    }
}

#[tauri::command]
fn library(app: AppHandle) -> Shelf {
    shelf(&library_dir(&app))
}

#[tauri::command]
async fn look_up(app: AppHandle, app_id: u32) -> Result<Found, String> {
    let client = SteamClient::new(DEFAULT_PACE).map_err(text)?;
    let page = client
        .fetch(&ReviewQuery::new(app_id).per_page(0), app_id)
        .await
        .map_err(text)?;
    let summary = page
        .query_summary
        .ok_or_else(|| format!("Steam serves no reviews for app {app_id}"))?;
    let held = embed::latest_snapshot(&library_dir(&app), app_id).is_ok();
    Ok(Found {
        app_id,
        name: client.name(app_id).await.unwrap_or_default(),
        reviews: summary.total_reviews,
        positive: summary.total_positive,
        negative: summary.total_negative,
        verdict: summary.review_score_desc,
        held,
    })
}

/// How far a crawl has got, as the window draws it.
#[derive(Debug, Clone, Serialize)]
struct Step {
    app_id: u32,
    shards_done: usize,
    shards_total: usize,
    unique: u64,
    valve_total: u64,
}

#[tauri::command]
async fn crawl(app: AppHandle, app_id: u32) -> Result<Shelf, String> {
    let out_dir = library_dir(&app);
    std::fs::create_dir_all(&out_dir).map_err(text)?;
    let options = CrawlOptions {
        out_dir: out_dir.clone(),
        concurrency: SHARDS_AT_ONCE,
        shard_target: DEFAULT_SHARD_TARGET,
        resume: true,
        top_up: false,
    };
    let client = SteamClient::new(DEFAULT_PACE).map_err(text)?;
    let window = app.clone();
    census_core::crawl(&client, app_id, &options, move |progress| {
        let _ = window.emit(
            "crawl",
            Step {
                app_id,
                shards_done: progress.shards_done,
                shards_total: progress.shards_total,
                unique: progress.unique,
                valve_total: progress.valve_total,
            },
        );
    })
    .await
    .map_err(text)?;
    Ok(shelf(&out_dir))
}

/// One category, counted.
#[derive(Debug, Clone, Serialize)]
struct Topic {
    id: String,
    label: String,
    mentions: u64,
    primary: u64,
    rate: Option<f64>,
    top_rate: Option<f64>,
    bias: Option<f64>,
    positive: Option<f64>,
}

/// What a game's corpus says, as the window draws it.
#[derive(Debug, Clone, Serialize)]
struct Analysis {
    app_id: u32,
    name: String,
    reviews: u64,
    top_of_the_pile: u64,
    positive_baseline: Option<f64>,
    topics: Vec<Topic>,
}

#[tauri::command]
fn analysis(app: AppHandle, app_id: u32) -> Result<Analysis, String> {
    let options = report::ReportOptions {
        out_dir: library_dir(&app),
        ..report::ReportOptions::default()
    };
    let built = report::build(&[app_id], &options).map_err(text)?;
    let one = built
        .apps
        .first()
        .ok_or_else(|| "that game has not been counted yet".to_owned())?;

    let topics = one
        .classification
        .categories
        .iter()
        .map(|category| Topic {
            id: category.id.clone(),
            label: category.label.clone(),
            mentions: category.mention_count,
            primary: category.primary_count,
            rate: one.rate(category.mention_count),
            top_rate: one.top_rate(category.top_mention_count),
            bias: one.bias(category),
            positive: category.positive_share(),
        })
        .collect();

    Ok(Analysis {
        app_id,
        name: one.crawl.title(),
        reviews: one.classification.reviews,
        top_of_the_pile: one.classification.top_helpful,
        positive_baseline: one.positive_baseline(),
        topics,
    })
}

/// One review, as evidence for a number.
#[derive(Debug, Clone, Serialize)]
struct Quote {
    id: String,
    text: String,
    language: String,
    voted_up: bool,
    votes_up: u32,
    created: i64,
    url: Option<String>,
    primary: String,
    mentions: Vec<String>,
}

/// A window onto the reviews behind one figure.
#[derive(Debug, Clone, Serialize)]
struct Behind {
    category: String,
    total: u64,
    from: usize,
    quotes: Vec<Quote>,
}

/// Every review filed under a category, a page at a time.
///
/// This is what separates a rate from an assertion: the number on screen is a count of these,
/// and there is no figure anywhere in the window that cannot be opened into the reviews it
/// was counted from.
#[tauri::command]
fn behind(app: AppHandle, app_id: u32, category: String, from: usize, count: usize) -> Result<Behind, String> {
    let dir = library_dir(&app);
    let snapshot = embed::latest_snapshot(&dir, app_id).map_err(text)?;

    let mut total: u64 = 0;
    let mut filed: Vec<(String, String, Vec<String>)> = Vec::new();
    census_core::evaluate::for_each_assignment(
        &snapshot.join("classifications.parquet"),
        |id, primary, mentions| {
            if primary != category && !mentions.iter().any(|named| named == &category) {
                return;
            }
            total += 1;
            if total as usize > from && filed.len() < count {
                filed.push((id.to_owned(), primary.to_owned(), mentions.to_vec()));
            }
        },
    )
    .map_err(text)?;

    let wanted: std::collections::HashSet<String> =
        filed.iter().map(|(id, _, _)| id.clone()).collect();
    let mut fetched = census_core::capture::reviews_for(&snapshot, &wanted).map_err(text)?;

    let quotes = filed
        .into_iter()
        .filter_map(|(id, primary, mentions)| {
            let review = fetched.remove(&id)?;
            let url = (!review.author_steamid.is_empty()).then(|| {
                format!(
                    "https://steamcommunity.com/profiles/{}/recommended/{app_id}/",
                    review.author_steamid
                )
            });
            Some(Quote {
                id,
                text: review.text,
                language: review.language,
                voted_up: review.voted_up,
                votes_up: review.votes_up,
                created: review.created,
                url,
                primary,
                mentions,
            })
        })
        .collect();

    Ok(Behind {
        category,
        total,
        from,
        quotes,
    })
}

/// How far a reading has got, as the window draws it.
#[derive(Debug, Clone, Serialize)]
struct ReadStep {
    app_id: u32,
    done: u64,
    reading_claims: bool,
}

/// Reads a whole corpus with the trained model.
///
/// Runs off the window's thread: it is minutes of arithmetic on a million claims, and a
/// webview that stops answering is a webview a person force-quits.
#[tauri::command]
async fn read_game(app: AppHandle, app_id: u32, language: Option<String>) -> Result<(), String> {
    let out_dir = library_dir(&app);
    let options = census_core::read::ReadOptions {
        out_dir: out_dir.clone(),
        language,
        ..census_core::read::ReadOptions::default()
    };
    let window = app.clone();

    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        let mut model =
            census_core::reader::ClaimReader::load(&census_core::reader::default_dir())
                .map_err(text)?;
        let report =
            census_core::read::read_corpus(&mut model, app_id, &options, |progress| {
                let _ = window.emit(
                    "read",
                    ReadStep {
                        app_id,
                        done: progress.done,
                        reading_claims: progress.reading_claims,
                    },
                );
            })
            .map_err(text)?;
        let snapshot = embed::latest_snapshot(&options.out_dir, app_id).map_err(text)?;
        report.save(&snapshot.join("reading.json")).map_err(text)
    })
    .await
    .map_err(text)?
}

/// One subject, as the window draws it after a corpus has been read.
#[derive(Debug, Clone, Serialize)]
struct Subject {
    id: String,
    label: String,
    reviews: u64,
    rate: Option<f64>,
    praised: u64,
    criticised: u64,
    mixed: u64,
    claims: u64,
    top_rate: Option<f64>,
    bias: Option<f64>,
    positive: Option<f64>,
}

/// What reading a corpus found, ready for the window.
#[derive(Debug, Clone, Serialize)]
struct Reading {
    app_id: u32,
    name: String,
    reviews: u64,
    corpus_reviews: u64,
    language: Option<String>,
    claims: u64,
    unclassified_claims: u64,
    silent_reviews: u64,
    top_of_the_pile: u64,
    positive_baseline: Option<f64>,
    model: String,
    threshold: f32,
    subjects: Vec<Subject>,
}

#[expect(
    clippy::cast_precision_loss,
    reason = "review counts are far below 2^53"
)]
fn share_of(part: u64, whole: u64) -> Option<f64> {
    (whole > 0).then(|| part as f64 / whole as f64)
}

#[tauri::command]
fn reading(app: AppHandle, app_id: u32) -> Result<Reading, String> {
    let dir = library_dir(&app);
    let snapshot = embed::latest_snapshot(&dir, app_id).map_err(text)?;
    let found: census_core::read::ReadReport =
        serde_json::from_slice(&std::fs::read(snapshot.join("reading.json")).map_err(|_| {
            "this game has not been read yet; run the reading pass first".to_owned()
        })?)
        .map_err(text)?;
    let facts = report::crawl_facts(&dir, app_id).map_err(text)?;

    let subjects = found
        .subjects
        .iter()
        .map(|subject| {
            let rate = share_of(subject.mention_reviews, found.reviews);
            let top_rate = share_of(subject.top_mention_reviews, found.top_helpful);
            Subject {
                id: subject.id.clone(),
                label: subject.label.clone(),
                reviews: subject.mention_reviews,
                rate,
                praised: subject.praised,
                criticised: subject.criticised,
                mixed: subject.mixed,
                claims: subject.claims,
                top_rate,
                bias: match (rate, top_rate) {
                    (Some(overall), Some(top)) if overall > 0.0 => Some(top / overall),
                    _ => None,
                },
                positive: share_of(subject.positive_mentions, subject.mention_reviews),
            }
        })
        .collect();

    Ok(Reading {
        app_id,
        name: facts.title(),
        reviews: found.reviews,
        corpus_reviews: found.corpus_reviews,
        language: found.language.clone(),
        claims: found.claims,
        unclassified_claims: found.unclassified_claims,
        silent_reviews: found.silent_reviews,
        top_of_the_pile: found.top_helpful,
        positive_baseline: share_of(found.positive, found.reviews),
        model: found.model.clone(),
        threshold: found.threshold,
        subjects,
    })
}

/// One claim shown as evidence, with the review it came from.
#[derive(Debug, Clone, Serialize)]
struct Evidence {
    review_id: String,
    claim: String,
    polarity: String,
    confidence: f32,
    voted_up: bool,
    votes_up: u32,
    created: i64,
    language: String,
    url: Option<String>,
    /// The rest of the review, so a claim can be read where it was written.
    review: String,
}

#[derive(Debug, Clone, Serialize)]
struct ClaimsBehind {
    subject: String,
    total: u64,
    from: usize,
    claims: Vec<Evidence>,
}

/// Every claim filed under a subject, most helpful review first, a page at a time.
#[tauri::command]
fn claims_behind(
    app: AppHandle,
    app_id: u32,
    subject: String,
    from: usize,
    count: usize,
) -> Result<ClaimsBehind, String> {
    let dir = library_dir(&app);
    let snapshot = embed::latest_snapshot(&dir, app_id).map_err(text)?;

    let mut total: u64 = 0;
    let mut wanted: Vec<(String, u16, String, f32)> = Vec::new();
    census_core::read::for_each_reading(
        &snapshot.join("readings.parquet"),
        |id, index, found, confidence, polarity| {
            if found != Some(subject.as_str()) {
                return;
            }
            total += 1;
            if total as usize > from && wanted.len() < count {
                wanted.push((id.to_owned(), index, polarity.to_owned(), confidence));
            }
        },
    )
    .map_err(text)?;

    let ids: std::collections::HashSet<String> =
        wanted.iter().map(|(id, _, _, _)| id.clone()).collect();
    let mut fetched = census_core::capture::reviews_for(&snapshot, &ids).map_err(text)?;

    let claims = wanted
        .into_iter()
        .filter_map(|(id, index, polarity, confidence)| {
            let review = fetched.get(&id)?;
            let claim = census_core::claims::split(&review.text)
                .into_iter()
                .nth(index as usize)?
                .into_owned();
            let url = (!review.author_steamid.is_empty()).then(|| {
                format!(
                    "https://steamcommunity.com/profiles/{}/recommended/{app_id}/",
                    review.author_steamid
                )
            });
            Some(Evidence {
                review_id: id,
                claim,
                polarity,
                confidence,
                voted_up: review.voted_up,
                votes_up: review.votes_up,
                created: review.created,
                language: review.language.clone(),
                url,
                review: review.text.clone(),
            })
        })
        .collect();

    fetched.clear();

    Ok(ClaimsBehind {
        subject,
        total,
        from,
        claims,
    })
}

/// # Errors
///
/// Fails if the webview cannot be created, which on Linux means the system webview is
/// missing and on Windows means WebView2 is not installed.
pub fn run() -> anyhow::Result<()> {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            library,
            look_up,
            crawl,
            analysis,
            behind,
            reading,
            claims_behind,
            read_game
        ])
        .run(tauri::generate_context!())?;
    Ok(())
}
