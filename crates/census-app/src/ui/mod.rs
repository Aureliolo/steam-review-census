//! The window.
//!
//! Everything here is a thin front for the same core the pipeline runs. A command in this
//! module may read the library, start a stage, and report what happened; it may not decide
//! anything a number depends on. Where the window and the terminal disagree about a figure,
//! one of them is calling the wrong function.

use std::path::{Path, PathBuf};

use census_core::{
    CrawlOptions, DEFAULT_PACE, DEFAULT_SHARD_TARGET, ReviewQuery, SteamClient, embed, report,
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
    if snapshot.join("reading.json").exists() {
        "read"
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
    games.sort_by_key(|game| game.name.to_lowercase());
    Shelf {
        path: dir.display().to_string(),
        games,
    }
}

#[tauri::command]
#[expect(
    clippy::needless_pass_by_value,
    reason = "tauri hands a command its arguments by value"
)]
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

/// How far a sweep has got, as the window draws it.
#[derive(Debug, Clone, Serialize)]
struct SweepStep {
    app_id: u32,
    pages: u32,
    rows: u64,
}

/// What a sweep brought in, as the window reports it.
#[derive(Debug, Clone, Serialize)]
struct Swept {
    app_id: u32,
    rows: u64,
    new: u64,
    edited: u64,
    since: i64,
}

/// Brings a held capture up to date with what was written or edited since.
#[tauri::command]
async fn sweep(app: AppHandle, app_id: u32) -> Result<Swept, String> {
    let out_dir = library_dir(&app);
    let client = SteamClient::new(DEFAULT_PACE).map_err(text)?;
    let window = app.clone();
    let report = census_core::crawl::sweep(&client, app_id, &out_dir, move |progress| {
        let _ = window.emit(
            "sweep",
            SweepStep {
                app_id,
                pages: progress.pages,
                rows: progress.rows,
            },
        );
    })
    .await
    .map_err(text)?;
    Ok(Swept {
        app_id,
        rows: report.rows,
        new: report.new,
        edited: report.edited,
        since: report.watermark,
    })
}

/// How far a reading has got, as the window draws it.
#[derive(Debug, Clone, Serialize)]
struct ReadStep {
    app_id: u32,
    done: u64,
    reading_claims: bool,
}

/// How far a model download has got, as the window draws it.
#[derive(Debug, Clone, Serialize)]
struct Fetch {
    file: String,
    downloaded: u64,
    total: Option<u64>,
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

    // A standard user has the binary and nothing else. The model is fetched by checksum the
    // first time it is needed, and the window is told how far the download has got, because
    // half a gigabyte with no progress shown is indistinguishable from a hang.
    let model_dir = census_core::reader::default_dir();
    if !model_dir.join("model.onnx").is_file() {
        if !census_core::reader::PUBLISHED.is_pinned() {
            return Err("no claim reader is installed and none has been published yet".to_owned());
        }
        let fetching = app.clone();
        census_core::reader::ensure(&model_dir, |progress| {
            let _ = fetching.emit(
                "fetch",
                Fetch {
                    file: progress.file.to_owned(),
                    downloaded: progress.downloaded,
                    total: progress.total,
                },
            );
        })
        .await
        .map_err(text)?;
    }

    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        let mut model = census_core::reader::ClaimReader::load(&model_dir).map_err(text)?;
        let report = census_core::read::read_corpus(&mut model, app_id, &options, |progress| {
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
    /// What the share of claims would be with the model's measured errors taken out, where
    /// this game has labels and the model finds the subject better than chance.
    corrected: Option<f64>,
    /// Of the labelled claims about this subject, the share the model found. A subject it
    /// misses most of is a row whose rate is a floor, and the window marks it.
    found: Option<f64>,
    /// The words the praise uses and the complaints do not, and the reverse, each with the
    /// reviewers behind it. Empty where nothing stands out, which is what a thin side shows.
    praised_terms: Vec<census_core::said::Term>,
    criticised_terms: Vec<census_core::said::Term>,
}

/// How often the model is measured to be wrong on this game, where it has been measured.
#[derive(Debug, Clone, Serialize)]
struct Measured {
    answered: u64,
    declined: u64,
    agreement: Option<f64>,
    low: Option<f64>,
    high: Option<f64>,
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
    /// The game in a paragraph, assembled from the same counts the table shows.
    in_short: String,
    /// When the capture was last brought up to date after these counts were made, so the
    /// window can say the counts describe the corpus as it was.
    swept_since: Option<i64>,
    /// Whether this build takes reviews apart differently from the pass that made these
    /// readings. The counts stand; a claim quoted by its index does not, until the game is
    /// read again.
    older_splitter: bool,
    subjects: Vec<Subject>,
    measured: Option<Measured>,
    /// Oldest first. Fewer than two and there is no line to draw.
    months: Vec<MonthOut>,
    /// Commonest first, over the whole capture rather than the counted language.
    languages: Vec<Language>,
}

/// One month of the corpus, as the window draws it.
#[derive(Debug, Clone, Serialize)]
struct MonthOut {
    /// `2024-02`, which sorts.
    label: String,
    /// `February 2024`, which reads.
    name: String,
    reviews: u64,
    /// Share recommending the game, or none for a month too small to carry a rate.
    positive: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct Language {
    name: String,
    reviews: u64,
    share: Option<f64>,
}

#[expect(
    clippy::cast_precision_loss,
    reason = "review counts are far below 2^53"
)]
fn share_of(part: u64, whole: u64) -> Option<f64> {
    (whole > 0).then(|| part as f64 / whole as f64)
}

/// One subject's row, with its measured error where the game has labels for it.
fn subject_row(
    found: &census_core::read::ReadReport,
    subject: &census_core::read::SubjectCount,
    agreement: Option<&census_core::ClaimAgreement>,
) -> Subject {
    let rate = share_of(subject.mention_reviews, found.reviews);
    let top_rate = share_of(subject.top_mention_reviews, found.top_helpful);
    let measured = agreement
        .and_then(|found| found.subjects.iter().find(|s| s.id == subject.id))
        .filter(|s| s.labelled >= 10);
    let said = found.said.iter().find(|said| said.subject == subject.id);
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
        corrected: measured.and_then(|s| {
            share_of(subject.claims, found.claims).and_then(|observed| s.corrected(observed))
        }),
        found: measured.and_then(census_core::measure::SubjectAgreement::recall),
        praised_terms: said.map(|said| said.praised.clone()).unwrap_or_default(),
        criticised_terms: said.map(|said| said.criticised.clone()).unwrap_or_default(),
    }
}

/// What the reading pass wrote beside a snapshot's readings.
fn read_report(snapshot: &std::path::Path) -> Result<census_core::read::ReadReport, String> {
    serde_json::from_slice(
        &std::fs::read(snapshot.join("reading.json")).map_err(|_| {
            "this game has not been read yet; run the reading pass first".to_owned()
        })?,
    )
    .map_err(text)
}

#[tauri::command]
#[expect(
    clippy::needless_pass_by_value,
    reason = "tauri hands a command its arguments by value"
)]
fn reading(app: AppHandle, app_id: u32) -> Result<Reading, String> {
    let dir = library_dir(&app);
    let snapshot = embed::latest_snapshot(&dir, app_id).map_err(text)?;
    let found = read_report(&snapshot)?;
    let facts = report::crawl_facts(&dir, app_id).map_err(text)?;

    // The measurement, where this game has labelled claims. A game without them still
    // renders; it just cannot say how often it is wrong, and the window says that instead.
    let reference = census_core::claimset::default_reference_dir(app_id);
    let agreement = reference
        .join("labels.json")
        .is_file()
        .then(|| census_core::measure::agreement(&dir, app_id, &reference).ok())
        .flatten();
    let subjects = found
        .subjects
        .iter()
        .map(|subject| subject_row(&found, subject, agreement.as_ref()))
        .collect();

    let measured = agreement.as_ref().map(|found| {
        let interval = found.interval();
        Measured {
            answered: found.answered,
            declined: found.declined,
            agreement: found.rate(),
            low: interval.map(|(low, _)| low),
            high: interval.map(|(_, high)| high),
        }
    });

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
        in_short: census_core::picture::in_short(&found),
        swept_since: facts
            .swept_unix
            .filter(|swept| found.captured_unix < *swept),
        older_splitter: found.cut_as_this_build().is_err(),
        subjects,
        measured,
        months: found
            .months
            .iter()
            .map(|month| MonthOut {
                label: month.label.clone(),
                name: census_core::time::month_name(&month.label),
                reviews: month.reviews,
                positive: month.positive_share_if_enough(),
            })
            .collect(),
        languages: found
            .languages
            .iter()
            .map(|(name, reviews)| Language {
                name: name.clone(),
                reviews: *reviews,
                share: share_of(*reviews, found.corpus_reviews),
            })
            .collect(),
    })
}

/// A review quoted as it was written, with no reading attached, because the model that counts
/// the table was never asked about it.
#[derive(Debug, Clone, Serialize)]
struct Quoted {
    review_id: String,
    voted_up: bool,
    votes_up: u32,
    created: i64,
    language: String,
    url: Option<String>,
    review: String,
}

/// One subject found in this game's own reviews, with the reviews it rests on.
#[derive(Debug, Clone, Serialize)]
struct InducedOut {
    id: String,
    label: String,
    description: String,
    /// The label of the spine subject this is a form of, when it is one.
    refines: Option<String>,
    /// How many handout reviews it was found in, which is more than are quoted.
    found_in: usize,
    reviews: Vec<Quoted>,
}

/// What this game's players talk about that the fixed subjects do not name, where the
/// induction has been run. Separate from the reading because it walks the capture for the
/// quoted reviews, and a game is selected far more often than this section is read.
#[tauri::command]
#[expect(
    clippy::needless_pass_by_value,
    reason = "tauri hands a command its arguments by value"
)]
fn induced(app: AppHandle, app_id: u32) -> Result<Vec<InducedOut>, String> {
    let dir = library_dir(&app);
    let snapshot = embed::latest_snapshot(&dir, app_id).map_err(text)?;
    let found = report::induced_for(app_id, &snapshot, report::DEFAULT_EXAMPLES).map_err(text)?;
    Ok(found
        .into_iter()
        .map(|evidence| InducedOut {
            refines: evidence.subject.refines.as_deref().and_then(|id| {
                census_core::CORE_SPINE
                    .iter()
                    .find(|category| category.id == id)
                    .map(|category| category.label.to_owned())
            }),
            found_in: evidence.subject.evidence.len(),
            id: evidence.subject.id,
            label: evidence.subject.label,
            description: evidence.subject.description,
            reviews: evidence
                .reviews
                .into_iter()
                .map(|review| Quoted {
                    url: (!review.author_steamid.is_empty()).then(|| {
                        format!(
                            "https://steamcommunity.com/profiles/{}/recommended/{app_id}/",
                            review.author_steamid
                        )
                    }),
                    review_id: review.id,
                    voted_up: review.voted_up,
                    votes_up: review.votes_up,
                    created: review.created,
                    language: review.language,
                    review: review.text,
                })
                .collect(),
        })
        .collect())
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

/// A claim chosen for a page, before the review it belongs to has been fetched.
type Wanted = (String, u16, String, f32);

/// Every claim filed under a subject, most helpful review first, a page at a time.
///
/// Narrowed to one side when `side` is given, and to the claims using a term when `term` is,
/// which is how a word that stands out opens onto the reviews it was counted from.
#[tauri::command]
#[expect(
    clippy::needless_pass_by_value,
    reason = "tauri hands a command its arguments by value"
)]
fn claims_behind(
    app: AppHandle,
    app_id: u32,
    subject: String,
    side: Option<String>,
    term: Option<String>,
    from: usize,
    count: usize,
) -> Result<ClaimsBehind, String> {
    let dir = library_dir(&app);
    let snapshot = embed::latest_snapshot(&dir, app_id).map_err(text)?;
    // Taken apart the way the reading pass took it apart, so a claim index names the same
    // sentence here that it named when the model read it.
    let found = read_report(&snapshot)?;
    found.cut_as_this_build().map_err(|_| {
        "these counts were made with an older way of taking reviews apart, so the points \
         behind them cannot be shown; read this game again"
            .to_owned()
    })?;
    let depth = found.depth;

    let (total, wanted) = match term
        .as_deref()
        .map(str::trim)
        .filter(|term| !term.is_empty())
    {
        Some(term) => claims_using(
            &snapshot,
            depth,
            &subject,
            side.as_deref(),
            term,
            from,
            count,
        ),
        None => claims_under(&snapshot, &subject, side.as_deref(), from, count),
    }
    .map_err(text)?;

    let ids: std::collections::HashSet<String> =
        wanted.iter().map(|(id, _, _, _)| id.clone()).collect();
    let mut fetched = census_core::capture::reviews_for(&snapshot, &ids).map_err(text)?;

    let claims = wanted
        .into_iter()
        .filter_map(|(id, index, polarity, confidence)| {
            let review = fetched.get(&id)?;
            let claim = depth
                .claims_of(&review.text)
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

/// The page of claims under a subject, and how many there are, from the readings alone.
fn claims_under(
    snapshot: &std::path::Path,
    subject: &str,
    side: Option<&str>,
    from: usize,
    count: usize,
) -> census_core::Result<(u64, Vec<Wanted>)> {
    let mut total: u64 = 0;
    let mut wanted: Vec<Wanted> = Vec::new();
    census_core::read::for_each_reading(
        &snapshot.join("readings.parquet"),
        |id, index, found, confidence, polarity| {
            if found != Some(subject) || side.is_some_and(|side| side != polarity) {
                return;
            }
            total += 1;
            if total > from as u64 && wanted.len() < count {
                wanted.push((id.to_owned(), index, polarity.to_owned(), confidence));
            }
        },
    )?;
    Ok((total, wanted))
}

/// The page of claims under a subject that use a term, and how many there are.
///
/// The readings say which claims are filed here; only the capture says what they say. So the
/// readings are gathered by review first, and the capture is walked once, taking each listed
/// review apart the way the reading pass did and keeping the claims that use the term. One
/// walk per page rather than one per review, which is the difference between a click and a
/// wait on a corpus of a million reviews.
fn claims_using(
    snapshot: &std::path::Path,
    depth: census_core::read::Depth,
    subject: &str,
    side: Option<&str>,
    term: &str,
    from: usize,
    count: usize,
) -> census_core::Result<(u64, Vec<Wanted>)> {
    let mut filed: std::collections::HashMap<String, Vec<(u16, String, f32)>> =
        std::collections::HashMap::new();
    census_core::read::for_each_reading(
        &snapshot.join("readings.parquet"),
        |id, index, found, confidence, polarity| {
            if found != Some(subject) || side.is_some_and(|side| side != polarity) {
                return;
            }
            filed
                .entry(id.to_owned())
                .or_default()
                .push((index, polarity.to_owned(), confidence));
        },
    )?;

    let mut total: u64 = 0;
    let mut wanted: Vec<Wanted> = Vec::new();
    census_core::capture::for_each_row(snapshot, |row, text| {
        let Some(claims) = filed.get(&row.recommendationid) else {
            return Ok(());
        };
        let split = depth.claims_of(text);
        for (index, polarity, confidence) in claims {
            let uses = split
                .get(usize::from(*index))
                .is_some_and(|claim| census_core::said::mentions(claim, term));
            if !uses {
                continue;
            }
            total += 1;
            if total > from as u64 && wanted.len() < count {
                wanted.push((
                    row.recommendationid.clone(),
                    *index,
                    polarity.clone(),
                    *confidence,
                ));
            }
        }
        Ok(())
    })?;
    Ok((total, wanted))
}

/// # Errors
///
/// Fails if the webview cannot be created, which on Linux means the system webview is
/// missing and on Windows means `WebView2` is not installed.
pub fn run() -> anyhow::Result<()> {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            library,
            look_up,
            crawl,
            reading,
            claims_behind,
            induced,
            read_game,
            sweep
        ])
        .run(tauri::generate_context!())?;
    Ok(())
}
