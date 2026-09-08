//! Gathering everything a readable report needs, without re-reading the corpus.
//!
//! The counts come from the sidecar `census classify` writes, so a report never recomputes a
//! pass that has already run and can never disagree with the run it claims to describe.
//!
//! What the sidecar cannot hold is review text. A report that says 14% of reviews complain
//! about performance is a claim a reader should be able to check, so a bounded, deterministic
//! handful of the reviews behind every number is fetched from the capture and shown. Bounded
//! because a million reviews will not fit in a page; deterministic because two runs against
//! the same corpus should show the same evidence.

use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
};

use serde::Deserialize;

use crate::{Result, bounded::Smallest, capture::CapturedReview, taxonomy::CORE_SPINE};

/// Reviews shown per category before a reader is asked to go to the corpus itself.
pub const DEFAULT_EXAMPLES: usize = 8;

/// A category has to reach this fraction of the top of the pile before the headline will
/// quote its bias. One in ten, so a fifty-review top needs five of them.
const HEADLINE_TOP_SHARE: u64 = 10;

#[derive(Debug, Clone)]
pub struct ReportOptions {
    pub out_dir: PathBuf,
    /// Reviews quoted per category. Two are drawn from the top of the pile where possible,
    /// so a category's examples are not all reviews nobody read.
    pub examples: usize,
    /// Changing this quotes different reviews. The same seed always quotes the same ones.
    pub seed: u64,
}

impl Default for ReportOptions {
    fn default() -> Self {
        Self {
            out_dir: PathBuf::from("data"),
            examples: DEFAULT_EXAMPLES,
            seed: 1,
        }
    }
}

/// What `census classify` recorded beside the assignments.
#[derive(Debug, Clone, Deserialize)]
pub struct Classification {
    pub app_id: u32,
    pub reviews: u64,
    /// Classified reviews that recommended the game. Absent from sidecars written before
    /// the classifier counted it.
    #[serde(default)]
    pub positive: u64,
    pub unmatched: u64,
    pub top_helpful: u64,
    pub mention_margin: f32,
    pub spine_version: String,
    pub model: String,
    #[serde(default)]
    pub anchors_fitted_from: Vec<u32>,
    pub categories: Vec<CategoryCount>,
    /// Reviews per language, most common first.
    #[serde(default)]
    pub languages: Vec<(String, u64)>,
    #[serde(default)]
    pub top_reviews: Vec<TopRow>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CategoryCount {
    pub id: String,
    pub label: String,
    pub primary_count: u64,
    pub mention_count: u64,
    pub top_mention_count: u64,
    /// Of the reviews mentioning this, how many still recommended the game. Absent from
    /// sidecars written before the classifier counted it.
    #[serde(default)]
    pub positive_mentions: u64,
}

impl CategoryCount {
    /// Share of the reviews mentioning this category that recommended the game.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "review counts are far below 2^53"
    )]
    pub fn positive_share(&self) -> Option<f64> {
        (self.mention_count > 0).then(|| self.positive_mentions as f64 / self.mention_count as f64)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TopRow {
    pub id: String,
    pub helpfulness: f64,
    pub votes_up: u32,
    pub primary: String,
    pub mentions: Vec<String>,
}

/// What the crawl recorded about the capture a report describes.
#[derive(Debug, Clone, Deserialize)]
pub struct CrawlFacts {
    pub app_id: u32,
    /// The store's name for the app. Captures made before the crawler asked for it have
    /// none, and a report falls back to the id rather than inventing one.
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub review_score_desc: String,
    #[serde(default)]
    pub rows_unique: u64,
    #[serde(default)]
    pub valve_total_reviews: u64,
    #[serde(default)]
    pub valve_total_positive: u64,
    #[serde(default)]
    pub valve_total_negative: u64,
    #[serde(default)]
    pub coverage: f64,
    #[serde(default)]
    pub snapshot_unix: i64,
    #[serde(default)]
    pub shards: u64,
}

impl CrawlFacts {
    /// What to call this game on screen.
    #[must_use]
    pub fn title(&self) -> String {
        if self.name.is_empty() {
            format!("App {}", self.app_id)
        } else {
            self.name.clone()
        }
    }
}

/// One review shown as evidence, with the categories it was filed under.
#[derive(Debug, Clone)]
pub struct Example {
    pub review: CapturedReview,
    pub primary: String,
    pub mentions: Vec<String>,
    /// Whether this review is one of the most-upvoted in the corpus.
    pub from_the_top: bool,
}

impl Example {
    /// Where this review lives on Steam, so any quoted claim can be checked at the source.
    #[must_use]
    pub fn url(&self, app_id: u32) -> Option<String> {
        (!self.review.author_steamid.is_empty()).then(|| {
            format!(
                "https://steamcommunity.com/profiles/{}/recommended/{app_id}/",
                self.review.author_steamid
            )
        })
    }
}

/// Everything one game contributes to a report.
#[derive(Debug, Clone)]
pub struct AppReport {
    pub crawl: CrawlFacts,
    pub classification: Classification,
    /// Evidence per category id, in taxonomy order.
    pub examples: Vec<(String, Vec<Example>)>,
    /// The most-upvoted reviews, which is what a reader skimming the store page sees.
    pub top: Vec<Example>,
    /// Measured agreement against a reference set, where one exists for this game.
    pub agreement: Option<crate::AgreementReport>,
}

impl AppReport {
    /// Share of the whole corpus that recommended the game, which is the line every
    /// category's own share should be read against.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "review counts are far below 2^53"
    )]
    pub fn positive_baseline(&self) -> Option<f64> {
        // Counted over the classified reviews rather than over Valve's own totals, which
        // include reviews with no text at all. A baseline drawn from a different population
        // than the shares it is compared against is worse than no baseline.
        (self.classification.reviews > 0)
            .then(|| self.classification.positive as f64 / self.classification.reviews as f64)
    }
}

impl AppReport {
    #[must_use]
    pub fn app_id(&self) -> u32 {
        self.classification.app_id
    }

    /// The category whose share of the top of the pile most overstates its share of the
    /// corpus, which is the single number this whole tool exists to produce.
    ///
    /// Only categories a real part of the top of the pile actually discusses are eligible.
    /// The top of the pile is a few dozen reviews, so one review mentioning something the
    /// corpus almost never mentions produces an enormous ratio out of a count of one, and
    /// leading a report with that would be reporting noise as a finding.
    #[must_use]
    pub fn worst_bias(&self) -> Option<(&CategoryCount, f64)> {
        let floor = self.classification.top_helpful.div_ceil(HEADLINE_TOP_SHARE);
        self.classification
            .categories
            .iter()
            .filter(|c| c.mention_count > 0 && c.top_mention_count >= floor.max(2))
            .filter_map(|c| self.bias(c).map(|factor| (c, factor)))
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
    }

    /// How much the top of the pile overstates a category, or `None` when nobody mentions it.
    #[must_use]
    pub fn bias(&self, category: &CategoryCount) -> Option<f64> {
        let overall = self.rate(category.mention_count)?;
        let top = self.top_rate(category.top_mention_count)?;
        (overall > 0.0).then_some(top / overall)
    }

    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "review counts are far below 2^53"
    )]
    pub fn rate(&self, part: u64) -> Option<f64> {
        (self.classification.reviews > 0).then(|| part as f64 / self.classification.reviews as f64)
    }

    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "the top of the pile is a few dozen reviews"
    )]
    pub fn top_rate(&self, part: u64) -> Option<f64> {
        (self.classification.top_helpful > 0)
            .then(|| part as f64 / self.classification.top_helpful as f64)
    }
}

/// A whole report, one section per game.
#[derive(Debug, Clone)]
pub struct Report {
    pub apps: Vec<AppReport>,
    /// When the report was rendered, which is not when the corpus was captured.
    pub generated_unix: i64,
}

/// Builds a report for every named app.
///
/// # Errors
///
/// Fails if a capture, its classifications, or the sidecar `census classify` writes are
/// missing.
pub fn build(app_ids: &[u32], options: &ReportOptions) -> Result<Report> {
    let mut apps = Vec::with_capacity(app_ids.len());
    for &app_id in app_ids {
        apps.push(build_one(app_id, options)?);
    }
    Ok(Report {
        apps,
        generated_unix: now_unix(),
    })
}

fn build_one(app_id: u32, options: &ReportOptions) -> Result<AppReport> {
    let snapshot = crate::embed::latest_snapshot(&options.out_dir, app_id)?;
    let classification: Classification = read_json(&snapshot.join("classification.json"))?;
    let crawl: CrawlFacts = read_json(&snapshot.join("crawl.json"))?;

    // Re-embedding a corpus and forgetting to classify it again leaves counts that describe
    // vectors nothing here holds any more. They would render perfectly and be wrong.
    let corpus_encoder = crate::embed::corpus_encoder(&options.out_dir, app_id)?;
    if corpus_encoder != classification.model {
        return Err(crate::Error::StaleAnchors {
            field: "embedding model",
            expected: corpus_encoder,
            actual: classification.model,
        });
    }

    let wanted_per_category = shortlist(&snapshot, options)?;
    let mut wanted: HashSet<String> = wanted_per_category
        .values()
        .flat_map(|filed| filed.iter().map(|one| one.id.clone()))
        .collect();
    let top_ids: HashSet<String> = classification
        .top_reviews
        .iter()
        .map(|row| row.id.clone())
        .collect();
    wanted.extend(top_ids.iter().cloned());

    let fetched = crate::capture::reviews_for(&snapshot, &wanted)?;

    let examples = CORE_SPINE
        .iter()
        .map(|category| {
            let quoted = wanted_per_category
                .get(category.id)
                .map(Vec::as_slice)
                .unwrap_or_default()
                .iter()
                .filter_map(|filed| {
                    Some(Example {
                        review: fetched.get(&filed.id)?.clone(),
                        primary: filed.primary.clone(),
                        mentions: filed.mentions.clone(),
                        from_the_top: top_ids.contains(&filed.id),
                    })
                })
                .collect();
            (category.id.to_owned(), quoted)
        })
        .collect();

    let top = classification
        .top_reviews
        .iter()
        .filter_map(|row| {
            Some(Example {
                review: fetched.get(&row.id)?.clone(),
                primary: row.primary.clone(),
                mentions: row.mentions.clone(),
                from_the_top: true,
            })
        })
        .collect();

    let agreement = agreement_for(app_id, &options.out_dir);
    Ok(AppReport {
        crawl,
        classification,
        examples,
        top,
        agreement,
    })
}

/// Picks which reviews to quote for each category, without holding the corpus.
///
/// Ranked by a hash of the review id, so the choice depends on the review rather than on
/// where it happened to sit in the file, and a corpus that gains reviews does not reshuffle
/// the evidence already shown. Mentions rather than main subjects: a category that is rarely
/// what a review is *about* but often something it *says* would otherwise have almost
/// nothing to show.
fn shortlist(snapshot: &Path, options: &ReportOptions) -> Result<HashMap<String, Vec<Filed>>> {
    let mut per_category: HashMap<&'static str, Smallest<[u8; 32], Filed>> = CORE_SPINE
        .iter()
        .map(|category| (category.id, Smallest::new(options.examples)))
        .collect();

    crate::evaluate::for_each_assignment(
        &snapshot.join("classifications.parquet"),
        |id, primary, mentions| {
            for mention in mentions {
                if let Some(keep) = per_category.get_mut(mention.as_str()) {
                    keep.offer(
                        crate::sample::rank(options.seed, "report", id),
                        Filed {
                            id: id.to_owned(),
                            primary: primary.to_owned(),
                            mentions: mentions.to_vec(),
                        },
                    );
                }
            }
        },
    )?;

    Ok(per_category
        .into_iter()
        .map(|(id, keep)| (id.to_owned(), keep.take()))
        .collect())
}

/// A shortlisted review and everything the classifier filed it under, so a quoted review can
/// show all its categories rather than only the one whose panel it happens to be in.
struct Filed {
    id: String,
    primary: String,
    mentions: Vec<String>,
}

/// A game's measured agreement, when it has a reference set and stored classifications to
/// compare. A report without one still renders; it just cannot say how often it is wrong.
fn agreement_for(app_id: u32, out_dir: &Path) -> Option<crate::AgreementReport> {
    let dir = crate::evaluate::default_reference_dir(app_id);
    let set = crate::ReferenceSet::load(&dir).ok()?;
    (set.spine_version == crate::CORE_SPINE_VERSION)
        .then(|| crate::compare(&set, out_dir, app_id).ok())
        .flatten()
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = std::fs::read(path).map_err(|_| crate::Error::NoClassifications {
        path: path.to_path_buf(),
    })?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app(top_helpful: u64, categories: Vec<(&str, u64, u64)>) -> AppReport {
        AppReport {
            crawl: CrawlFacts {
                app_id: 1,
                name: String::new(),
                review_score_desc: String::new(),
                rows_unique: 0,
                valve_total_reviews: 0,
                valve_total_positive: 0,
                valve_total_negative: 0,
                coverage: 1.0,
                snapshot_unix: 0,
                shards: 1,
            },
            classification: Classification {
                app_id: 1,
                reviews: 100_000,
                positive: 70_000,
                unmatched: 0,
                top_helpful,
                mention_margin: 0.01,
                spine_version: "core-3".to_owned(),
                model: "test".to_owned(),
                anchors_fitted_from: Vec::new(),
                categories: categories
                    .into_iter()
                    .map(|(id, mentions, top)| CategoryCount {
                        id: id.to_owned(),
                        label: id.to_owned(),
                        primary_count: mentions,
                        mention_count: mentions,
                        top_mention_count: top,
                        positive_mentions: mentions / 2,
                    })
                    .collect(),
                languages: Vec::new(),
                top_reviews: Vec::new(),
            },
            examples: Vec::new(),
            top: Vec::new(),
            agreement: None,
        }
    }

    #[test]
    fn a_single_review_at_the_top_never_becomes_the_headline() {
        // One of fifty is 2%, and 2% against a corpus rate of 0.02% is a hundredfold "bias"
        // built on one person. The report should lead with the finding that survives that
        // review being a fluke.
        let report = app(
            50,
            vec![
                ("fluke", 20, 1),
                ("real", 3_000, 12),
                ("common", 40_000, 20),
            ],
        );
        let (category, factor) = report.worst_bias().unwrap();
        assert_eq!(category.id, "real");
        assert!((factor - 8.0).abs() < 1e-9, "factor was {factor}");
    }

    #[test]
    fn a_top_of_the_pile_too_small_to_divide_still_reports_something() {
        // Five reviews cannot supply a tenth each, so the floor falls back to two rather
        // than excluding every category and leaving the report with no finding at all.
        let report = app(5, vec![("one", 10_000, 2), ("two", 50_000, 3)]);
        assert_eq!(report.worst_bias().unwrap().0.id, "one");
    }

    #[test]
    fn a_category_nobody_mentions_has_no_bias_rather_than_an_infinite_one() {
        let report = app(50, vec![("absent", 0, 0)]);
        assert!(report.worst_bias().is_none());
        assert_eq!(report.bias(&report.classification.categories[0]), None);
    }
}
