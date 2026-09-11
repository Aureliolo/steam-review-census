//! Gathering everything a readable report needs, without re-reading the corpus.
//!
//! The counts come from the sidecar the reading pass writes, so a report never recomputes a
//! pass that has already run and can never disagree with the run it claims to describe.
//!
//! What the sidecar cannot hold is review text. A report that says 14% of reviews complain
//! about performance is a claim a reader should be able to check, so a bounded, deterministic
//! handful of the claims behind every number is fetched from the capture and shown. Bounded
//! because a million reviews will not fit in a page; deterministic because two runs against
//! the same corpus should show the same evidence.
//!
//! The evidence is claims rather than whole reviews, because a claim is what the count is made
//! of. Quoting the review would ask a reader to find the sentence that earned the subject, and
//! on a long review that sentence is one of thirty.

use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
};

use serde::Deserialize;

use crate::{Result, bounded::Smallest, capture::CapturedReview, taxonomy::CORE_SPINE};

/// Reviews shown per category before a reader is asked to go to the corpus itself.
pub const DEFAULT_EXAMPLES: usize = 8;

/// Of those, how many are taken from the most-helpful reviews rather than at random.
const FROM_THE_TOP: usize = 2;

/// A category has to reach this fraction of the top of the pile before the headline will
/// quote its bias. One in ten, so a fifty-review top needs five of them.
const HEADLINE_TOP_SHARE: u64 = 10;

/// A share of Valve's own total, which is the one figure anywhere in this tool that a reader
/// is entitled to read as a claim of completeness.
///
/// Rounding is a courtesy everywhere else and a lie here: a capture that reached all but a
/// hundred of a million reviews is not complete, and printing 100% says it is. Shared with
/// the crawler's own summary, which was rounding the same figure up in the same way.
#[must_use]
pub fn coverage(share: f64) -> String {
    if share < 1.0 && (share * 10_000.0).round() >= 10_000.0 {
        return ">99.99%".to_owned();
    }
    format!("{:.2}%", share * 100.0)
}

#[derive(Debug, Clone)]
pub struct ReportOptions {
    pub out_dir: PathBuf,
    /// Reviews quoted per category. The first two come from the top of the pile where the
    /// category reaches it, so a category's evidence is not made entirely of reviews nobody
    /// ever read.
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
    /// When the capture was last brought up to date, where it has been.
    #[serde(default)]
    pub swept_unix: Option<i64>,
    #[serde(default)]
    pub sweeps: u64,
    /// Rows the sweeps added: reviews that arrived since the crawl and reviews edited since.
    #[serde(default)]
    pub rows_swept: u64,
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

    /// When the capture last changed: its last sweep, or the crawl where there has been none.
    #[must_use]
    pub fn changed_unix(&self) -> i64 {
        self.swept_unix.unwrap_or(self.snapshot_unix)
    }
}

/// One claim shown as evidence, with the review it was made in.
///
/// The claim rather than the review is what a subject's count is made of, so it is what a
/// reader checking the count has to be shown. The review comes with it because a claim reading
/// "it doesn't" means nothing alone, and because the link back to Steam belongs to the review.
#[derive(Debug, Clone)]
pub struct Example {
    pub review: CapturedReview,
    /// The words the model read, which is one point from the review and not all of it.
    pub claim: String,
    /// Which point of the review this is, counting from zero.
    pub index: u16,
    pub polarity: String,
    /// How sure the model was. Shown, because a claim scraping past the threshold and one the
    /// model is certain of are not equally good evidence and should not look it.
    pub confidence: f32,
    /// Every subject this review raises, so a quoted claim sits in the context of what else
    /// its author said rather than looking like their only point.
    pub also: Vec<String>,
    /// Whether this review is one of the most-helpful in the corpus.
    pub from_the_top: bool,
}

impl Example {
    /// Whether the model read this. An induced subject's evidence is whole reviews a
    /// language model picked out, and the counting model has never seen them; they carry no
    /// polarity and no confidence, and the page must not invent either.
    #[must_use]
    pub fn was_read(&self) -> bool {
        !self.polarity.is_empty()
    }

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
    pub reading: crate::read::ReadReport,
    /// Evidence per subject id, in taxonomy order.
    pub examples: Vec<(String, Vec<Example>)>,
    /// The most-helpful reviews, which is what a reader skimming the store page sees.
    pub top: Vec<Example>,
    /// Measured agreement against a reference set, or why there is none.
    pub agreement: Measurement,
    /// What this game's players talk about that the spine has no row for, where a reading
    /// has induced any, with the reviews behind each.
    pub induced: Vec<InducedEvidence>,
}

/// One induced subject with the reviews that earned it, fetched so the page can quote them.
#[derive(Debug, Clone)]
pub struct InducedEvidence {
    pub subject: crate::induced::Induced,
    pub reviews: Vec<CapturedReview>,
}

/// Whether a game's error has been measured, and what stopped it if not.
///
/// A game with a reference set labelled against another taxonomy is not a game nobody has
/// labelled, and a page that says it is tells a reader to go and do work that has been done.
/// Every taxonomy change puts every game in that state until its set is labelled again, so it
/// is a state the report spends real time in rather than an edge case.
#[derive(Debug, Clone, Default)]
pub enum Measurement {
    /// Nobody has labelled this game.
    #[default]
    Unlabelled,
    /// A set exists, naming a taxonomy this build does not have. Scoring against it would
    /// mark the model on subjects nobody labelling it was offered.
    OtherTaxonomy(String),
    /// Boxed because the report behind it dwarfs the other two and every game carries one.
    Measured(Box<crate::measure::ClaimAgreement>),
}

impl Measurement {
    /// The report, where there is one.
    #[must_use]
    pub fn report(&self) -> Option<&crate::measure::ClaimAgreement> {
        match self {
            Self::Measured(report) => Some(report),
            _ => None,
        }
    }
}

impl AppReport {
    /// Share of the whole corpus that recommended the game, which is the line every
    /// subject's own share should be read against.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "review counts are far below 2^53"
    )]
    pub fn positive_baseline(&self) -> Option<f64> {
        // Counted over the reviews that were read rather than over Valve's own totals, which
        // include reviews with no text at all. A baseline drawn from a different population
        // than the shares it is compared against is worse than no baseline.
        (self.reading.reviews > 0)
            .then(|| self.reading.positive as f64 / self.reading.reviews as f64)
    }

    #[must_use]
    pub const fn app_id(&self) -> u32 {
        self.reading.app_id
    }

    /// The subject whose share of the top of the pile most overstates its share of the
    /// corpus, which is the single number this whole tool exists to produce.
    ///
    /// Only subjects a real part of the top of the pile actually discusses are eligible.
    /// The top of the pile is a few dozen reviews, so one review mentioning something the
    /// corpus almost never mentions produces an enormous ratio out of a count of one, and
    /// leading a report with that would be reporting noise as a finding.
    #[must_use]
    pub fn worst_bias(&self) -> Option<(&crate::read::SubjectCount, f64)> {
        let floor = self.reading.top_helpful.div_ceil(HEADLINE_TOP_SHARE);
        self.reading
            .subjects
            .iter()
            .filter(|s| s.mention_reviews > 0 && s.top_mention_reviews >= floor.max(2))
            .filter_map(|s| self.bias(s).map(|factor| (s, factor)))
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
    }

    /// How much the top of the pile overstates a subject, or `None` when nobody raises it.
    #[must_use]
    pub fn bias(&self, subject: &crate::read::SubjectCount) -> Option<f64> {
        let overall = self.rate(subject.mention_reviews)?;
        let top = self.top_rate(subject.top_mention_reviews)?;
        (overall > 0.0).then_some(top / overall)
    }

    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "review counts are far below 2^53"
    )]
    pub fn rate(&self, part: u64) -> Option<f64> {
        (self.reading.reviews > 0).then(|| part as f64 / self.reading.reviews as f64)
    }

    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "the top of the pile is a few dozen reviews"
    )]
    pub fn top_rate(&self, part: u64) -> Option<f64> {
        (self.reading.top_helpful > 0).then(|| part as f64 / self.reading.top_helpful as f64)
    }
}

/// A whole report, one section per game.
#[derive(Debug, Clone)]
pub struct Report {
    pub apps: Vec<AppReport>,
    /// When the report was rendered, which is not when the corpus was captured.
    pub generated_unix: i64,
}

/// One category counted over every game in a report at once.
#[derive(Debug, Clone, Copy)]
pub struct Pooled {
    pub id: &'static str,
    pub label: &'static str,
    pub top_mentions: u64,
    pub top_reviews: u64,
    pub mentions: u64,
    pub reviews: u64,
}

impl Pooled {
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "review counts are far below 2^53"
    )]
    pub fn rate(&self) -> Option<f64> {
        (self.reviews > 0).then(|| self.mentions as f64 / self.reviews as f64)
    }

    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "the top of the pile is a few dozen reviews a game"
    )]
    pub fn top_rate(&self) -> Option<f64> {
        (self.top_reviews > 0).then(|| self.top_mentions as f64 / self.top_reviews as f64)
    }

    #[must_use]
    pub fn bias(&self) -> Option<f64> {
        let overall = self.rate()?;
        let top = self.top_rate()?;
        (overall > 0.0).then_some(top / overall)
    }
}

impl Report {
    /// Every category counted over every game, so a rate is about the whole set of corpora.
    #[must_use]
    pub fn pooled(&self) -> Vec<Pooled> {
        crate::CORE_SPINE
            .iter()
            .map(|category| {
                let mut pooled = Pooled {
                    id: category.id,
                    label: category.label,
                    top_mentions: 0,
                    top_reviews: 0,
                    mentions: 0,
                    reviews: 0,
                };
                for app in &self.apps {
                    // A game that never had this subject counted still contributes its
                    // reviews to the denominator: it is a game where nobody raised it, not a
                    // game that was not asked.
                    pooled.reviews += app.reading.reviews;
                    pooled.top_reviews += app.reading.top_helpful;
                    if let Some(count) = app.reading.subjects.iter().find(|s| s.id == category.id) {
                        pooled.mentions += count.mention_reviews;
                        pooled.top_mentions += count.top_mention_reviews;
                    }
                }
                pooled
            })
            .collect()
    }

    /// The category the top of the pile overstates most, over every game at once.
    ///
    /// The same claim each game's own section opens with, and the one this tool exists to
    /// make, except that a single corpus can always be answered with "that is just that
    /// game". Made of every game in the report, it cannot be.
    ///
    /// The eligibility floor is the per-game one scaled to the pooled top of the pile: a
    /// category raised by two of two thousand most-helpful reviews produces an enormous ratio
    /// out of a count of two, and leading with that would be reporting noise as a finding.
    #[must_use]
    pub fn worst_bias(&self) -> Option<(Pooled, f64)> {
        self.pooled()
            .into_iter()
            .filter(|c| {
                c.mentions > 0 && c.top_mentions >= c.top_reviews.div_ceil(HEADLINE_TOP_SHARE)
            })
            .filter_map(|c| c.bias().map(|factor| (c, factor)))
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
    }
}

/// Builds a report for every named app.
///
/// # Errors
///
/// Fails if a capture, its readings, or the sidecar the reading pass writes are missing.
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

/// What the crawl recorded about a game's most recent capture, without reading the corpus.
///
/// The name lives here and nowhere else, so anything that wants to call a game by its name
/// rather than by nine digits reads it from the capture.
///
/// # Errors
///
/// Fails if there is no capture, or its crawl record is missing or malformed.
pub fn crawl_facts(out_dir: &Path, app_id: u32) -> Result<CrawlFacts> {
    let snapshot = crate::embed::latest_snapshot(out_dir, app_id)?;
    read_json(&snapshot.join("crawl.json"))
}

fn build_one(app_id: u32, options: &ReportOptions) -> Result<AppReport> {
    let snapshot = crate::embed::latest_snapshot(&options.out_dir, app_id)?;
    let reading: crate::read::ReadReport = read_json(&snapshot.join("reading.json"))?;
    let crawl = crawl_facts(&options.out_dir, app_id)?;

    // A reading made against a different taxonomy counts subjects this build does not have,
    // and one made by a model that has since been replaced describes answers nothing here
    // holds any more. Either would render perfectly and be wrong.
    if reading.spine_version != crate::CORE_SPINE_VERSION {
        return Err(crate::Error::StaleAnchors {
            field: "taxonomy",
            expected: crate::CORE_SPINE_VERSION.to_owned(),
            actual: reading.spine_version.clone(),
        });
    }

    let drawn = shortlist(&snapshot, options)?;
    let top_ids: HashSet<String> = top_of_the_pile(&snapshot, reading.top_helpful)?;
    let mut wanted: HashSet<String> = drawn
        .values()
        .flat_map(|claims| claims.iter().map(|one| one.review_id.clone()))
        .collect();
    wanted.extend(top_ids.iter().cloned());

    let fetched = crate::capture::reviews_for(&snapshot, &wanted)?;

    // Which subjects each quoted review raises anywhere, so a claim can be shown next to the
    // rest of its author's point rather than as though it were all they said.
    let mut raised: HashMap<String, Vec<String>> = HashMap::new();
    crate::read::for_each_reading(
        &snapshot.join("readings.parquet"),
        |id, _, subject, _, _| {
            if let (Some(subject), true) = (subject, wanted.contains(id)) {
                let all = raised.entry(id.to_owned()).or_default();
                if !all.iter().any(|seen| seen == subject) {
                    all.push(subject.to_owned());
                }
            }
        },
    )?;

    let quote = |claim: &DrawnClaim| -> Option<Example> {
        let review = fetched.get(&claim.review_id)?;
        Some(Example {
            // Taken apart the way the reading pass took it apart, or the index names a
            // different sentence from the one that was read.
            claim: reading
                .depth
                .claims_of(&review.text)
                .into_iter()
                .nth(claim.index as usize)?
                .into_owned(),
            index: claim.index,
            polarity: claim.polarity.clone(),
            confidence: claim.confidence,
            also: raised.get(&claim.review_id).cloned().unwrap_or_default(),
            from_the_top: top_ids.contains(&claim.review_id),
            review: review.clone(),
        })
    };

    let examples = CORE_SPINE
        .iter()
        .map(|subject| {
            let mut quoted: Vec<Example> = Vec::new();
            // The top of the pile first, where the subject reaches it at all. A random sample
            // of a million reviews will almost never contain one of the fifty most-helpful,
            // and what those fifty said about a subject next to what everyone said is the
            // whole argument in miniature.
            let drawn_here = drawn.get(subject.id).map(Vec::as_slice).unwrap_or_default();
            for claim in drawn_here
                .iter()
                .filter(|claim| top_ids.contains(&claim.review_id))
                .take(FROM_THE_TOP)
                .chain(drawn_here.iter())
            {
                if quoted.len() >= options.examples {
                    break;
                }
                if quoted
                    .iter()
                    .any(|shown| shown.review.id == claim.review_id && shown.index == claim.index)
                {
                    continue;
                }
                if let Some(example) = quote(claim) {
                    quoted.push(example);
                }
            }
            (subject.id.to_owned(), quoted)
        })
        .collect();

    let mut top: Vec<Example> = drawn
        .values()
        .flatten()
        .filter(|claim| top_ids.contains(&claim.review_id))
        .filter_map(quote)
        .collect();
    top.sort_by_key(|example| std::cmp::Reverse(example.review.votes_up));
    top.dedup_by(|a, b| a.review.id == b.review.id);

    let agreement = agreement_for(app_id, &options.out_dir);
    let induced = induced_for(app_id, &snapshot, options.examples)?;
    Ok(AppReport {
        crawl,
        reading,
        examples,
        top,
        agreement,
        induced,
    })
}

/// A game's induced subjects with a few of the reviews behind each, or nothing.
///
/// A game nobody has induced subjects for renders without the section, and the page says so
/// in one line rather than leaving an empty heading.
/// The subjects induced for a game, each with the first few reviews it rests on fetched from
/// the capture. Empty where the induction has not been run.
///
/// # Errors
///
/// Fails if the induced set is unreadable or the capture cannot be walked.
pub fn induced_for(app_id: u32, snapshot: &Path, examples: usize) -> Result<Vec<InducedEvidence>> {
    let Some(set) = crate::induced::load(&crate::induced::default_path(app_id))? else {
        return Ok(Vec::new());
    };
    let wanted: HashSet<String> = set
        .subjects
        .iter()
        .flat_map(|subject| subject.evidence.iter().take(examples).cloned())
        .collect();
    let fetched = crate::capture::reviews_for(snapshot, &wanted)?;
    Ok(set
        .subjects
        .into_iter()
        .map(|subject| {
            let reviews = subject
                .evidence
                .iter()
                .take(examples)
                .filter_map(|id| fetched.get(id).cloned())
                .collect();
            InducedEvidence { subject, reviews }
        })
        .collect())
}

/// The ids of the most-helpful reviews in a capture, as Steam ranks them.
///
/// The reading pass counted these by the same rule, so a report and the counts it renders
/// agree on which reviews are "the top of the pile" without either storing a list.
fn top_of_the_pile(snapshot: &Path, how_many: u64) -> Result<HashSet<String>> {
    let wanted = usize::try_from(how_many).unwrap_or(usize::MAX);
    let mut ranked: Smallest<std::cmp::Reverse<u64>, String> = Smallest::new(wanted);
    crate::capture::for_each_row(snapshot, |row, _| {
        // The same key the reading pass ranks by, bit for bit, so the two cannot disagree
        // about which reviews the top of the pile contains.
        ranked.offer(
            std::cmp::Reverse(row.helpfulness.to_bits()),
            row.recommendationid,
        );
        Ok(())
    })?;
    Ok(ranked.take().into_iter().collect())
}

/// Picks which claims to quote for each subject, without holding the corpus.
///
/// Ranked by a hash of the review id and claim index, so the choice depends on the claim
/// rather than on where it happened to sit in the file, and a corpus that gains reviews does
/// not reshuffle the evidence already shown.
///
/// Drawn per polarity, not per subject. A subject that is nine parts praise to one part
/// complaint would otherwise show nine praises and, often, no complaint at all, and the
/// complaint is what a reader opening the row came to see. So each side gets its own draw and
/// the shown evidence says what was said on each, in the proportions the counts beside it
/// already give.
///
/// More are drawn than will be shown. A drawn claim can turn out to be unquotable, because
/// the splitter that produced the readings and the splitter in this build can disagree about
/// how many points a review makes, and a subject left with nothing to show is a count nobody
/// can check.
fn shortlist(snapshot: &Path, options: &ReportOptions) -> Result<HashMap<String, Vec<DrawnClaim>>> {
    let draw = options.examples * 2;
    let mut per_side: HashMap<(&'static str, &'static str), Smallest<[u8; 32], DrawnClaim>> =
        HashMap::new();
    for subject in CORE_SPINE {
        for side in crate::taxonomy::POLARITY {
            per_side.insert((subject.id, side), Smallest::new(draw));
        }
    }

    crate::read::for_each_reading(
        &snapshot.join("readings.parquet"),
        |id, index, subject, confidence, polarity| {
            let Some(subject) = subject.and_then(|id| CORE_SPINE.iter().find(|c| c.id == id))
            else {
                return;
            };
            let Some(side) = crate::taxonomy::POLARITY
                .iter()
                .find(|side| **side == polarity)
            else {
                return;
            };
            if let Some(keep) = per_side.get_mut(&(subject.id, *side)) {
                keep.offer(
                    crate::bounded::rank(options.seed, "report", &format!("{id}:{index}")),
                    DrawnClaim {
                        review_id: id.to_owned(),
                        index,
                        polarity: polarity.to_owned(),
                        confidence,
                    },
                );
            }
        },
    )?;

    // Interleaved praise, complaint, neutral, so taking the first N of a subject's list gives
    // every side that has anything a turn before any side gets a second.
    let mut per_subject: HashMap<String, Vec<DrawnClaim>> = HashMap::new();
    for subject in CORE_SPINE {
        let mut sides: Vec<std::vec::IntoIter<DrawnClaim>> = crate::taxonomy::POLARITY
            .iter()
            .filter_map(|side| per_side.remove(&(subject.id, side)))
            .map(|keep| keep.take().into_iter())
            .collect();
        let mut merged = Vec::new();
        loop {
            let mut any = false;
            for side in &mut sides {
                if let Some(claim) = side.next() {
                    merged.push(claim);
                    any = true;
                }
            }
            if !any {
                break;
            }
        }
        per_subject.insert(subject.id.to_owned(), merged);
    }
    Ok(per_subject)
}

/// A shortlisted claim, before the review it belongs to has been fetched.
#[derive(Debug, Clone)]
struct DrawnClaim {
    review_id: String,
    index: u16,
    polarity: String,
    confidence: f32,
}

/// A game's measured agreement, when it has a claim reference set and stored readings to
/// compare. A report without one still renders; it just cannot say how often it is wrong.
fn agreement_for(app_id: u32, out_dir: &Path) -> Measurement {
    let dir = crate::claimset::default_reference_dir(app_id);
    let Ok(labels) = std::fs::read(dir.join("labels.json")) else {
        return Measurement::Unlabelled;
    };
    let Ok(first) = serde_json::from_slice::<Vec<crate::claimset::ClaimLabel>>(&labels) else {
        return Measurement::Unlabelled;
    };
    if first.is_empty() {
        return Measurement::Unlabelled;
    }
    crate::measure::agreement(out_dir, app_id, &dir).map_or(Measurement::Unlabelled, |report| {
        Measurement::Measured(Box::new(report))
    })
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

    /// The one number this tool reports that a reader may read as a claim of completeness,
    /// so it is the one number that must never be rounded into one.
    #[test]
    fn a_capture_that_missed_something_never_claims_all_of_it() {
        assert_eq!(coverage(1.0), "100.00%");
        assert_eq!(
            coverage(0.999_99),
            ">99.99%",
            "ten reviews short of a million"
        );
        assert_eq!(coverage(0.999_995), ">99.99%");
        assert_eq!(
            coverage(0.999_91),
            "99.99%",
            "an honest figure needs no hedging"
        );
        assert_eq!(coverage(0.994), "99.40%");
        assert_eq!(coverage(0.5), "50.00%");
    }

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
                swept_unix: None,
                sweeps: 0,
                rows_swept: 0,
            },
            reading: crate::read::ReadReport {
                app_id: 1,
                reviews: 100_000,
                corpus_reviews: 100_000,
                language: None,
                depth: crate::read::Depth::Deep,
                claims: 300_000,
                unclassified_claims: 0,
                silent_reviews: 0,
                positive: 70_000,
                top_helpful,
                model: "test".to_owned(),
                trained_on: String::new(),
                usual_declined: None,
                spine_version: crate::CORE_SPINE_VERSION.to_owned(),
                threshold: 0.5,
                device: "cpu".to_owned(),
                captured_unix: 0,
                subjects: categories
                    .into_iter()
                    .map(|(id, mentions, top)| crate::read::SubjectCount {
                        id: id.to_owned(),
                        label: id.to_owned(),
                        mention_reviews: mentions,
                        primary_reviews: mentions,
                        claims: mentions,
                        praised: mentions / 2,
                        criticised: mentions / 4,
                        mixed: 0,
                        top_mention_reviews: top,
                        positive_mentions: mentions / 2,
                    })
                    .collect(),
                said: Vec::new(),
                languages: Vec::new(),
                months: Vec::new(),
                elapsed: std::time::Duration::ZERO,
            },
            examples: Vec::new(),
            top: Vec::new(),
            agreement: Measurement::Unlabelled,
            induced: Vec::new(),
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
    fn a_subject_nobody_raises_has_no_bias_rather_than_an_infinite_one() {
        let report = app(50, vec![("absent", 0, 0)]);
        assert!(report.worst_bias().is_none());
        assert_eq!(report.bias(&report.reading.subjects[0]), None);
    }
}
