//! Drawing the reviews that a reference set will be built from.
//!
//! The first reference set was sampled by hand, and the manifest records what that cost:
//! labellers resolved the same boundary two different ways in near-identical cases because
//! nothing about the process was fixed. Sampling is the half of that which a program can
//! own, so it does.
//!
//! Two subsets come out, and they are disjoint on purpose.
//!
//! **Random** is drawn uniformly from the corpus and is the only subset any figure may be
//! quoted from. It is what the corpus actually looks like.
//!
//! **Stratified** is drawn per predicted category and deliberately over-represents whatever
//! the classifier rarely picks, because those categories are exactly the ones with too few
//! examples to anchor. Quoting a rate from it would describe the sampling rather than the
//! game. A review drawn into the random subset is never also offered to the stratified one,
//! or the held-out subset would be training data by another name.
//!
//! Selection is deterministic given a seed: candidates are ranked by the hash of the seed
//! and the review id, and the top ones win. That makes a reference set reproducible from the
//! corpus and the seed alone, with no stored random state, and makes re-sampling after a
//! re-crawl keep the reviews that still exist rather than reshuffling everything.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    Error, Result,
    evaluate::ReferenceLabel,
    taxonomy::{CONFIDENCE, CORE_SPINE, CORE_SPINE_VERSION},
};

/// How many reviews to draw, and how to spread them.
#[derive(Debug, Clone, Copy)]
pub struct SampleOptions {
    /// Uniformly drawn reviews per app. The measuring subset.
    pub random_per_app: usize,
    /// Reviews per category, pooled across every app being sampled. The anchoring subset.
    pub stratified_per_category: usize,
    pub seed: u64,
}

impl Default for SampleOptions {
    fn default() -> Self {
        Self {
            random_per_app: 100,
            stratified_per_category: 40,
            seed: 1,
        }
    }
}

/// One review offered for labelling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampledReview {
    pub id: String,
    pub app_id: u32,
    /// `random` or `stratified`. Recorded in the label so a report can never blend the two.
    pub subset: String,
    /// What the classifier called it. Present so the stratification is auditable, and
    /// deliberately not shown to whoever labels: a labeller told the answer agrees with it.
    pub predicted: String,
    pub review: String,
}

/// What a sampling run drew, per app.
#[derive(Debug, Clone)]
pub struct SampleReport {
    pub app_id: u32,
    pub corpus: u64,
    pub random: usize,
    pub stratified: usize,
    /// Stratified counts per category, so a category nobody can supply is visible up front.
    pub per_category: Vec<(&'static str, usize)>,
}

/// Ranks a review deterministically for a given seed and purpose.
///
/// Hashing rather than shuffling means the choice depends only on the review, so a corpus
/// that gains reviews does not renumber the ones already labelled.
#[must_use]
pub fn rank(seed: u64, purpose: &str, id: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(seed.to_le_bytes());
    hasher.update(purpose.as_bytes());
    hasher.update(id.as_bytes());
    hasher.finalize().into()
}

struct Candidate {
    id: String,
    app_id: u32,
    predicted: String,
}

/// Selection wants the few hundred reviews with the lowest hashes out of a corpus of
/// millions, which [`crate::bounded::Smallest`] answers without holding the corpus.
type BestByKey = crate::bounded::Smallest<[u8; 32], Candidate>;

impl Candidate {
    fn into(self, subset: &str) -> SampledReview {
        SampledReview {
            id: self.id,
            app_id: self.app_id,
            subset: subset.to_owned(),
            predicted: self.predicted,
            review: String::new(),
        }
    }

    fn clone_into_sample(&self, subset: &str) -> SampledReview {
        SampledReview {
            id: self.id.clone(),
            app_id: self.app_id,
            subset: subset.to_owned(),
            predicted: self.predicted.clone(),
            review: String::new(),
        }
    }
}

/// Draws a reference sample across one or more classified corpora.
///
/// # Errors
///
/// Fails if any app's capture or classifications cannot be read.
pub fn draw(
    out_dir: &Path,
    app_ids: &[u32],
    options: &SampleOptions,
) -> Result<(Vec<SampledReview>, Vec<SampleReport>)> {
    let mut chosen: Vec<SampledReview> = Vec::new();
    let mut leftovers: Vec<([u8; 32], Candidate)> = Vec::new();
    let mut reports: Vec<SampleReport> = Vec::new();

    // A category can hold at most this many candidates from one app before the worst of them
    // can no longer matter. The random subset is drawn first and removed from the stratified
    // pool afterwards, so carrying that much slack guarantees enough survivors to fill the
    // category however the two overlap.
    let per_app_slack = options
        .stratified_per_category
        .saturating_add(options.random_per_app);

    for &app_id in app_ids {
        let snapshot = crate::embed::latest_snapshot(out_dir, app_id)?;
        let path = snapshot.join("classifications.parquet");

        // Bounded from the first row. A corpus of a million reviews yields a few hundred
        // here, so materialising it to sort it costs hundreds of megabytes to discard
        // essentially all of them.
        let mut random = BestByKey::new(options.random_per_app);
        let mut by_category: HashMap<&'static str, BestByKey> = HashMap::new();
        let mut corpus = 0_u64;

        crate::evaluate::for_each_prediction(&path, |id, predicted| {
            corpus += 1;
            let Some(category) = CORE_SPINE.iter().find(|c| c.id == predicted) else {
                return;
            };
            random.offer(
                rank(options.seed, "random", id),
                Candidate {
                    id: id.to_owned(),
                    app_id,
                    predicted: category.id.to_owned(),
                },
            );
            by_category
                .entry(category.id)
                .or_insert_with(|| BestByKey::new(per_app_slack))
                .offer(
                    rank(options.seed, "stratified", id),
                    Candidate {
                        id: id.to_owned(),
                        app_id,
                        predicted: category.id.to_owned(),
                    },
                );
        })?;

        let drawn = random.take();
        let taken: HashSet<String> = drawn.iter().map(|c| c.id.clone()).collect();
        let count = drawn.len();
        chosen.extend(drawn.into_iter().map(|candidate| candidate.into("random")));
        for keep in by_category.into_values() {
            leftovers.extend(
                keep.take_with_keys()
                    .into_iter()
                    .filter(|(_, c)| !taken.contains(&c.id)),
            );
        }

        reports.push(SampleReport {
            app_id,
            corpus,
            random: count,
            stratified: 0,
            per_category: Vec::new(),
        });
    }

    let per_app_category = stratify(&leftovers, options.stratified_per_category, &mut chosen);
    for report in &mut reports {
        report.stratified = chosen
            .iter()
            .filter(|r| r.app_id == report.app_id && r.subset == "stratified")
            .count();
        report.per_category = CORE_SPINE
            .iter()
            .map(|c| {
                let count = per_app_category
                    .get(&(report.app_id, c.id))
                    .copied()
                    .unwrap_or(0);
                (c.id, count)
            })
            .collect();
    }

    attach_texts(out_dir, app_ids, &mut chosen)?;
    chosen.retain(|r| !r.review.trim().is_empty());
    Ok((chosen, reports))
}

/// What a labeller returns for one review, before the sample's own fields are put back.
///
/// Every judgement but `secondary` is optional here so that leaving one out is something
/// [`ingest`] can report, rather than something serde turns into a default several hundred
/// reviews at a time. An empty `secondary` and an omitted one do mean the same thing: a
/// review about one subject takes one category and nothing else.
#[derive(Debug, Clone, Deserialize)]
pub struct ReturnedLabel {
    pub id: String,
    pub primary: String,
    #[serde(default)]
    pub secondary: Vec<String>,
    pub ironic: Option<bool>,
    pub confidence: Option<String>,
    pub ambiguous: Option<bool>,
}

/// What ingesting a set of returned labels produced, and everything wrong with it.
#[derive(Debug, Clone, Default)]
pub struct IngestReport {
    pub accepted: usize,
    /// Reviews drawn for labelling that nobody returned a label for.
    pub missing: Vec<String>,
    /// Labels naming a review that was never drawn, usually a batch pasted into the wrong app.
    pub unknown_reviews: Vec<String>,
    /// Labels naming a category the taxonomy does not have.
    pub unknown_categories: Vec<String>,
    /// Reviews labelled more than once, kept at the first label seen.
    pub duplicates: Vec<String>,
    /// Labels that came back without one of the judgements every label carries, or with a
    /// confidence that is not one of the words the sheet asks for. They are dropped: a
    /// judgement nobody made is not a judgement, and `ambiguous` in particular decides how
    /// agreement is reported, so defaulting it to false would invent the answer.
    pub unjudged: Vec<String>,
}

impl IngestReport {
    /// Whether anything was wrong enough that the set should not be used as it stands.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.missing.is_empty()
            && self.unknown_reviews.is_empty()
            && self.unknown_categories.is_empty()
            && self.duplicates.is_empty()
            && self.unjudged.is_empty()
    }
}

/// Writes a reference set's labels to `labels.json`.
///
/// # Errors
///
/// Fails if the file cannot be written.
pub fn write_labels(dir: &Path, labels: &[ReferenceLabel]) -> Result<std::path::PathBuf> {
    let path = dir.join("labels.json");
    std::fs::write(&path, serde_json::to_vec_pretty(labels)?)?;
    Ok(path)
}

/// A reference set's manifest as it is written back, with everything else carried through.
///
/// Only the taxonomy is the tool's to write. Who produced the labels, whether a person has
/// checked them and what the set is for are provenance, and provenance a program fills in for
/// itself is provenance nobody wrote down.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Manifest {
    app_id: u32,
    produced_by: String,
    spine_version: String,
    #[serde(default)]
    human_verified: bool,
    #[serde(flatten)]
    rest: serde_json::Map<String, serde_json::Value>,
}

/// Records in the manifest which taxonomy these labels were checked against.
///
/// The version had to be typed by hand beside labels a program wrote, and the two drift in
/// both directions: a manifest still naming the old spine makes a fresh set unusable, and one
/// updated ahead of the labels makes a stale set look current, which is worse because
/// everything downstream then runs. `ingest` is the only place that sees the labels and the
/// taxonomy at the same moment, so it is the place that can say.
///
/// # Errors
///
/// Fails if the manifest cannot be read, parsed or written.
pub fn record_taxonomy(dir: &Path, app_id: u32) -> Result<String> {
    let path = dir.join("manifest.json");
    let mut manifest = match std::fs::read(&path) {
        Ok(bytes) => serde_json::from_slice::<Manifest>(&bytes)?,
        Err(_) => Manifest {
            app_id,
            produced_by: String::new(),
            spine_version: String::new(),
            human_verified: false,
            rest: serde_json::Map::new(),
        },
    };
    let was = std::mem::replace(&mut manifest.spine_version, CORE_SPINE_VERSION.to_owned());
    std::fs::write(&path, serde_json::to_vec_pretty(&manifest)?)?;
    Ok(was)
}

/// Merges returned labels into a reference set, checking them against the drawn sample.
///
/// Labelling happens outside this program, so what comes back is unverified input: ids that
/// were never drawn, categories that do not exist, reviews labelled twice by overlapping
/// batches, reviews quietly skipped. Each of those corrupts a measurement in a way that is
/// invisible later, so all four are checked here and reported rather than assumed away.
///
/// `subset` is taken from the sample rather than from the labeller, so no labeller can move
/// a review between the training and measuring halves.
///
/// # Errors
///
/// Fails if the sample or a label file cannot be read or parsed.
pub fn ingest(
    reference_dir: &Path,
    labels_dir: &Path,
) -> Result<(Vec<ReferenceLabel>, IngestReport)> {
    let drawn: Vec<SampledReview> = serde_json::from_slice(
        &std::fs::read(reference_dir.join("sample.json")).map_err(|_| Error::NoReferenceSet {
            path: reference_dir.join("sample.json"),
        })?,
    )?;
    let subsets: HashMap<&str, &str> = drawn
        .iter()
        .map(|r| (r.id.as_str(), r.subset.as_str()))
        .collect();
    let known: HashSet<&str> = CORE_SPINE.iter().map(|c| c.id).collect();

    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(labels_dir)?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();

    let mut report = IngestReport::default();
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<ReferenceLabel> = Vec::new();

    for file in files {
        let returned: Vec<ReturnedLabel> = serde_json::from_slice(&std::fs::read(&file)?)?;
        for label in returned {
            let Some(subset) = subsets.get(label.id.as_str()) else {
                report.unknown_reviews.push(label.id);
                continue;
            };
            if !seen.insert(label.id.clone()) {
                report.duplicates.push(label.id);
                continue;
            }
            let judged = label
                .confidence
                .filter(|word| CONFIDENCE.contains(&word.as_str()))
                .zip(label.ironic)
                .zip(label.ambiguous);
            let Some(((confidence, ironic), ambiguous)) = judged else {
                report.unjudged.push(label.id);
                continue;
            };
            for named in std::iter::once(&label.primary).chain(&label.secondary) {
                if !known.contains(named.as_str()) {
                    report.unknown_categories.push(named.clone());
                }
            }
            out.push(ReferenceLabel {
                id: label.id,
                primary: label.primary,
                secondary: label.secondary,
                ironic,
                confidence,
                subset: (*subset).to_owned(),
                ambiguous,
            });
        }
    }

    report.missing = drawn
        .iter()
        .filter(|r| !seen.contains(&r.id))
        .map(|r| r.id.clone())
        .collect();
    report.unknown_categories.sort_unstable();
    report.unknown_categories.dedup();
    report.accepted = out.len();
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok((out, report))
}

/// One review as it is handed to a labeller.
///
/// Carries the id and the text and nothing else. The classifier's own guess is deliberately
/// withheld: a labeller shown a proposed answer agrees with it far more often than one
/// reading the review cold, and a reference set that inherits the classifier's mistakes
/// cannot measure them.
///
/// Which game it is stays out for a different reason, and it costs something real: a review
/// of a headset game saying only that it is immersive is unlabelable without knowing the
/// game, and a labeller who knew would file it correctly. But the classifier reads the text
/// and nothing else, so a label made from more than that measures the difference in what the
/// two were shown rather than how well the classifier reads.
#[derive(Debug, Clone, Serialize)]
pub struct LabellingItem<'a> {
    pub id: &'a str,
    pub review: &'a str,
}

/// Splits one app's sample into fixed-size batches ready to hand out.
///
/// # Errors
///
/// Fails if the directory cannot be created or a batch cannot be written.
pub fn write_batches(
    dir: &Path,
    app_id: u32,
    drawn: &[SampledReview],
    batch_size: usize,
) -> Result<usize> {
    let mine: Vec<&SampledReview> = drawn.iter().filter(|r| r.app_id == app_id).collect();
    if mine.is_empty() || batch_size == 0 {
        return Ok(0);
    }
    let batches = dir.join("batches");
    std::fs::create_dir_all(&batches)?;

    let mut written = 0;
    for (index, chunk) in mine.chunks(batch_size).enumerate() {
        let items: Vec<LabellingItem<'_>> = chunk
            .iter()
            .map(|review| LabellingItem {
                id: &review.id,
                review: &review.review,
            })
            .collect();
        std::fs::write(
            batches.join(format!("batch-{index:03}.json")),
            serde_json::to_vec_pretty(&items)?,
        )?;
        written += 1;
    }
    Ok(written)
}

/// Writes one app's drawn reviews to `sample.json` in its reference directory.
///
/// # Errors
///
/// Fails if the directory cannot be created or the file cannot be written.
pub fn write_for_app(dir: &Path, app_id: u32, drawn: &[SampledReview]) -> Result<usize> {
    let mine: Vec<&SampledReview> = drawn.iter().filter(|r| r.app_id == app_id).collect();
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("sample.json"), serde_json::to_vec_pretty(&mine)?)?;
    Ok(mine.len())
}

/// Fills each category by taking one candidate from each app in turn.
///
/// Round-robin rather than best-ranked-first because one game can easily hold every
/// candidate for a category, and a category anchored on a single game learns that game's
/// vocabulary for it and nothing more general.
fn stratify<'a>(
    leftovers: &'a [([u8; 32], Candidate)],
    target: usize,
    chosen: &mut Vec<SampledReview>,
) -> HashMap<(u32, &'a str), usize> {
    let mut counts: HashMap<(u32, &str), usize> = HashMap::new();

    for category in CORE_SPINE {
        let mut by_app: HashMap<u32, Vec<&Candidate>> = HashMap::new();
        let mut candidates: Vec<&([u8; 32], Candidate)> = leftovers
            .iter()
            .filter(|(_, c)| c.predicted == category.id)
            .collect();
        candidates.sort_unstable_by_key(|(key, _)| *key);
        for (_, candidate) in candidates {
            by_app.entry(candidate.app_id).or_default().push(candidate);
        }
        let mut apps: Vec<u32> = by_app.keys().copied().collect();
        apps.sort_unstable();
        let mut picked = 0;
        for round in 0.. {
            let mut advanced = false;
            for app_id in &apps {
                if picked >= target {
                    break;
                }
                let Some(candidate) = by_app.get(app_id).and_then(|v| v.get(round)) else {
                    continue;
                };
                advanced = true;
                picked += 1;
                *counts.entry((*app_id, category.id)).or_default() += 1;
                chosen.push(candidate.clone_into_sample("stratified"));
            }
            if !advanced || picked >= target {
                break;
            }
        }
    }
    counts
}

/// Reads back only the reviews that were drawn.
///
/// Holding a million review bodies in memory in order to choose a few hundred of them would
/// cost gigabytes to no purpose, so selection works on ids and text arrives afterwards.
fn attach_texts(out_dir: &Path, app_ids: &[u32], chosen: &mut [SampledReview]) -> Result<()> {
    for &app_id in app_ids {
        let ids: HashSet<String> = chosen
            .iter()
            .filter(|r| r.app_id == app_id)
            .map(|r| r.id.clone())
            .collect();
        if ids.is_empty() {
            continue;
        }
        let snapshot = crate::embed::latest_snapshot(out_dir, app_id)?;
        let texts = crate::capture::texts_for(&snapshot, &ids)?;
        for review in chosen.iter_mut().filter(|r| r.app_id == app_id) {
            if let Some(text) = texts.get(&review.id) {
                review.review.clone_from(text);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranking_is_stable_for_a_review_and_changes_with_the_seed() {
        assert_eq!(rank(1, "random", "abc"), rank(1, "random", "abc"));
        assert_ne!(rank(1, "random", "abc"), rank(2, "random", "abc"));
        assert_ne!(rank(1, "random", "abc"), rank(1, "stratified", "abc"));
    }

    #[test]
    fn the_two_purposes_order_the_same_reviews_differently() {
        // If they agreed, the stratified subset would be drawn from the reviews the random
        // subset most nearly took, rather than independently of it.
        let ids: Vec<String> = (0..200).map(|i| format!("review-{i}")).collect();
        let mut by_random: Vec<&String> = ids.iter().collect();
        by_random.sort_by_key(|id| rank(7, "random", id));
        let mut by_stratified: Vec<&String> = ids.iter().collect();
        by_stratified.sort_by_key(|id| rank(7, "stratified", id));
        assert_ne!(by_random, by_stratified);
    }

    /// `ambiguous` decides whether a review's disagreement is reported against the classifier
    /// or against the taxonomy, and `ironic` is a claim about what the text does. Filling
    /// either in for a labeller who did not answer puts a figure on the page that nobody
    /// stood behind.
    #[test]
    fn a_judgement_nobody_made_is_refused_rather_than_filled_in() {
        let root = std::env::temp_dir().join("census-ingest-judgements");
        let reference = root.join("reference");
        let returned = root.join("returned");
        std::fs::create_dir_all(&reference).unwrap();
        std::fs::create_dir_all(&returned).unwrap();
        std::fs::write(
            reference.join("sample.json"),
            r#"[{"app_id":7,"id":"a","subset":"random","predicted":"bugs","review":"x"},
                {"app_id":7,"id":"b","subset":"random","predicted":"bugs","review":"y"},
                {"app_id":7,"id":"c","subset":"random","predicted":"bugs","review":"z"}]"#,
        )
        .unwrap();
        std::fs::write(
            returned.join("batch-000.json"),
            r#"[{"id":"a","primary":"bugs","ironic":false,"confidence":"high","ambiguous":true},
                {"id":"b","primary":"bugs","ironic":false,"confidence":"high"},
                {"id":"c","primary":"bugs","ironic":false,"confidence":"fairly sure",
                 "ambiguous":false}]"#,
        )
        .unwrap();

        let (labels, report) = ingest(&reference, &returned).unwrap();
        std::fs::remove_dir_all(&root).ok();

        assert_eq!(report.accepted, 1, "an unjudged label was kept");
        assert_eq!(
            report.unjudged,
            vec!["b".to_owned(), "c".to_owned()],
            "a missing judgement and an invented confidence were both let through"
        );
        assert!(
            !report.is_clean(),
            "a set whose judgements went missing was called clean"
        );
        assert!(labels[0].ambiguous, "the judgement that was made was lost");
    }

    /// The taxonomy a set was labelled against decides whether fitting from it is allowed at
    /// all, and it was typed by hand beside labels a program wrote. Everything else in the
    /// manifest says who produced the labels, which no program is in a position to claim.
    #[test]
    fn recording_the_taxonomy_leaves_the_provenance_beside_it_alone() {
        let dir = std::env::temp_dir().join("census-manifest-taxonomy");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("manifest.json"),
            r#"{"app_id":7,"produced_by":"three people and an argument",
                "spine_version":"core-1","human_verified":true,
                "notes":["the one thing nobody should lose"]}"#,
        )
        .unwrap();

        let was = record_taxonomy(&dir, 7).unwrap();
        let written = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(was, "core-1", "the version it replaced is not reported");
        let back: serde_json::Value = serde_json::from_str(&written).unwrap();
        assert_eq!(back["spine_version"], CORE_SPINE_VERSION);
        assert_eq!(back["produced_by"], "three people and an argument");
        assert_eq!(back["human_verified"], true);
        assert_eq!(back["notes"][0], "the one thing nobody should lose");
    }

    #[test]
    fn defaults_hold_back_more_reviews_for_measuring_than_any_one_category_gets() {
        // The random subset is the only thing a figure may be quoted from, so it should not
        // be the smaller half of the set by accident.
        let options = SampleOptions::default();
        assert!(options.random_per_app > options.stratified_per_category);
    }
}
