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

use crate::{Result, taxonomy::CORE_SPINE};

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
fn rank(seed: u64, purpose: &str, id: &str) -> [u8; 32] {
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
    key: [u8; 32],
}

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
    let mut leftovers: Vec<Candidate> = Vec::new();
    let mut reports: Vec<SampleReport> = Vec::new();

    for &app_id in app_ids {
        let snapshot = crate::embed::latest_snapshot(out_dir, app_id)?;
        let predictions =
            crate::evaluate::primary_categories(&snapshot.join("classifications.parquet"))?;
        let corpus = predictions.len() as u64;

        let mut pool: Vec<Candidate> = predictions
            .into_iter()
            .map(|(id, predicted)| Candidate {
                key: rank(options.seed, "random", &id),
                id,
                app_id,
                predicted,
            })
            .collect();
        pool.sort_unstable_by_key(|candidate| candidate.key);

        let split = options.random_per_app.min(pool.len());
        let rest = pool.split_off(split);
        chosen.extend(pool.into_iter().map(|candidate| candidate.into("random")));
        leftovers.extend(rest.into_iter().map(|mut candidate| {
            candidate.key = rank(options.seed, "stratified", &candidate.id);
            candidate
        }));

        reports.push(SampleReport {
            app_id,
            corpus,
            random: split,
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
    leftovers: &'a [Candidate],
    target: usize,
    chosen: &mut Vec<SampledReview>,
) -> HashMap<(u32, &'a str), usize> {
    let mut counts: HashMap<(u32, &str), usize> = HashMap::new();

    for category in CORE_SPINE {
        let mut by_app: HashMap<u32, Vec<&Candidate>> = HashMap::new();
        let mut candidates: Vec<&Candidate> = leftovers
            .iter()
            .filter(|c| c.predicted == category.id)
            .collect();
        candidates.sort_unstable_by_key(|candidate| candidate.key);
        for candidate in candidates {
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

    #[test]
    fn defaults_hold_back_more_reviews_for_measuring_than_any_one_category_gets() {
        // The random subset is the only thing a figure may be quoted from, so it should not
        // be the smaller half of the set by accident.
        let options = SampleOptions::default();
        assert!(options.random_per_app > options.stratified_per_category);
    }
}
