//! The vectors a review is compared against.
//!
//! The zero-setup classifier embeds a *written description* of each category and takes the
//! nearest one. That description is prose about a topic, and reviews about that topic are
//! not prose about it: "runs like garbage on my 4090" and "the game runs badly, low frame
//! rate, poor optimisation" sit in noticeably different places even in a model that
//! understands both. Measured against a reference set the gap showed up as 51.1% primary
//! agreement on reviews the labeller called clear-cut.
//!
//! An anchor fitted from labelled reviews removes that mismatch: the category is
//! represented by the mean of things people actually wrote, in the register they wrote them
//! in. The cost is that it needs labels, and the labels are thin. In the one reference set
//! that exists, four of seventeen categories have no example at all and seven have fewer
//! than five, so a set of pure prototypes would have holes exactly where the description was
//! the only thing holding the category up.
//!
//! So each anchor is a blend, weighted by how much evidence there is for that category: a
//! category with a hundred labelled reviews is almost entirely its reviews, a category with
//! five is mostly its description nudged towards them, and a category with none is its
//! description unchanged. That makes fitting a strict improvement on the descriptions rather
//! than a replacement that can regress the long tail.

use std::{collections::HashMap, path::Path};

use serde::{Deserialize, Serialize};

use crate::{
    Error, Result,
    embed::Embedder,
    evaluate::ReferenceLabel,
    model::{EMBEDDING_DIM, MODEL_ID},
    taxonomy::{CORE_SPINE, CORE_SPINE_VERSION, embedding_text},
};

/// Labelled examples a category needs before its own reviews outweigh its description.
///
/// At this value five examples move an anchor 38% of the way towards its reviews and a
/// hundred move it 93%. Fitted by cross-validation on the training subset of the reference
/// set; `search` re-fits it whenever a set is fitted, and this is only the fallback for
/// callers that do not search.
pub const DEFAULT_SMOOTHING: f32 = 8.0;

/// Weight given to a category the labeller named as a secondary topic rather than the main
/// subject.
///
/// A review whose main subject is combat but which also complains about the price is real
/// evidence about price, just weaker and mixed with everything else the review says. For
/// the rarest categories these are the only evidence there is.
pub const DEFAULT_SECONDARY_WEIGHT: f32 = 0.5;

/// Smoothing values tried when fitting. Spans "five examples are already enough" to "a
/// category needs about thirty before it stops leaning on its description".
const SMOOTHING_GRID: &[f32] = &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];

/// Secondary weights tried when fitting, from ignoring secondary topics to treating them as
/// being just as much about the category as the main subject is.
const SECONDARY_GRID: &[f32] = &[0.0, 0.25, 0.5, 1.0];

/// Mention margins tried when fitting. Wider than the range swept by hand, because fitted
/// anchors change the scale of the similarity gaps the margin is measured in.
const MARGIN_GRID: &[f32] = &[0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06];

/// How the anchors were blended out of descriptions and labelled reviews.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FitParams {
    pub smoothing: f32,
    pub secondary_weight: f32,
    /// How close to the best match another category must score to count as mentioned.
    pub mention_margin: f32,
    /// Whether to score categories by how far above their own average a review sits, rather
    /// than by raw similarity. See [`Anchors::calibrate`].
    pub calibrate: bool,
}

impl Default for FitParams {
    fn default() -> Self {
        Self {
            smoothing: DEFAULT_SMOOTHING,
            secondary_weight: DEFAULT_SECONDARY_WEIGHT,
            mention_margin: crate::classify::DEFAULT_MENTION_MARGIN,
            calibrate: true,
        }
    }
}

/// One labelled review, as the fitter sees it.
#[derive(Debug, Clone)]
pub struct Example {
    pub vector: Vec<f32>,
    /// Index into [`CORE_SPINE`].
    pub primary: usize,
    pub secondary: Vec<usize>,
}

/// One category's anchor, with enough provenance to say what it was built from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anchor {
    pub id: String,
    /// Labelled examples behind this anchor, secondary mentions counted fractionally.
    pub evidence: f32,
    /// How far the anchor moved from its written description towards those examples, 0 to 1.
    pub learned_share: f32,
    /// This anchor's mean similarity to the corpus, subtracted before categories compete.
    /// Zero when the set has not been calibrated.
    pub bias: f32,
    pub vector: Vec<f32>,
}

/// The full set of category anchors, in [`CORE_SPINE`] order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anchors {
    pub spine_version: String,
    pub model: String,
    /// App whose reference labels produced these, if any.
    ///
    /// Recorded because an anchor set fitted on one game carries that game's vocabulary,
    /// and whether it transfers to another is an open question rather than an assumption.
    pub fitted_from: Option<u32>,
    pub params: Option<FitParams>,
    pub categories: Vec<Anchor>,
}

impl Anchors {
    /// Embeds the written category descriptions. The zero-setup path, needing no labels.
    ///
    /// # Errors
    ///
    /// Fails if the embedding model cannot be run.
    pub fn from_descriptions(embedder: &mut Embedder) -> Result<Self> {
        let text: Vec<String> = CORE_SPINE.iter().map(embedding_text).collect();
        let vectors = embedder.embed(&text)?;
        Ok(Self {
            spine_version: CORE_SPINE_VERSION.to_owned(),
            model: MODEL_ID.to_owned(),
            fitted_from: None,
            params: None,
            categories: CORE_SPINE
                .iter()
                .zip(vectors)
                .map(|(category, vector)| Anchor {
                    id: category.id.to_owned(),
                    evidence: 0.0,
                    learned_share: 0.0,
                    bias: 0.0,
                    vector,
                })
                .collect(),
        })
    }

    /// Blends the descriptions towards the labelled examples.
    #[must_use]
    pub fn fit(&self, examples: &[Example], app_id: u32, params: FitParams) -> Self {
        let mut sums = vec![vec![0.0_f32; EMBEDDING_DIM]; CORE_SPINE.len()];
        let mut evidence = vec![0.0_f32; CORE_SPINE.len()];

        for example in examples {
            accumulate(&mut sums[example.primary], &example.vector, 1.0);
            evidence[example.primary] += 1.0;
            for &slot in &example.secondary {
                accumulate(&mut sums[slot], &example.vector, params.secondary_weight);
                evidence[slot] += params.secondary_weight;
            }
        }

        let categories = self
            .categories
            .iter()
            .enumerate()
            .map(|(slot, description)| {
                let weight = evidence[slot];
                let share = weight / (weight + params.smoothing);
                // The mean of many vectors is shorter than the mean of few, so blending the
                // raw sum would make diverse categories quietly count for less than the
                // share says they do.
                normalise(&mut sums[slot]);
                let mut vector: Vec<f32> = description
                    .vector
                    .iter()
                    .zip(&sums[slot])
                    .map(|(desc, learned)| (1.0 - share) * desc + share * learned)
                    .collect();
                normalise(&mut vector);
                Anchor {
                    id: description.id.clone(),
                    evidence: weight,
                    learned_share: share,
                    bias: 0.0,
                    vector,
                }
            })
            .collect();

        Self {
            spine_version: CORE_SPINE_VERSION.to_owned(),
            model: MODEL_ID.to_owned(),
            fitted_from: Some(app_id),
            params: Some(params),
            categories,
        }
    }

    /// Records how similar each anchor is to the corpus on average, so that categories
    /// compete on how far above their own average a review sits rather than on raw
    /// similarity.
    ///
    /// Fitting creates a scale problem it then loses to. An anchor built from real reviews
    /// sits inside the region of the space where reviews live, so it scores higher against
    /// *every* review than a written description does, whatever the review is about. Mixing
    /// the two kinds in one argmax therefore hands the fitted categories reviews that belong
    /// to the categories still standing on their descriptions, which is precisely the long
    /// tail that fitting was supposed to leave alone.
    ///
    /// Subtracting each anchor's mean removes the offset without touching the ordering
    /// within a category. Only review vectors are used and no labels, so held-out reviews
    /// may contribute: this is the same corpus the classifier is about to be run over, not
    /// information about the answers.
    pub fn calibrate(&mut self, corpus: &[Vec<f32>]) {
        if corpus.is_empty() {
            return;
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "corpora are millions of reviews at most"
        )]
        let count = corpus.len() as f32;
        for anchor in &mut self.categories {
            let total: f32 = corpus
                .iter()
                .map(|review| dot(&anchor.vector, review))
                .sum();
            anchor.bias = total / count;
        }
    }

    /// How far above its own corpus average each category scores this review.
    ///
    /// Identical to plain cosine similarity for an uncalibrated set, whose biases are zero.
    #[must_use]
    pub fn similarities(&self, vector: &[f32]) -> Vec<f32> {
        self.categories
            .iter()
            .map(|anchor| dot(&anchor.vector, vector) - anchor.bias)
            .collect()
    }

    /// The margin the anchors were fitted with, or the hand-swept default.
    #[must_use]
    pub fn mention_margin(&self) -> f32 {
        self.params
            .map_or(crate::classify::DEFAULT_MENTION_MARGIN, |p| {
                p.mention_margin
            })
    }

    /// # Errors
    ///
    /// Fails if the file cannot be written.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }

    /// Reads a fitted anchor set, refusing one that cannot mean what it says.
    ///
    /// # Errors
    ///
    /// Fails if the file is missing or malformed, or if it was fitted against a different
    /// taxonomy or embedding model. Using such a set would silently change every number the
    /// tool reports.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|_| Error::NoAnchors {
            path: path.to_path_buf(),
        })?;
        let anchors: Self = serde_json::from_slice(&bytes)?;

        if anchors.spine_version != CORE_SPINE_VERSION {
            return Err(Error::StaleAnchors {
                field: "taxonomy",
                expected: CORE_SPINE_VERSION.to_owned(),
                actual: anchors.spine_version,
            });
        }
        if anchors.model != MODEL_ID {
            return Err(Error::StaleAnchors {
                field: "embedding model",
                expected: MODEL_ID.to_owned(),
                actual: anchors.model,
            });
        }
        let ids: Vec<&str> = anchors.categories.iter().map(|a| a.id.as_str()).collect();
        let expected: Vec<&str> = CORE_SPINE.iter().map(|c| c.id).collect();
        if ids != expected {
            return Err(Error::StaleAnchors {
                field: "category order",
                expected: expected.join(","),
                actual: ids.join(","),
            });
        }
        if let Some(bad) = anchors
            .categories
            .iter()
            .find(|a| a.vector.len() != EMBEDDING_DIM)
        {
            return Err(Error::StaleAnchors {
                field: "vector width",
                expected: EMBEDDING_DIM.to_string(),
                actual: bad.vector.len().to_string(),
            });
        }
        Ok(anchors)
    }
}

/// Looks up the stored vector of every review in the most recent capture, by review id.
///
/// Reference labels name reviews by id while embeddings are keyed by the hash of the text,
/// so fitting needs the capture as well as the vectors: the same text posted twice is
/// embedded once, and two ids legitimately share a vector.
///
/// # Errors
///
/// Fails if the capture or its embeddings are missing.
pub fn corpus_vectors(out_dir: &Path, app_id: u32) -> Result<HashMap<String, Vec<f32>>> {
    let snapshot = crate::embed::latest_snapshot(out_dir, app_id)?;
    let by_hash = crate::classify::load_embeddings(&snapshot.join("embeddings.parquet"))?;

    let mut by_id = HashMap::new();
    for review in crate::classify::read_reviews(&snapshot)? {
        let review = review?;
        if let Some(vector) = by_hash.get(&review.text_hash) {
            by_id.insert(review.recommendationid, vector.clone());
        }
    }
    Ok(by_id)
}

/// Turns reference labels into fitting examples, dropping any the corpus cannot supply.
///
/// A label is dropped when the review is not in this capture, or when it names a category
/// the taxonomy does not have. Both are silent here because both are reported by
/// `census evaluate` as unmatched labels and unknown categories, and a fitter that refused
/// to run over a partially matched reference set would be unusable after any re-crawl.
#[must_use]
pub fn to_examples<S: std::hash::BuildHasher>(
    labels: &[ReferenceLabel],
    vectors: &HashMap<String, Vec<f32>, S>,
) -> Vec<Example> {
    let slot: HashMap<&str, usize> = CORE_SPINE
        .iter()
        .enumerate()
        .map(|(index, category)| (category.id, index))
        .collect();

    labels
        .iter()
        .filter_map(|label| {
            Some(Example {
                vector: vectors.get(&label.id)?.clone(),
                primary: *slot.get(label.primary.as_str())?,
                secondary: label
                    .secondary
                    .iter()
                    .filter_map(|id| slot.get(id.as_str()).copied())
                    .collect(),
            })
        })
        .collect()
}

/// Whether the search may choose calibration, or is told.
///
/// Overridable because the search picks on overall agreement, which an imbalanced corpus
/// lets the largest categories decide. Someone reading per-category prevalence may
/// reasonably want the setting that serves the tail even where it costs a point overall.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Calibration {
    #[default]
    Search,
    Always,
    Never,
}

impl Calibration {
    fn candidates(self) -> impl Iterator<Item = bool> {
        match self {
            Self::Search => [false, true].as_slice(),
            Self::Always => [true].as_slice(),
            Self::Never => [false].as_slice(),
        }
        .iter()
        .copied()
    }
}

/// What a parameter search found, and how confident that finding is entitled to be.
#[derive(Debug, Clone)]
pub struct FitOutcome {
    pub params: FitParams,
    /// Held-out primary agreement of the chosen blend, averaged across folds.
    pub primary_agreement: f64,
    /// Held-out primary agreement of the unfitted descriptions, on the same folds.
    pub baseline_agreement: f64,
    /// Whether the description-only baseline was better off calibrated too.
    pub baseline_calibrated: bool,
    /// Held-out mention macro F1 of the chosen blend and margin.
    pub mention_macro_f1: f64,
    pub baseline_macro_f1: f64,
    pub folds: usize,
    pub examples: usize,
}

/// Fits the blend and the mention margin by cross-validation over the training examples.
///
/// Two stages, because the two settings answer different questions and one objective cannot
/// serve both. The blend is chosen on held-out primary agreement, which is the question the
/// classifier is mainly asked. The margin only affects which *additional* categories a
/// review keeps, so it is chosen afterwards, on held-out mention macro F1, with the blend
/// already fixed.
///
/// Every score here is on examples the fold did not see. Selecting on training-fold scores
/// would pick whichever setting memorised the labels hardest, which for anchors is always
/// the least smoothing.
#[must_use]
pub fn search(
    descriptions: &Anchors,
    examples: &[Example],
    corpus: &[Vec<f32>],
    folds: usize,
    calibration: Calibration,
) -> FitOutcome {
    let folds = folds.max(2).min(examples.len().max(2));
    let score = |params, objective| {
        cross_validate(descriptions, examples, corpus, folds, params, objective)
    };

    let mut best = (f64::NEG_INFINITY, FitParams::default());
    for &smoothing in SMOOTHING_GRID {
        for &secondary_weight in SECONDARY_GRID {
            for calibrate in calibration.candidates() {
                let params = FitParams {
                    smoothing,
                    secondary_weight,
                    calibrate,
                    ..FitParams::default()
                };
                let agreement = score(params, Objective::Primary);
                if agreement > best.0 {
                    best = (agreement, params);
                }
            }
        }
    }

    // The description-only baseline is searched over everything that is not the blend
    // itself, so neither the margin nor the calibration can be credited to fitting when it
    // would have helped the descriptions just as much.
    let flat = FitParams {
        smoothing: f32::INFINITY,
        secondary_weight: 0.0,
        ..FitParams::default()
    };
    let mut baseline = (f64::NEG_INFINITY, flat);
    for calibrate in calibration.candidates() {
        let params = FitParams { calibrate, ..flat };
        let agreement = score(params, Objective::Primary);
        if agreement > baseline.0 {
            baseline = (agreement, params);
        }
    }

    let best_margin = |mut params: FitParams| {
        let mut found = (f64::NEG_INFINITY, params.mention_margin);
        for &candidate in MARGIN_GRID {
            params.mention_margin = candidate;
            let f1 = score(params, Objective::Mentions);
            if f1 > found.0 {
                found = (f1, candidate);
            }
        }
        found
    };
    let (mention_macro_f1, mention_margin) = best_margin(best.1);
    let (baseline_macro_f1, _) = best_margin(baseline.1);

    FitOutcome {
        params: FitParams {
            mention_margin,
            ..best.1
        },
        primary_agreement: best.0,
        baseline_agreement: baseline.0,
        baseline_calibrated: baseline.1.calibrate,
        mention_macro_f1,
        baseline_macro_f1,
        folds,
        examples: examples.len(),
    }
}

/// What a cross-validated search is trying to maximise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    /// Share of reviews whose main subject is identified correctly.
    ///
    /// Dominated by whatever the corpus is mostly about: in the one reference set that
    /// exists, two categories carry 72% of the labels, so a setting that improves those two
    /// and ruins the other fifteen still wins on this.
    Primary,
    /// Mean per-category F1 over mentions, counting every category once however rare.
    ///
    /// The objective that matches what the tool is for. Prevalence and helpfulness bias are
    /// reported per category, so a category the classifier never predicts is not a small
    /// error but a missing row, and this is the only objective that treats it as one.
    Mentions,
}

/// Round-robin folds. The reference set arrives grouped by predicted category, so dealing
/// examples out in turn gives every fold nearly the same class balance, which random
/// assignment would not at these counts: several categories have five examples in total.
fn cross_validate(
    descriptions: &Anchors,
    examples: &[Example],
    corpus: &[Vec<f32>],
    folds: usize,
    params: FitParams,
    objective: Objective,
) -> f64 {
    let mut agreed = 0_u64;
    let mut total = 0_u64;
    let mut confusion = vec![Counts::default(); CORE_SPINE.len()];

    for fold in 0..folds {
        let train: Vec<Example> = examples
            .iter()
            .enumerate()
            .filter(|(index, _)| index % folds != fold)
            .map(|(_, example)| example.clone())
            .collect();
        let mut fitted = descriptions.fit(&train, 0, params);
        if params.calibrate {
            fitted.calibrate(corpus);
        }

        for example in examples.iter().skip(fold).step_by(folds) {
            let sims = fitted.similarities(&example.vector);
            let (primary, mentions) = crate::classify::assign(&sims, params.mention_margin);
            match objective {
                Objective::Primary => {
                    total += 1;
                    if primary == example.primary {
                        agreed += 1;
                    }
                }
                Objective::Mentions => {
                    for (slot, counts) in confusion.iter_mut().enumerate() {
                        let expected = example.primary == slot || example.secondary.contains(&slot);
                        let found = mentions & (1 << slot) != 0;
                        counts.expected += u64::from(expected);
                        counts.found += u64::from(found);
                        counts.hit += u64::from(expected && found);
                    }
                }
            }
        }
    }

    match objective {
        Objective::Primary => ratio(agreed, total),
        Objective::Mentions => macro_f1(&confusion),
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Counts {
    expected: u64,
    found: u64,
    hit: u64,
}

/// Unweighted mean F1 over the categories the examples actually use, so a rare category the
/// anchors never find costs as much as a common one.
fn macro_f1(confusion: &[Counts]) -> f64 {
    let scores: Vec<f64> = confusion
        .iter()
        .filter(|c| c.expected > 0)
        .map(|c| {
            let precision = ratio(c.hit, c.found);
            let recall = ratio(c.hit, c.expected);
            if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            }
        })
        .collect();
    if scores.is_empty() {
        return 0.0;
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "the taxonomy has a few dozen categories at most"
    )]
    let count = scores.len() as f64;
    scores.iter().sum::<f64>() / count
}

#[expect(
    clippy::cast_precision_loss,
    reason = "reference sets are a few hundred reviews"
)]
fn ratio(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        return 0.0;
    }
    part as f64 / whole as f64
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn accumulate(into: &mut [f32], vector: &[f32], weight: f32) {
    for (slot, value) in into.iter_mut().zip(vector) {
        *slot += value * weight;
    }
}

fn normalise(vector: &mut [f32]) {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for value in vector.iter_mut() {
            *value /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(slot: usize) -> Vec<f32> {
        let mut vector = vec![0.0; EMBEDDING_DIM];
        vector[slot] = 1.0;
        vector
    }

    fn descriptions() -> Anchors {
        Anchors {
            spine_version: CORE_SPINE_VERSION.to_owned(),
            model: MODEL_ID.to_owned(),
            fitted_from: None,
            params: None,
            categories: CORE_SPINE
                .iter()
                .enumerate()
                .map(|(slot, category)| Anchor {
                    id: category.id.to_owned(),
                    evidence: 0.0,
                    learned_share: 0.0,
                    bias: 0.0,
                    vector: unit(slot),
                })
                .collect(),
        }
    }

    #[test]
    fn a_category_with_no_examples_keeps_its_description_exactly() {
        let fitted = descriptions().fit(&[], 1, FitParams::default());
        for (before, after) in descriptions().categories.iter().zip(&fitted.categories) {
            assert_eq!(before.vector, after.vector, "{} moved", after.id);
            assert!(after.learned_share.abs() < f32::EPSILON);
        }
    }

    #[test]
    fn examples_pull_an_anchor_towards_them_without_reaching_them() {
        // Every example points somewhere the description does not, so any movement shows up
        // as similarity to a direction the description had none of.
        let examples: Vec<Example> = (0..5)
            .map(|_| Example {
                vector: unit(300),
                primary: 0,
                secondary: vec![],
            })
            .collect();
        let fitted = descriptions().fit(&examples, 1, FitParams::default());

        let moved = fitted.categories[0].vector[300];
        assert!(moved > 0.0, "the anchor did not move at all");
        assert!(
            fitted.categories[0].vector[0] > moved,
            "five examples should not outweigh the description"
        );
        assert!((fitted.categories[0].evidence - 5.0).abs() < 1e-6);
    }

    #[test]
    fn evidence_accumulates_faster_than_the_anchor_gives_way() {
        let many: Vec<Example> = (0..200)
            .map(|_| Example {
                vector: unit(300),
                primary: 1,
                secondary: vec![],
            })
            .collect();
        let fitted = descriptions().fit(&many, 1, FitParams::default());
        assert!(
            fitted.categories[1].learned_share > 0.9,
            "two hundred examples should dominate a description"
        );
        assert!(fitted.categories[1].vector[300] > fitted.categories[1].vector[1]);
    }

    #[test]
    fn secondary_mentions_count_for_less_than_main_subjects() {
        let secondary = [Example {
            vector: unit(300),
            primary: 5,
            secondary: vec![2],
        }];
        let params = FitParams {
            secondary_weight: 0.5,
            ..FitParams::default()
        };
        let fitted = descriptions().fit(&secondary, 1, params);
        assert!((fitted.categories[5].evidence - 1.0).abs() < 1e-6);
        assert!((fitted.categories[2].evidence - 0.5).abs() < 1e-6);
        assert!(fitted.categories[5].learned_share > fitted.categories[2].learned_share);
    }

    #[test]
    fn anchors_stay_unit_length_so_similarities_remain_cosines() {
        let examples = [Example {
            vector: unit(300),
            primary: 4,
            secondary: vec![9],
        }];
        let fitted = descriptions().fit(&examples, 1, FitParams::default());
        for anchor in &fitted.categories {
            let norm = anchor.vector.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "{} was not unit length",
                anchor.id
            );
        }
    }

    #[test]
    fn a_set_fitted_against_another_taxonomy_is_refused_rather_than_used() {
        let dir = std::env::temp_dir().join("census-anchor-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("stale.json");
        let mut anchors = descriptions();
        anchors.spine_version = "core-1".to_owned();
        anchors.save(&path).unwrap();

        let error = Anchors::load(&path).unwrap_err();
        assert!(
            matches!(
                error,
                Error::StaleAnchors {
                    field: "taxonomy",
                    ..
                }
            ),
            "loaded anchors from a taxonomy this build does not have: {error}"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_saved_set_survives_the_round_trip() {
        let dir = std::env::temp_dir().join("census-anchor-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("round-trip.json");
        let examples = [Example {
            vector: unit(300),
            primary: 0,
            secondary: vec![],
        }];
        let fitted = descriptions().fit(&examples, 296_970, FitParams::default());
        fitted.save(&path).unwrap();

        let loaded = Anchors::load(&path).unwrap();
        assert_eq!(loaded.fitted_from, Some(296_970));
        assert_eq!(loaded.categories[0].vector, fitted.categories[0].vector);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn calibration_stops_a_fitted_anchor_outbidding_a_description_on_reviews_that_are_not_its_own()
    {
        // Two categories. The first is fitted and so sits among the reviews; the second
        // still stands on its description, off in a direction reviews never point. A review
        // that belongs to the second is nonetheless nearer the first in raw cosine, which is
        // the scale problem rather than a judgement about the review.
        let mut anchors = descriptions();
        anchors.categories[0].vector = unit(300);
        anchors.categories[1].vector = unit(301);
        let corpus: Vec<Vec<f32>> = (0..50)
            .map(|index| {
                let mut review = unit(300);
                if index % 10 == 0 {
                    review[301] = 0.35;
                    normalise(&mut review);
                }
                review
            })
            .collect();

        let belongs_to_second = corpus[0].clone();
        let raw = anchors.similarities(&belongs_to_second);
        assert!(raw[0] > raw[1], "this test needs the mismatch it is about");

        anchors.calibrate(&corpus);
        let calibrated = anchors.similarities(&belongs_to_second);
        assert!(
            calibrated[1] > calibrated[0],
            "calibration left the description anchor unable to win its own review"
        );
    }

    #[test]
    fn calibration_leaves_the_ordering_between_reviews_within_a_category_alone() {
        let mut anchors = descriptions();
        let corpus: Vec<Vec<f32>> = (0..8).map(|slot| unit(slot % 4)).collect();
        let near = unit(2);
        let far = unit(9);

        let before = (
            anchors.similarities(&near)[2],
            anchors.similarities(&far)[2],
        );
        anchors.calibrate(&corpus);
        let after = (
            anchors.similarities(&near)[2],
            anchors.similarities(&far)[2],
        );
        assert!(before.0 > before.1 && after.0 > after.1);
        assert!(
            (before.0 - before.1 - (after.0 - after.1)).abs() < 1e-6,
            "subtracting a constant should not change a gap"
        );
    }

    #[test]
    fn an_uncalibrated_set_scores_exactly_as_plain_cosine() {
        let anchors = descriptions();
        let review = unit(3);
        assert!(anchors.categories.iter().all(|a| a.bias == 0.0));
        assert!((anchors.similarities(&review)[3] - 1.0).abs() < 1e-6);
        assert!(anchors.similarities(&review)[4].abs() < 1e-6);
    }

    #[test]
    fn the_search_never_returns_a_setting_it_did_not_score() {
        let examples: Vec<Example> = (0..40)
            .map(|index| Example {
                vector: unit(index % 4),
                primary: index % 4,
                secondary: vec![],
            })
            .collect();
        let corpus: Vec<Vec<f32>> = (0..4).map(unit).collect();
        let outcome = search(&descriptions(), &examples, &corpus, 5, Calibration::Search);
        assert!(SMOOTHING_GRID.contains(&outcome.params.smoothing));
        assert!(SECONDARY_GRID.contains(&outcome.params.secondary_weight));
        assert!(MARGIN_GRID.contains(&outcome.params.mention_margin));
        assert_eq!(outcome.examples, 40);
    }
}
