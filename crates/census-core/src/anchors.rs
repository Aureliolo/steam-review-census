//! The vectors a review is compared against.
//!
//! The zero-setup classifier embeds a *written description* of each category and takes the
//! nearest one. That description is prose about a topic, and reviews about that topic are
//! not prose about it: "runs like garbage on my 4090" and "the game runs badly, low frame
//! rate, poor optimisation" sit in noticeably different places even in a model that
//! understands both. Measured against a reference set, the descriptions alone agree with
//! the labels on roughly a third of reviews.
//!
//! An anchor fitted from labelled reviews removes that mismatch: the category is
//! represented by the mean of things people actually wrote, in the register they wrote them
//! in. The cost is that it needs labels, and labels are thinnest exactly where a category is
//! rare, which is where the description was the only thing holding it up. A set of pure
//! prototypes would therefore have holes in precisely the wrong places.
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
    /// What the corpus mean is used for before categories compete. See [`Scoring`].
    pub scoring: Scoring,
}

impl Default for FitParams {
    fn default() -> Self {
        Self {
            smoothing: DEFAULT_SMOOTHING,
            secondary_weight: DEFAULT_SECONDARY_WEIGHT,
            mention_margin: crate::classify::DEFAULT_MENTION_MARGIN,
            scoring: Scoring::Bias,
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
    /// Apps whose reference labels produced these. Empty for written descriptions.
    ///
    /// Recorded because an anchor set carries the vocabulary of the games it was fitted on,
    /// and whether it reaches a game outside that list is a measurement rather than an
    /// assumption. `census fit --leave-one-out` is what makes it.
    #[serde(default)]
    pub fitted_from: Vec<u32>,
    pub params: Option<FitParams>,
    /// The corpus mean subtracted from every review before it is compared, when the set was
    /// fitted with [`Scoring::Centred`].
    #[serde(default)]
    pub centre: Option<Vec<f32>>,
    pub categories: Vec<Anchor>,
}

impl Anchors {
    /// How wide these vectors are, which is a fact about the encoder that produced them
    /// rather than about the build reading them.
    #[must_use]
    pub fn dimensions(&self) -> usize {
        self.categories.first().map_or(0, |a| a.vector.len())
    }

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
            model: embedder.encoder().id().to_owned(),
            fitted_from: Vec::new(),
            params: None,
            centre: None,
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
    pub fn fit(&self, examples: &[Example], app_ids: &[u32], params: FitParams) -> Self {
        let mut sums = vec![vec![0.0_f32; self.dimensions()]; CORE_SPINE.len()];
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
            model: self.model.clone(),
            fitted_from: app_ids.to_vec(),
            params: Some(params),
            centre: None,
            categories,
        }
    }

    /// Applies whatever the fitted scoring needs the corpus mean for.
    ///
    /// One place, because a set fitted with one treatment and scored under another is not a
    /// worse answer but a meaningless one, and three callers each remembering to do it is
    /// three chances to forget.
    pub fn prepare(&mut self, scoring: Scoring, centroid: &[f32]) {
        match scoring {
            Scoring::Raw => {}
            Scoring::Bias => self.calibrate(centroid),
            Scoring::Centred => self.centre_on(centroid),
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
    /// Takes the corpus centroid rather than the corpus itself. A dot product is linear in
    /// its second argument, so the mean of an anchor's similarity to every review *is* its
    /// similarity to the mean review. The two are equal rather than approximately equal, and
    /// the second costs 384 multiplications instead of a walk over a million vectors. That
    /// matters because fitting calibrates once per fold per parameter combination tried:
    /// against six corpora the old form was several hundred passes over three million
    /// vectors, and this form is a few thousand multiplications.
    pub fn calibrate(&mut self, centroid: &[f32]) {
        if centroid.len() != self.dimensions() {
            return;
        }
        for anchor in &mut self.categories {
            anchor.bias = dot(&anchor.vector, centroid);
        }
    }

    /// Moves the origin to the corpus mean, so what is compared is what makes a review
    /// unlike the others rather than what every review has in common.
    ///
    /// The centre has to be kept, because the same subtraction has to be done to every
    /// review the set is later run over, and an anchor set is used long after the corpus
    /// mean that produced it has been forgotten.
    pub fn centre_on(&mut self, centroid: &[f32]) {
        if centroid.len() != self.dimensions() {
            return;
        }
        for anchor in &mut self.categories {
            for (slot, mean) in anchor.vector.iter_mut().zip(centroid) {
                *slot -= mean;
            }
            normalise(&mut anchor.vector);
            anchor.bias = 0.0;
        }
        self.centre = Some(centroid.to_vec());
    }

    /// How strongly each category matches this review, on whatever scale the set was fitted
    /// with. Plain cosine similarity for a set that was fitted with neither treatment.
    #[must_use]
    pub fn similarities(&self, vector: &[f32]) -> Vec<f32> {
        let Some(centre) = self.centre.as_ref().filter(|c| c.len() == vector.len()) else {
            return self
                .categories
                .iter()
                .map(|anchor| dot(&anchor.vector, vector) - anchor.bias)
                .collect();
        };
        let mut moved: Vec<f32> = vector.iter().zip(centre).map(|(v, m)| v - m).collect();
        normalise(&mut moved);
        self.categories
            .iter()
            .map(|anchor| dot(&anchor.vector, &moved))
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
        let ids: Vec<&str> = anchors.categories.iter().map(|a| a.id.as_str()).collect();
        let expected: Vec<&str> = CORE_SPINE.iter().map(|c| c.id).collect();
        if ids != expected {
            return Err(Error::StaleAnchors {
                field: "category order",
                expected: expected.join(","),
                actual: ids.join(","),
            });
        }
        let width = anchors.dimensions();
        if let Some(bad) = anchors.categories.iter().find(|a| a.vector.len() != width) {
            return Err(Error::StaleAnchors {
                field: "vector width",
                expected: width.to_string(),
                actual: bad.vector.len().to_string(),
            });
        }
        Ok(anchors)
    }
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
    Only(Scoring),
}

impl Calibration {
    fn candidates(self) -> impl Iterator<Item = Scoring> {
        match self {
            Self::Search => [Scoring::Raw, Scoring::Bias, Scoring::Centred].as_slice(),
            Self::Only(Scoring::Raw) => [Scoring::Raw].as_slice(),
            Self::Only(Scoring::Bias) => [Scoring::Bias].as_slice(),
            Self::Only(Scoring::Centred) => [Scoring::Centred].as_slice(),
        }
        .iter()
        .copied()
    }
}

/// How a review's distance from a category is turned into a score the categories compete on.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scoring {
    /// Plain cosine similarity to the anchor.
    #[default]
    Raw,
    /// Each anchor's own mean similarity to the corpus subtracted from its score, so a
    /// category is scored by how far above its own average a review sits.
    Bias,
    /// The corpus mean removed from both sides before they are compared.
    ///
    /// Sentence embeddings share a strong common direction: every review is somewhat similar
    /// to every category, and the categories nearest that direction collect reviews that are
    /// not about them.
    ///
    /// Its numerator ranks exactly as [`Scoring::Bias`] does, because the two extra terms it
    /// picks up are the same for every category. What makes it a different answer is the
    /// division that follows: each category is scaled by how far its own anchor sits from
    /// the corpus mean, which is a per-category factor rather than a per-category offset.
    ///
    /// Measured and rejected on 2026-09-09 across the six reference sets under
    /// gte-multilingual-base: 50.5% primary agreement and 0.458 mention macro F1 on the 600
    /// held-back reviews, against 54.2% and 0.498 for raw similarity. It stays a candidate
    /// because the search re-answers the question for every corpus and every encoder, and
    /// the reason it loses here is a property of this embedding space rather than of the
    /// method.
    Centred,
}

/// Whether an anchor set picks the same primary category as the labels, review by review.
///
/// Works on vectors already in hand rather than on stored classifications, so a set can be
/// measured against reviews it was never used to classify. That is what makes leaving a
/// whole game out of the fit and testing on it cheap enough to do six times over.
///
/// Per review rather than as a total because two anchor sets are compared over the same
/// reviews, and the ones they both place the same way say nothing about which is better.
#[must_use]
pub fn agreements(anchors: &Anchors, examples: &[Example]) -> Vec<bool> {
    examples
        .iter()
        .map(|example| {
            let sims = anchors.similarities(&example.vector);
            let (primary, _) = crate::classify::assign(&sims, anchors.mention_margin());
            primary == example.primary
        })
        .collect()
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
    /// Held-out mention macro F1 of the chosen blend and margin.
    pub mention_macro_f1: f64,
    pub baseline_macro_f1: f64,
    /// What the blend was chosen to win. The other figure is a cost, not a claim.
    pub objective: Objective,
    pub folds: usize,
    pub examples: usize,
}

/// Fits the blend and the mention margin by cross-validation over the training examples.
///
/// Which of the two questions the search is trying to win is the caller's to say, because
/// they pull in different directions: the secondary weight that best identifies a review's
/// main subject is zero, and zero leaves every category that is usually somebody's second
/// subject with almost no evidence and a written description it never moves off.
///
/// The margin is settled last and always on mentions, whatever the blend was chosen for,
/// since it decides nothing else.
///
/// Every score here is on examples the fold did not see. Selecting on training-fold scores
/// would pick whichever setting memorised the labels hardest, which for anchors is always
/// the least smoothing.
#[must_use]
pub fn search(
    descriptions: &Anchors,
    examples: &[Example],
    centroid: &[f32],
    folds: usize,
    calibration: Calibration,
    objective: Objective,
) -> FitOutcome {
    let folds = folds.max(2).min(examples.len().max(2));
    let score = |params, objective| {
        cross_validate(descriptions, examples, centroid, folds, params, objective)
    };
    let margins: &[f32] = match objective {
        // The margin cannot change which category scores highest, so trying it here would
        // only spend folds to arrive back where it started.
        Objective::Primary => &[crate::classify::DEFAULT_MENTION_MARGIN],
        Objective::Mentions => MARGIN_GRID,
    };

    let mut best = (f64::NEG_INFINITY, FitParams::default());
    for &smoothing in SMOOTHING_GRID {
        for &secondary_weight in SECONDARY_GRID {
            for scoring in calibration.candidates() {
                for &mention_margin in margins {
                    let params = FitParams {
                        smoothing,
                        secondary_weight,
                        mention_margin,
                        scoring,
                    };
                    let found = score(params, objective);
                    if found > best.0 {
                        best = (found, params);
                    }
                }
            }
        }
    }

    // The description-only baseline is searched over everything that is not the blend
    // itself, so neither the margin nor the treatment of the corpus mean can be credited to
    // fitting when it would have helped the descriptions just as much. Each figure takes the
    // best the descriptions manage on that question, rather than reporting one setting on
    // both, which would flatter fitting on whichever question the setting was not chosen for.
    let flat = FitParams {
        smoothing: f32::INFINITY,
        secondary_weight: 0.0,
        ..FitParams::default()
    };
    let mut baseline_agreement = f64::NEG_INFINITY;
    let mut baseline_macro_f1 = f64::NEG_INFINITY;
    for scoring in calibration.candidates() {
        let params = FitParams { scoring, ..flat };
        baseline_agreement = baseline_agreement.max(score(params, Objective::Primary));
        for &mention_margin in MARGIN_GRID {
            let params = FitParams {
                mention_margin,
                ..params
            };
            baseline_macro_f1 = baseline_macro_f1.max(score(params, Objective::Mentions));
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
    let params = FitParams {
        mention_margin,
        ..best.1
    };

    FitOutcome {
        params,
        // Reported rather than selected on, so a blend chosen for one question still says
        // plainly what it cost the other.
        primary_agreement: score(params, Objective::Primary),
        baseline_agreement,
        mention_macro_f1,
        baseline_macro_f1,
        objective,
        folds,
        examples: examples.len(),
    }
}

/// What a cross-validated search is trying to maximise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    /// Share of reviews whose main subject is identified correctly.
    ///
    /// Dominated by whatever the corpus is mostly about. Across the reference sets two
    /// categories carry roughly two labels in five, so a setting that improves those two
    /// and ruins every other category still wins on this.
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
    centroid: &[f32],
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
        let mut fitted = descriptions.fit(&train, &[], params);
        fitted.prepare(params.scoring, centroid);

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
                Objective::Mentions => tally(&mut confusion, example, mentions),
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

/// What one anchor set gets right on a set of labelled reviews.
#[derive(Debug, Clone, Copy, Default)]
pub struct Scored {
    /// Reviews whose main subject was identified as the labeller identified it.
    pub agreed: u64,
    pub compared: u64,
    /// Mean per-category F1 over mentions, counting every category once however rare.
    pub mention_macro_f1: f64,
}

impl Scored {
    /// Share of reviews whose main subject was agreed, or zero when nothing was compared.
    #[must_use]
    pub fn primary_agreement(&self) -> f64 {
        ratio(self.agreed, self.compared)
    }
}

/// Scores an anchor set against labelled reviews it is not being fitted to.
///
/// Both questions at once, because they are answered from the same pass and reporting only
/// one of them is how a set that finds two categories well and nineteen not at all comes to
/// look like a good one.
#[must_use]
pub fn score(anchors: &Anchors, examples: &[Example], margin: f32) -> Scored {
    let mut agreed = 0_u64;
    let mut confusion = vec![Counts::default(); CORE_SPINE.len()];
    for example in examples {
        let sims = anchors.similarities(&example.vector);
        let (primary, mentions) = crate::classify::assign(&sims, margin);
        agreed += u64::from(primary == example.primary);
        tally(&mut confusion, example, mentions);
    }
    Scored {
        agreed,
        compared: examples.len() as u64,
        mention_macro_f1: macro_f1(&confusion),
    }
}

fn tally(confusion: &mut [Counts], example: &Example, mentions: u32) {
    for (slot, counts) in confusion.iter_mut().enumerate() {
        let expected = example.primary == slot || example.secondary.contains(&slot);
        let found = mentions & (1 << slot) != 0;
        counts.expected += u64::from(expected);
        counts.found += u64::from(found);
        counts.hit += u64::from(expected && found);
    }
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

    const DIM: usize = crate::model::Encoder::E5Small.dimensions();

    fn unit(slot: usize) -> Vec<f32> {
        let mut vector = vec![0.0; DIM];
        vector[slot] = 1.0;
        vector
    }

    fn descriptions() -> Anchors {
        Anchors {
            spine_version: CORE_SPINE_VERSION.to_owned(),
            model: crate::model::Encoder::E5Small.id().to_owned(),
            fitted_from: Vec::new(),
            params: None,
            centre: None,
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
        let fitted = descriptions().fit(&[], &[1], FitParams::default());
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
        let fitted = descriptions().fit(&examples, &[1], FitParams::default());

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
        let fitted = descriptions().fit(&many, &[1], FitParams::default());
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
        let fitted = descriptions().fit(&secondary, &[1], params);
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
        let fitted = descriptions().fit(&examples, &[1], FitParams::default());
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
        let fitted = descriptions().fit(&examples, &[296_970], FitParams::default());
        fitted.save(&path).unwrap();

        let loaded = Anchors::load(&path).unwrap();
        assert_eq!(loaded.fitted_from, vec![296_970]);
        assert_eq!(loaded.categories[0].vector, fitted.categories[0].vector);
        std::fs::remove_file(&path).ok();
    }

    /// The mean of a set of vectors, which is all calibration needs of a corpus.
    fn centroid_of(corpus: &[Vec<f32>]) -> Vec<f32> {
        let mut total = vec![0.0_f32; DIM];
        for review in corpus {
            accumulate(&mut total, review, 1.0);
        }
        #[expect(clippy::cast_precision_loss, reason = "test corpora are tiny")]
        let count = corpus.len() as f32;
        for value in &mut total {
            *value /= count;
        }
        total
    }

    #[test]
    fn calibrating_on_the_centroid_is_the_same_as_averaging_over_every_review() {
        // The whole reason a million-review corpus collapses to 384 floats. If these ever
        // disagreed, calibration would be an approximation rather than the identity it
        // claims to be, and the speed would have been bought with accuracy.
        let corpus: Vec<Vec<f32>> = (0..64)
            .map(|index| {
                let mut review = unit(index % 7);
                review[300] = 0.5;
                review[index % DIM] += 0.25;
                normalise(&mut review);
                review
            })
            .collect();

        let mut anchors = descriptions();
        anchors.calibrate(&centroid_of(&corpus));

        #[expect(clippy::cast_precision_loss, reason = "test corpora are tiny")]
        let count = corpus.len() as f32;
        for (slot, anchor) in anchors.categories.iter().enumerate() {
            let naive: f32 = corpus.iter().map(|r| dot(&unit(slot), r)).sum::<f32>() / count;
            assert!(
                (anchor.bias - naive).abs() < 1e-5,
                "{} calibrated to {} but averages {naive}",
                anchor.id,
                anchor.bias
            );
        }
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

        anchors.calibrate(&centroid_of(&corpus));
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
        anchors.calibrate(&centroid_of(&corpus));
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

    /// Centring earns its place only if it can reach an answer calibrating cannot, and the
    /// two agree on everything except the scale each category is divided by.
    #[test]
    fn centring_reaches_an_answer_calibrating_cannot() {
        // A common direction every review and every anchor shares, which is what makes a
        // broad category the nearest thing to reviews that are not about it.
        // Two slots no category's own anchor sits in, so nothing but this arrangement
        // decides the answer. The corpus mean lies along the second of them.
        let (across, along) = (DIM - 2, DIM - 1);
        let plane = |x: f32, y: f32| {
            let mut vector = vec![0.0_f32; DIM];
            vector[across] = x;
            vector[along] = y;
            vector
        };

        let mut anchors = descriptions();
        anchors.categories[1].vector = plane(1.0, 0.0);
        anchors.categories[2].vector = plane(0.0, 1.0);
        // Mostly what every review has in common, with a little of its own.
        let review = plane(0.436, 0.9);
        let centroid = plane(0.0, 0.5);

        let mut calibrated = anchors.clone();
        calibrated.prepare(Scoring::Bias, &centroid);
        let mut centred = anchors.clone();
        centred.prepare(Scoring::Centred, &centroid);

        // The two differ only by a per-category divisor, so a review they order differently
        // is the entire justification for offering both.
        let gap = |set: &Anchors| {
            let sims = set.similarities(&review);
            sims[2] - sims[1]
        };
        assert!(
            gap(&calibrated) * gap(&centred) < 0.0,
            "calibrated put the two categories {:?} and centred {:?}, which is the same \
             answer twice",
            gap(&calibrated),
            gap(&centred)
        );
        assert!(
            centred.centre.is_some(),
            "the centre has to be kept, or the same subtraction cannot be done again"
        );
        assert!(
            calibrated.centre.is_none(),
            "calibrating needs nothing of the corpus at scoring time"
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
        let corpus: Vec<Vec<f32>> = (0..4).map(unit).collect();
        for objective in [Objective::Primary, Objective::Mentions] {
            let outcome = search(
                &descriptions(),
                &spread_over_four(),
                &centroid_of(&corpus),
                5,
                Calibration::Search,
                objective,
            );
            assert!(SMOOTHING_GRID.contains(&outcome.params.smoothing));
            assert!(SECONDARY_GRID.contains(&outcome.params.secondary_weight));
            assert!(MARGIN_GRID.contains(&outcome.params.mention_margin));
            assert_eq!(outcome.objective, objective);
            assert_eq!(outcome.examples, 40);
        }
    }

    /// Both figures come back whichever one was chosen on, or a caller cannot see what the
    /// choice cost and the flag is unreadable.
    #[test]
    fn a_search_reports_the_question_it_was_not_asked() {
        let corpus: Vec<Vec<f32>> = (0..4).map(unit).collect();
        for objective in [Objective::Primary, Objective::Mentions] {
            let outcome = search(
                &descriptions(),
                &spread_over_four(),
                &centroid_of(&corpus),
                5,
                Calibration::Search,
                objective,
            );
            assert!(
                outcome.primary_agreement > 0.0,
                "{objective:?} reports no agreement"
            );
            assert!(
                outcome.mention_macro_f1 > 0.0,
                "{objective:?} reports no F1"
            );
        }
    }

    #[test]
    fn a_perfect_set_scores_perfectly_on_reviews_it_was_never_fitted_to() {
        let examples = spread_over_four();
        let fitted = descriptions().fit(&examples, &[], FitParams::default());
        let scored = score(&fitted, &examples, crate::classify::DEFAULT_MENTION_MARGIN);

        assert_eq!(scored.compared, 40);
        assert!(
            (scored.primary_agreement() - 1.0).abs() < 1e-9,
            "four separated categories should never be confused: {scored:?}"
        );
        assert_eq!(
            score(&fitted, &[], crate::classify::DEFAULT_MENTION_MARGIN).compared,
            0
        );
        assert!(
            (score(&fitted, &[], crate::classify::DEFAULT_MENTION_MARGIN).primary_agreement())
                .abs()
                < 1e-9
        );
    }

    fn spread_over_four() -> Vec<Example> {
        (0..40)
            .map(|index| Example {
                vector: unit(index % 4),
                primary: index % 4,
                secondary: vec![],
            })
            .collect()
    }
}
