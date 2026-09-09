//! Measuring the classifier against a reference set.
//!
//! What this produces is **agreement with the reference labels**, not accuracy. The
//! distinction is not pedantry: the reference set here was produced by a language model,
//! and a model checking a model measures consistency between them, not correctness. A
//! category where the two agree 90% of the time may still be 90% wrong in the same
//! direction.
//!
//! Agreement is still worth measuring. It bounds how much the cheap similarity path gives
//! up against a model that actually reads, and it finds categories the two disagree about
//! systematically, which is where the taxonomy is usually at fault. It just cannot be
//! reported as accuracy until some of the reference set is checked by a human.

use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
};

use arrow::array::{Array, ListArray, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};

use crate::{Error, Result, taxonomy::CORE_SPINE};

/// One review's reference labels.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ReferenceLabel {
    pub id: String,
    pub primary: String,
    #[serde(default)]
    pub secondary: Vec<String>,
    /// Whether the text says the opposite of what it appears to say.
    ///
    /// A claim about the words alone. The labeller is never shown whether the reviewer
    /// recommended the game, so this cannot be, and must not be read as, a report that the
    /// rating and the text disagree: that is this flag joined to the rating, and the join is
    /// the tool's job.
    #[serde(default)]
    pub ironic: bool,
    /// How sure the labeller was of the primary category: high, medium or low.
    ///
    /// About the labeller, where [`Self::ambiguous`] is about the review. Recorded and not
    /// yet read by anything.
    #[serde(default)]
    pub confidence: String,
    /// How this review entered the sample. A set stratified by predicted category
    /// deliberately over-represents categories the classifier rarely picks, so a single
    /// blended agreement figure over such a set understates corpus-wide agreement. Only
    /// the randomly drawn subset estimates what the corpus looks like.
    #[serde(default)]
    pub subset: String,
    /// Whether whoever produced the label said the call was genuinely contested.
    ///
    /// Disagreement on these says as much about the taxonomy as about the classifier, so
    /// mixing them into one figure hides which of the two is at fault.
    #[serde(default)]
    pub ambiguous: bool,
}

/// A reference set as it is stored on disk.
///
/// Split across two files on purpose. `manifest.json` is written by hand and says where the
/// labels came from; `labels.json` is generated. Keeping the provenance out of the
/// generated file makes it harder to lose track of what produced a set, which is the one
/// fact that determines what its numbers may be called.
#[derive(Debug, Clone, Deserialize)]
pub struct ReferenceSet {
    pub app_id: u32,
    /// What produced these labels. Printed with every result so nobody mistakes a
    /// model-labelled set for a human-labelled one.
    pub produced_by: String,
    pub spine_version: String,
    /// Whether any part of this set was checked by a person. Until this is true, results
    /// are agreement, never accuracy.
    #[serde(default)]
    pub human_verified: bool,
    #[serde(default, skip_deserializing)]
    pub labels: Vec<ReferenceLabel>,
}

impl ReferenceSet {
    /// # Errors
    ///
    /// Fails if either file is missing or malformed.
    pub fn load(dir: &Path) -> Result<Self> {
        let read = |name: &str| -> Result<Vec<u8>> {
            std::fs::read(dir.join(name)).map_err(|_| Error::NoReferenceSet {
                path: dir.join(name),
            })
        };
        let mut set: Self = serde_json::from_slice(&read("manifest.json")?)?;
        set.labels = serde_json::from_slice(&read("labels.json")?)?;
        Ok(set)
    }

    /// Whether every label names a category the taxonomy actually has.
    #[must_use]
    pub fn unknown_categories(&self) -> Vec<String> {
        let known: HashSet<&str> = CORE_SPINE.iter().map(|c| c.id).collect();
        let mut unknown: Vec<String> = self
            .labels
            .iter()
            .flat_map(|l| std::iter::once(&l.primary).chain(l.secondary.iter()))
            .filter(|id| !known.contains(id.as_str()))
            .cloned()
            .collect();
        unknown.sort_unstable();
        unknown.dedup();
        unknown
    }
}

#[derive(Debug, Clone)]
pub struct CategoryAgreement {
    pub id: &'static str,
    pub label: &'static str,
    /// Reviews the reference set puts in this category as their main subject.
    pub reference_primary: u64,
    /// Reviews the classifier puts here as their main subject.
    pub predicted_primary: u64,
    pub primary_agreed: u64,
    pub reference_mentions: u64,
    pub predicted_mentions: u64,
    pub mention_agreed: u64,
    /// Where the reviews the labels put here actually went, in [`CORE_SPINE`] order.
    ///
    /// The diagonal is the agreements. Knowing a category scores badly says nothing about
    /// what to do; knowing it is read as one particular other category is a boundary rule
    /// waiting to be written, and that is a fact about the taxonomy rather than the model.
    pub taken_as: Vec<u64>,
}

impl CategoryAgreement {
    /// The category this one is most often mistaken for, where that is a pattern rather than
    /// a coincidence.
    ///
    /// One review going one way is a tie broken by taxonomy order, and naming it would turn
    /// a list of arbitrary neighbours into something that reads like a finding. Reported only
    /// when it is several reviews and a real share of the ones this category loses.
    #[must_use]
    pub fn mistaken_for(&self) -> Option<(&'static str, u64)> {
        const WORTH_NAMING: u64 = 3;
        const SHARE_OF_THE_MISSES: f64 = 0.25;

        let mine = CORE_SPINE.iter().position(|c| c.id == self.id);
        let missed: u64 = self
            .taken_as
            .iter()
            .enumerate()
            .filter(|(slot, _)| Some(*slot) != mine)
            .map(|(_, count)| count)
            .sum();
        let (slot, count) = self
            .taken_as
            .iter()
            .enumerate()
            .filter(|(slot, _)| Some(*slot) != mine)
            .max_by_key(|(_, count)| **count)?;
        if *count < WORTH_NAMING || rate(*count, missed)? < SHARE_OF_THE_MISSES {
            return None;
        }
        Some((CORE_SPINE.get(slot)?.label, *count))
    }
    /// Of the mentions the classifier claims, the share the reference set also has.
    #[must_use]
    pub fn precision(&self) -> Option<f64> {
        rate(self.mention_agreed, self.predicted_mentions)
    }

    /// Of the mentions the reference set has, the share the classifier found.
    #[must_use]
    pub fn recall(&self) -> Option<f64> {
        rate(self.mention_agreed, self.reference_mentions)
    }

    /// `None` only when there is nothing at all to be right or wrong about: the reference
    /// set never raises the category and the classifier never files anything under it.
    ///
    /// A category the classifier files nothing under, where the reference set does raise it,
    /// has undefined precision and is nonetheless the worst outcome available. Reporting
    /// that as unmeasurable drops it from the macro average, which makes the average rise
    /// the more categories go completely missing.
    #[must_use]
    pub fn f1(&self) -> Option<f64> {
        if self.reference_mentions == 0 && self.predicted_mentions == 0 {
            return None;
        }
        let (p, r) = (
            self.precision().unwrap_or(0.0),
            self.recall().unwrap_or(0.0),
        );
        Some(if p + r > 0.0 {
            2.0 * p * r / (p + r)
        } else {
            0.0
        })
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "review counts are far below 2^53"
)]
fn rate(part: u64, whole: u64) -> Option<f64> {
    (whole > 0).then(|| part as f64 / whole as f64)
}

/// Agreement over one slice of the reference set.
///
/// Sliced by sampling subset first and only then by contestedness, because the two cuts are
/// not independent and a contested-versus-clear split taken across the whole set would
/// average a randomly drawn sample together with one deliberately enriched for categories
/// the classifier rarely picks. Once anchors are fitted the nesting matters more still: the
/// stratified subset is training data, and any figure computed over it says what the
/// anchors memorised rather than what they know.
#[derive(Debug, Clone)]
pub struct Slice {
    pub subset: String,
    /// `None` for the subset taken whole, otherwise whether the labeller called the reviews
    /// in this slice contested.
    pub contested: Option<bool>,
    pub compared: u64,
    pub agreed: u64,
}

impl Slice {
    #[must_use]
    pub fn agreement(&self) -> Option<f64> {
        rate(self.agreed, self.compared)
    }

    /// 95% Wilson score interval for the agreement rate.
    #[must_use]
    pub fn interval(&self) -> Option<(f64, f64)> {
        wilson(self.agreed, self.compared)
    }
}

/// 95% Wilson score interval for a proportion.
///
/// Reference sets are small: a randomly drawn subset runs to a hundred reviews or so, where
/// six reviews changing hands moves the headline six points. A bare percentage invites
/// reading such a swing as an improvement, so every rate reported anywhere in this tool
/// carries the range it is actually entitled to claim. Wilson rather than the textbook
/// normal interval, which misbehaves badly at these counts and happily returns bounds
/// outside zero to one.
#[must_use]
pub fn wilson(part: u64, whole: u64) -> Option<(f64, f64)> {
    const Z: f64 = 1.959_963_985;
    let hits = rate(part, whole)?;
    #[expect(
        clippy::cast_precision_loss,
        reason = "reference sets are a few hundred reviews"
    )]
    let n = whole as f64;
    let denominator = Z.mul_add(Z / n, 1.0);
    let centre = hits + Z * Z / (2.0 * n);
    let spread = Z * (hits * (1.0 - hits) / n + Z * Z / (4.0 * n * n)).sqrt();
    Some((
        ((centre - spread) / denominator).max(0.0),
        ((centre + spread) / denominator).min(1.0),
    ))
}

/// Two-sided exact McNemar test for two classifiers judged on the same reviews.
///
/// `gained` is the reviews the second gets right and the first does not, `lost` the reverse.
/// Reviews both place the same way carry no information about which is better and are
/// deliberately absent from the arithmetic, which is what makes this the right test for a
/// paired comparison and an unpaired interval the wrong one.
///
/// Exact rather than the usual chi-squared approximation: reference sets of this size swap
/// a couple of dozen reviews, and the approximation is unreliable there. `None` when nothing
/// changed hands, where there is no comparison to make rather than a perfect tie.
#[must_use]
pub fn mcnemar_exact(gained: u64, lost: u64) -> Option<f64> {
    let swapped = gained + lost;
    if swapped == 0 {
        return None;
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "reference sets are a few hundred reviews"
    )]
    let n = swapped as f64;
    // The tail of a fair binomial, summed by the ratio between neighbouring terms so no
    // factorial is ever formed.
    let mut term = 0.5_f64.powf(n);
    let mut tail = term;
    for k in 0..gained.min(lost) {
        #[expect(
            clippy::cast_precision_loss,
            reason = "reference sets are a few hundred reviews"
        )]
        let step = (swapped - k) as f64 / (k + 1) as f64;
        term *= step;
        tail += term;
    }
    Some((2.0 * tail).min(1.0))
}

#[derive(Debug, Clone)]
pub struct AgreementReport {
    /// The games behind this comparison. One per game evaluated, several once pooled.
    pub apps: Vec<u32>,
    pub produced_by: String,
    pub compared: u64,
    /// Reference labels with no matching classification, usually because the corpus was
    /// re-crawled or classified under a different taxonomy.
    pub unmatched: u64,
    pub primary_agreement: Option<f64>,
    /// Which anchors produced the classifications being compared, as recorded when they were
    /// written. More than one value means the file mixes runs and none of it can be trusted.
    pub anchors: Vec<String>,
    /// Each sampling subset whole, followed by its clear and contested slices.
    pub slices: Vec<Slice>,
    pub categories: Vec<CategoryAgreement>,
}

impl AgreementReport {
    /// Mean F1 across categories the reference set actually uses.
    ///
    /// Unweighted on purpose: a rare category the classifier never finds should drag this
    /// down as hard as a common one, because that is exactly the failure a corpus-weighted
    /// average would hide.
    #[must_use]
    pub fn macro_f1(&self) -> Option<f64> {
        let scores: Vec<f64> = self
            .categories
            .iter()
            .filter(|c| c.reference_mentions > 0)
            .filter_map(CategoryAgreement::f1)
            .collect();
        #[expect(
            clippy::cast_precision_loss,
            reason = "the taxonomy has a few dozen categories at most"
        )]
        (!scores.is_empty()).then(|| scores.iter().sum::<f64>() / scores.len() as f64)
    }
}

/// Sums several games' comparisons into one.
///
/// Six sets of a hundred held-out reviews are a six-hundred-review measurement, and the
/// pooled figure is the one worth quoting: at a hundred reviews the honest band around a
/// rate is about twenty points wide, which hides most of what any change does. Pooling also
/// stops one game's idiosyncrasies from reading as a property of the classifier.
///
/// Categories are summed as counts rather than averaged as rates, so a category with four
/// mentions in one game and forty in another counts for what it actually is.
#[must_use]
pub fn pooled(reports: &[AgreementReport]) -> AgreementReport {
    let mut categories: Vec<CategoryAgreement> = CORE_SPINE
        .iter()
        .map(|c| CategoryAgreement {
            id: c.id,
            label: c.label,
            reference_primary: 0,
            predicted_primary: 0,
            primary_agreed: 0,
            reference_mentions: 0,
            predicted_mentions: 0,
            mention_agreed: 0,
            taken_as: vec![0; CORE_SPINE.len()],
        })
        .collect();
    let mut slices: Vec<Slice> = Vec::new();
    let mut apps = Vec::new();
    let mut anchors: Vec<String> = Vec::new();
    let mut produced: Vec<String> = Vec::new();
    let (mut compared, mut unmatched, mut agreed) = (0, 0, 0);

    for report in reports {
        apps.extend(report.apps.iter().copied());
        anchors.extend(report.anchors.iter().cloned());
        produced.push(report.produced_by.clone());
        compared += report.compared;
        unmatched += report.unmatched;
        for from in &report.categories {
            // Matched by id rather than by position: a report from another build may hold a
            // different set of categories, and adding a row into the wrong category is the
            // kind of mistake that produces a plausible number.
            let Some(into) = categories.iter_mut().find(|into| into.id == from.id) else {
                continue;
            };
            into.reference_primary += from.reference_primary;
            into.predicted_primary += from.predicted_primary;
            into.primary_agreed += from.primary_agreed;
            into.reference_mentions += from.reference_mentions;
            into.predicted_mentions += from.predicted_mentions;
            into.mention_agreed += from.mention_agreed;
            for (slot, count) in from.taken_as.iter().enumerate() {
                if let Some(into) = into.taken_as.get_mut(slot) {
                    *into += count;
                }
            }
        }
        agreed += report
            .categories
            .iter()
            .map(|c| c.primary_agreed)
            .sum::<u64>();
        for slice in &report.slices {
            match slices
                .iter_mut()
                .find(|into| into.subset == slice.subset && into.contested == slice.contested)
            {
                Some(into) => {
                    into.compared += slice.compared;
                    into.agreed += slice.agreed;
                }
                None => slices.push(slice.clone()),
            }
        }
    }

    anchors.sort_unstable();
    anchors.dedup();
    produced.sort_unstable();
    produced.dedup();
    AgreementReport {
        apps,
        produced_by: produced.join("; "),
        compared,
        unmatched,
        primary_agreement: rate(agreed, compared),
        anchors,
        slices,
        categories,
    }
}

/// Compares stored classifications against a reference set.
///
/// # Errors
///
/// Fails if the reference set or the classifications cannot be read.
pub fn compare(reference: &ReferenceSet, out_dir: &Path, app_id: u32) -> Result<AgreementReport> {
    let snapshot = crate::embed::latest_snapshot(out_dir, app_id)?;
    // Only the labelled reviews are kept. A reference set names a few hundred reviews, and
    // holding a million predictions to look up four hundred of them costs a hundred times
    // more memory than the answer.
    let wanted: HashSet<&str> = reference.labels.iter().map(|l| l.id.as_str()).collect();
    let predictions = load_predictions(&snapshot.join("classifications.parquet"), &wanted)?;

    let mut stats: Vec<CategoryAgreement> = CORE_SPINE
        .iter()
        .map(|c| CategoryAgreement {
            id: c.id,
            label: c.label,
            reference_primary: 0,
            predicted_primary: 0,
            primary_agreed: 0,
            reference_mentions: 0,
            predicted_mentions: 0,
            mention_agreed: 0,
            taken_as: vec![0; CORE_SPINE.len()],
        })
        .collect();
    let index: HashMap<&str, usize> = CORE_SPINE
        .iter()
        .enumerate()
        .map(|(i, c)| (c.id, i))
        .collect();

    let mut compared = 0_u64;
    let mut unmatched = 0_u64;
    let mut primary_agreed = 0_u64;
    let mut slices: HashMap<(String, Option<bool>), (u64, u64)> = HashMap::new();
    let mut anchors: HashSet<String> = HashSet::new();

    for label in &reference.labels {
        let Some(predicted) = predictions.get(&label.id) else {
            unmatched += 1;
            continue;
        };
        compared += 1;

        if let Some(&slot) = index.get(label.primary.as_str()) {
            stats[slot].reference_primary += 1;
            if let Some(&went) = index.get(predicted.primary.as_str()) {
                stats[slot].taken_as[went] += 1;
            }
        }
        if let Some(&slot) = index.get(predicted.primary.as_str()) {
            stats[slot].predicted_primary += 1;
        }
        anchors.insert(predicted.anchors.clone());
        let agreed = label.primary == predicted.primary;
        for cut in [None, Some(label.ambiguous)] {
            let slice = slices.entry((label.subset.clone(), cut)).or_insert((0, 0));
            slice.0 += 1;
            slice.1 += u64::from(agreed);
        }
        if agreed {
            primary_agreed += 1;
            if let Some(&slot) = index.get(label.primary.as_str()) {
                stats[slot].primary_agreed += 1;
            }
        }

        let reference_mentions: HashSet<&str> = std::iter::once(label.primary.as_str())
            .chain(label.secondary.iter().map(String::as_str))
            .collect();
        let predicted_mentions: HashSet<&str> =
            predicted.mentions.iter().map(String::as_str).collect();

        for (id, &slot) in &index {
            if reference_mentions.contains(id) {
                stats[slot].reference_mentions += 1;
            }
            if predicted_mentions.contains(id) {
                stats[slot].predicted_mentions += 1;
            }
            if reference_mentions.contains(id) && predicted_mentions.contains(id) {
                stats[slot].mention_agreed += 1;
            }
        }
    }

    let mut slices: Vec<Slice> = slices
        .into_iter()
        .map(|((subset, contested), (compared, agreed))| Slice {
            subset,
            contested,
            compared,
            agreed,
        })
        .collect();
    slices.sort_by(|a, b| {
        a.subset
            .cmp(&b.subset)
            .then(a.contested.is_some().cmp(&b.contested.is_some()))
            .then(a.contested.cmp(&b.contested))
    });

    let mut anchors: Vec<String> = anchors.into_iter().collect();
    anchors.sort_unstable();

    Ok(AgreementReport {
        apps: vec![app_id],
        produced_by: reference.produced_by.clone(),
        compared,
        unmatched,
        primary_agreement: rate(primary_agreed, compared),
        anchors,
        slices,
        categories: stats,
    })
}

#[derive(Debug)]
struct Prediction {
    primary: String,
    mentions: Vec<String>,
    anchors: String,
}

fn load_predictions(path: &Path, wanted: &HashSet<&str>) -> Result<HashMap<String, Prediction>> {
    let file = std::fs::File::open(path).map_err(|_| Error::NoCapture {
        path: path.to_path_buf(),
    })?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(8192)
        .build()?;

    let mut out = HashMap::new();
    for batch in reader {
        let batch = batch?;
        let ids = downcast::<StringArray>(&batch, "recommendationid")?;
        let primaries = downcast::<StringArray>(&batch, "primary_category")?;
        let mentions = downcast::<ListArray>(&batch, "mentions")?;
        let anchors = downcast::<StringArray>(&batch, "anchors").map_err(|_| {
            Error::StaleClassifications {
                path: path.to_path_buf(),
                field: "anchors",
            }
        })?;
        for row in 0..batch.num_rows() {
            if !wanted.contains(ids.value(row)) {
                continue;
            }
            let listed = mentions.value(row);
            let listed = listed
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or(Error::MalformedPayload { field: "mentions" })?;
            out.insert(
                ids.value(row).to_owned(),
                Prediction {
                    primary: primaries.value(row).to_owned(),
                    mentions: (0..listed.len())
                        .map(|i| listed.value(i).to_owned())
                        .collect(),
                    anchors: anchors.value(row).to_owned(),
                },
            );
        }
    }
    Ok(out)
}

fn downcast<'a, T: 'static>(
    batch: &'a arrow::record_batch::RecordBatch,
    name: &'static str,
) -> Result<&'a T> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<T>())
        .ok_or(Error::MalformedPayload { field: name })
}

/// Visits every review's primary category, one at a time.
///
/// # Errors
///
/// Fails if the classifications are missing or were written by an older build.
pub fn for_each_prediction(path: &Path, mut visit: impl FnMut(&str, &str)) -> Result<()> {
    let file = std::fs::File::open(path).map_err(|_| Error::NoCapture {
        path: path.to_path_buf(),
    })?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(8192)
        .build()?;

    for batch in reader {
        let batch = batch?;
        let ids = downcast::<StringArray>(&batch, "recommendationid")?;
        let primaries = downcast::<StringArray>(&batch, "primary_category")?;
        for row in 0..batch.num_rows() {
            visit(ids.value(row), primaries.value(row));
        }
    }
    Ok(())
}

/// Streams every stored assignment, main subject and everything else it mentions.
///
/// Separate from [`for_each_prediction`] because reading the mention list costs an array
/// downcast and a slice per row, which a caller that only wants the main subject should not
/// pay for on a million reviews.
///
/// # Errors
///
/// Fails if the file is missing or was written by an older build.
pub fn for_each_assignment(
    path: &Path,
    mut visit: impl FnMut(&str, &str, &[String]),
) -> Result<()> {
    let file = std::fs::File::open(path).map_err(|_| Error::NoCapture {
        path: path.to_path_buf(),
    })?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(8192)
        .build()?;

    // The mention list is a slice inside a batch-owned array, so the names are copied out
    // rather than borrowed: the array is dropped between rows and a borrow would not outlive
    // the visit it was gathered for.
    let mut named: Vec<String> = Vec::new();
    for batch in reader {
        let batch = batch?;
        let ids = downcast::<StringArray>(&batch, "recommendationid")?;
        let primaries = downcast::<StringArray>(&batch, "primary_category")?;
        let mentions = downcast::<ListArray>(&batch, "mentions")?;
        for row in 0..batch.num_rows() {
            let listed = mentions.value(row);
            let listed = listed
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or(Error::MalformedPayload { field: "mentions" })?;
            named.clear();
            for slot in 0..listed.len() {
                if !listed.is_null(slot) {
                    named.push(listed.value(slot).to_owned());
                }
            }
            visit(ids.value(row), primaries.value(row), &named);
        }
    }
    Ok(())
}

/// Where a reference set for an app is expected to live.
#[must_use]
pub fn default_reference_dir(app_id: u32) -> PathBuf {
    PathBuf::from("reference").join(app_id.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slice(compared: u64, agreed: u64) -> Slice {
        Slice {
            subset: "random".to_owned(),
            contested: Some(false),
            compared,
            agreed,
        }
    }

    #[test]
    fn forty_seven_reviews_pin_agreement_no_better_than_a_twenty_seven_point_band() {
        // The measured baseline: 24 of 47 clear calls on the random subset, which reads as
        // 51.1% and means somewhere in [37, 65]. Any claim that a change moved this has to
        // clear a band that wide.
        let (low, high) = slice(47, 24).interval().unwrap();
        assert!((slice(47, 24).agreement().unwrap() - 0.510_638).abs() < 1e-5);
        assert!((low - 0.372).abs() < 0.002, "lower bound was {low}");
        assert!((high - 0.647).abs() < 0.002, "upper bound was {high}");
        assert!(high - low > 0.27, "the band should be this wide at n=47");
    }

    #[test]
    fn intervals_stay_inside_zero_and_one_even_when_every_review_agrees() {
        let (low, high) = slice(12, 12).interval().unwrap();
        assert!(high <= 1.0, "upper bound escaped: {high}");
        assert!(low > 0.7, "twelve for twelve should not admit a low rate");

        let (low, high) = slice(12, 0).interval().unwrap();
        assert!(low >= 0.0, "lower bound escaped: {low}");
        assert!(high < 0.3, "nought for twelve should not admit a high rate");
    }

    #[test]
    fn more_reviews_narrow_the_interval_at_the_same_rate() {
        let narrow = slice(600, 300).interval().unwrap();
        let wide = slice(60, 30).interval().unwrap();
        assert!(
            narrow.1 - narrow.0 < wide.1 - wide.0,
            "ten times the reviews did not buy any precision"
        );
    }

    #[test]
    fn twelve_reviews_gained_against_two_lost_is_the_published_p_value() {
        // The figure the first reference set was reported with: fitting anchors moved 12
        // reviews to the right category and 2 away from it, p = 0.013.
        let p = mcnemar_exact(12, 2).unwrap();
        assert!((p - 0.012_939).abs() < 1e-6, "p was {p}");
    }

    #[test]
    fn an_even_split_is_as_unremarkable_as_a_result_can_be() {
        assert!((mcnemar_exact(7, 7).unwrap() - 1.0).abs() < 1e-12);
        assert!(mcnemar_exact(9, 7).unwrap() > 0.8);
    }

    #[test]
    fn the_same_margin_over_more_reviews_is_the_stronger_evidence() {
        let few = mcnemar_exact(9, 3).unwrap();
        let many = mcnemar_exact(90, 30).unwrap();
        assert!(many < few, "{many} should be far below {few}");
        assert!(few < 0.15 && few > 0.05, "twelve swaps prove little: {few}");
    }

    #[test]
    fn two_sets_that_place_every_review_alike_have_nothing_to_compare() {
        assert_eq!(mcnemar_exact(0, 0), None);
    }

    #[test]
    fn an_empty_slice_has_no_agreement_rather_than_a_zero_one() {
        assert_eq!(slice(0, 0).agreement(), None);
        assert_eq!(slice(0, 0).interval(), None);
    }

    fn agreement(pred: u64, refr: u64, hit: u64) -> CategoryAgreement {
        CategoryAgreement {
            id: "bugs",
            label: "Bugs and crashes",
            reference_primary: 0,
            predicted_primary: 0,
            primary_agreed: 0,
            reference_mentions: refr,
            predicted_mentions: pred,
            mention_agreed: hit,
            taken_as: vec![0; CORE_SPINE.len()],
        }
    }

    #[test]
    fn precision_and_recall_answer_different_questions() {
        // Claimed twenty, reference has ten, eight overlap.
        let stat = agreement(20, 10, 8);
        assert!((stat.precision().unwrap() - 0.4).abs() < 1e-9);
        assert!((stat.recall().unwrap() - 0.8).abs() < 1e-9);
        let f1 = stat.f1().unwrap();
        assert!((f1 - 0.533_333).abs() < 1e-5, "f1 was {f1}");
    }

    /// A confusion count read off the wrong axis still produces a plausible sentence, so
    /// the direction is pinned: this is where the reviews the labels put here ended up.
    #[test]
    fn a_category_reports_what_it_is_mistaken_for_and_not_the_reverse() {
        let slot = |id: &str| CORE_SPINE.iter().position(|c| c.id == id).expect(id);
        let mut stat = agreement(0, 12, 0);
        stat.taken_as = vec![0; CORE_SPINE.len()];
        stat.taken_as[slot("bugs")] = 40;
        stat.taken_as[slot("performance")] = 7;
        stat.taken_as[slot("verdict")] = 3;

        let (label, count) = stat.mistaken_for().expect("seven went to performance");
        assert_eq!(count, 7, "the forty it got right are not a confusion");
        assert_eq!(
            label,
            crate::taxonomy::by_id("performance")
                .expect("in the spine")
                .label
        );

        stat.taken_as = vec![0; CORE_SPINE.len()];
        stat.taken_as[slot("bugs")] = 40;
        assert_eq!(
            stat.mistaken_for(),
            None,
            "a category nothing is confused with has nothing to report"
        );

        // One review going one way is a tie broken by taxonomy order.
        stat.taken_as[slot("performance")] = 1;
        stat.taken_as[slot("verdict")] = 1;
        assert_eq!(
            stat.mistaken_for(),
            None,
            "a single review is not a pattern"
        );

        // Several reviews, but scattered so evenly that no neighbour is the answer.
        stat.taken_as = vec![0; CORE_SPINE.len()];
        for id in ["performance", "verdict", "story", "graphics", "price"] {
            stat.taken_as[slot(id)] = 4;
        }
        assert_eq!(
            stat.mistaken_for(),
            None,
            "a category confused with everything equally is confused with nothing in \
             particular"
        );
    }

    #[test]
    fn a_category_nobody_claims_has_no_score_rather_than_a_perfect_one() {
        let stat = agreement(0, 0, 0);
        assert_eq!(stat.precision(), None);
        assert_eq!(stat.recall(), None);
        assert_eq!(stat.f1(), None);
    }

    #[test]
    fn a_category_the_classifier_never_finds_scores_zero_not_nothing() {
        let stat = agreement(0, 12, 0);
        assert_eq!(stat.precision(), None, "nothing was claimed");
        assert_eq!(stat.recall(), Some(0.0), "twelve were missed");
        assert_eq!(
            stat.f1(),
            Some(0.0),
            "filing nothing under a category the labels do raise is the worst available \
             outcome, and must not leave the category out of the average"
        );
    }

    /// The failure the whole shape of `f1` exists to prevent, stated as an inequality: a set
    /// of anchors that stops finding a category entirely must never score higher for it.
    #[test]
    fn losing_a_category_altogether_cannot_improve_the_average() {
        let categories = |predicted, agreed| AgreementReport {
            apps: vec![1],
            produced_by: "test".to_owned(),
            compared: 100,
            unmatched: 0,
            primary_agreement: Some(0.5),
            anchors: Vec::new(),
            slices: Vec::new(),
            categories: vec![agreement(40, 40, 30), agreement(predicted, 20, agreed)],
        };
        let found = categories(20, 8).macro_f1().expect("two scored categories");
        let lost = categories(0, 0).macro_f1().expect("still two categories");

        assert!(
            lost < found,
            "giving up on a category scored {lost} against {found} for finding some of it"
        );
    }

    #[test]
    fn getting_a_category_entirely_wrong_scores_zero_rather_than_nothing() {
        // Claimed one, reference has ten, none overlap. Both rates are defined and both are
        // zero, so F1 must be zero. Returning None here would quietly drop the category
        // from the macro average and make the total improve as this category got worse.
        let stat = agreement(1, 10, 0);
        assert_eq!(stat.precision(), Some(0.0));
        assert_eq!(stat.recall(), Some(0.0));
        assert_eq!(stat.f1(), Some(0.0));
    }

    #[test]
    fn macro_f1_lets_a_rare_category_drag_the_score_down() {
        let report = AgreementReport {
            apps: vec![1],
            produced_by: "test".to_owned(),
            compared: 100,
            unmatched: 0,
            primary_agreement: Some(0.5),
            anchors: Vec::new(),
            slices: Vec::new(),
            categories: vec![agreement(100, 100, 100), agreement(1, 10, 0)],
        };
        // Corpus-weighted this would look near perfect; unweighted it does not.
        let macro_f1 = report.macro_f1().unwrap();
        assert!(macro_f1 < 0.55, "macro f1 was {macro_f1}");
    }

    #[test]
    fn pooling_adds_counts_and_never_averages_rates() {
        let game = |app: u32, compared, agreed, hits| AgreementReport {
            apps: vec![app],
            produced_by: "test".to_owned(),
            compared,
            unmatched: 1,
            primary_agreement: Some(0.0),
            anchors: vec!["fitted:1".to_owned()],
            slices: vec![
                Slice {
                    subset: "random".to_owned(),
                    contested: None,
                    compared,
                    agreed,
                },
                Slice {
                    subset: "random".to_owned(),
                    contested: Some(false),
                    compared,
                    agreed,
                },
            ],
            categories: vec![CategoryAgreement {
                id: "bugs",
                label: "Bugs and crashes",
                reference_primary: compared,
                predicted_primary: compared,
                primary_agreed: hits,
                reference_mentions: compared,
                predicted_mentions: compared,
                mention_agreed: hits,
                taken_as: vec![0; CORE_SPINE.len()],
            }],
        };
        // A game where nine of ten agree and one where one of ninety does. Averaging the two
        // rates would call that 50%; the truth is ten of a hundred.
        let pooled = pooled(&[game(1, 10, 9, 9), game(2, 90, 1, 1)]);

        assert_eq!(pooled.apps, vec![1, 2]);
        assert_eq!(pooled.compared, 100);
        assert_eq!(pooled.unmatched, 2);
        assert_eq!(pooled.primary_agreement, Some(0.1));
        assert_eq!(pooled.anchors, vec!["fitted:1".to_owned()]);
        assert_eq!(pooled.slices.len(), 2, "slices merged by subset and cut");
        assert_eq!(pooled.slices[0].compared, 100);
        assert_eq!(pooled.slices[0].agreed, 10);
        assert_eq!(pooled.categories[1].reference_mentions, 100);
    }

    #[test]
    fn a_classification_file_from_an_older_build_names_the_fix_rather_than_reading_as_malformed() {
        use std::sync::Arc;

        use arrow::{
            array::{Float32Builder, ListBuilder, StringBuilder, UInt32Builder},
            datatypes::{DataType, Field, Schema},
            record_batch::RecordBatch,
        };
        use parquet::arrow::ArrowWriter;

        // The schema as it stood before runs recorded which anchors produced them.
        let schema = Arc::new(Schema::new(vec![
            Field::new("recommendationid", DataType::Utf8, false),
            Field::new("appid", DataType::UInt32, false),
            Field::new("primary_category", DataType::Utf8, false),
            Field::new("primary_score", DataType::Float32, false),
            Field::new(
                "mentions",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                false,
            ),
            Field::new("spine_version", DataType::Utf8, false),
        ]));

        let mut ids = StringBuilder::new();
        let mut appids = UInt32Builder::new();
        let mut primaries = StringBuilder::new();
        let mut scores = Float32Builder::new();
        let mut mentions = ListBuilder::new(StringBuilder::new());
        let mut spine = StringBuilder::new();
        ids.append_value("1");
        appids.append_value(296_970);
        primaries.append_value("bugs");
        scores.append_value(0.5);
        mentions.values().append_value("bugs");
        mentions.append(true);
        spine.append_value("core-2");

        let dir = std::env::temp_dir().join("census-stale-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("classifications.parquet");
        let mut writer = ArrowWriter::try_new(
            std::fs::File::create(&path).unwrap(),
            Arc::clone(&schema),
            None,
        )
        .unwrap();
        writer
            .write(
                &RecordBatch::try_new(
                    schema,
                    vec![
                        Arc::new(ids.finish()),
                        Arc::new(appids.finish()),
                        Arc::new(primaries.finish()),
                        Arc::new(scores.finish()),
                        Arc::new(mentions.finish()),
                        Arc::new(spine.finish()),
                    ],
                )
                .unwrap(),
            )
            .unwrap();
        writer.close().unwrap();

        let error = load_predictions(&path, &HashSet::from(["1"])).unwrap_err();
        assert!(
            matches!(
                error,
                Error::StaleClassifications {
                    field: "anchors",
                    ..
                }
            ),
            "an older file should name the command that fixes it: {error}"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn unknown_categories_in_a_reference_set_are_reported() {
        let set = ReferenceSet {
            app_id: 1,
            produced_by: "test".to_owned(),
            spine_version: "core-2".to_owned(),
            human_verified: false,
            labels: vec![ReferenceLabel {
                id: "1".to_owned(),
                primary: "bugs".to_owned(),
                secondary: vec!["not-a-category".to_owned()],
                ironic: false,
                confidence: "high".to_owned(),
                subset: "random".to_owned(),
                ambiguous: false,
            }],
        };
        assert_eq!(set.unknown_categories(), vec!["not-a-category".to_owned()]);
    }
}
