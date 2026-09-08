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
use serde::Deserialize;

use crate::{Error, Result, taxonomy::CORE_SPINE};

/// One review's reference labels.
#[derive(Debug, Clone, Deserialize)]
pub struct ReferenceLabel {
    pub id: String,
    pub primary: String,
    #[serde(default)]
    pub secondary: Vec<String>,
    #[serde(default)]
    pub ironic: bool,
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
}

impl CategoryAgreement {
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

    /// `None` only when precision or recall is undefined, never merely because both are
    /// zero. Returning `None` for a category the classifier gets entirely wrong would drop
    /// it from the macro average and make the total look better the worse that category is.
    #[must_use]
    pub fn f1(&self) -> Option<f64> {
        let (p, r) = (self.precision()?, self.recall()?);
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
    ///
    /// Reference sets are small: the randomly drawn subset of the only set that exists is
    /// sixty reviews, where six reviews changing hands moves the headline ten points. A bare
    /// percentage invites reading such a swing as an improvement, so every rate reported
    /// here carries the range it is actually entitled to claim. Wilson rather than the
    /// textbook normal interval, which misbehaves badly at these counts and happily returns
    /// bounds outside zero to one.
    #[must_use]
    pub fn interval(&self) -> Option<(f64, f64)> {
        const Z: f64 = 1.959_963_985;
        let hits = self.agreement()?;
        #[expect(
            clippy::cast_precision_loss,
            reason = "reference sets are a few hundred reviews"
        )]
        let n = self.compared as f64;
        let denominator = Z.mul_add(Z / n, 1.0);
        let centre = hits + Z * Z / (2.0 * n);
        let spread = Z * (hits * (1.0 - hits) / n + Z * Z / (4.0 * n * n)).sqrt();
        Some((
            ((centre - spread) / denominator).max(0.0),
            ((centre + spread) / denominator).min(1.0),
        ))
    }
}

#[derive(Debug, Clone)]
pub struct AgreementReport {
    pub app_id: u32,
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

/// Compares stored classifications against a reference set.
///
/// # Errors
///
/// Fails if the reference set or the classifications cannot be read.
pub fn compare(reference: &ReferenceSet, out_dir: &Path, app_id: u32) -> Result<AgreementReport> {
    let snapshot = crate::embed::latest_snapshot(out_dir, app_id)?;
    let predictions = load_predictions(&snapshot.join("classifications.parquet"))?;

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
        app_id,
        produced_by: reference.produced_by.clone(),
        compared,
        unmatched,
        primary_agreement: rate(primary_agreed, compared),
        anchors,
        slices,
        categories: stats,
    })
}

struct Prediction {
    primary: String,
    mentions: Vec<String>,
    anchors: String,
}

fn load_predictions(path: &Path) -> Result<HashMap<String, Prediction>> {
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
        let anchors = downcast::<StringArray>(&batch, "anchors")?;
        for row in 0..batch.num_rows() {
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
            app_id: 1,
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
