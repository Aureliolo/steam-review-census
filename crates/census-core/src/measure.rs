//! Comparing what the model read against what a labeller said.
//!
//! A census that cannot say how often it is wrong is an opinion with decimal places. This is
//! where the figure comes from: stored readings joined to a claim reference set, per subject,
//! with the two things that make the number honest kept separate from it.
//!
//! The first is **abstention**. The model declines claims it is not sure about, and a score
//! that quietly drops those is a score for a classifier nobody is running. Declined claims are
//! counted and reported, and the agreement figure says over how many it was computed.
//!
//! The second is **contest**. A labeller marks a claim ambiguous when two subjects genuinely
//! both fit and the rules do not settle which. Agreement on those says as much about the
//! taxonomy as about the model, so it is reported apart from the rest rather than averaged in.

use std::{collections::HashMap, path::Path};

use serde::Serialize;

use crate::{Result, claimset::ClaimLabel, taxonomy::CORE_SPINE};

/// One subject's agreement.
#[derive(Debug, Clone, Serialize)]
pub struct SubjectAgreement {
    pub id: &'static str,
    pub label: &'static str,
    /// Claims the labeller put here.
    pub labelled: u64,
    /// Claims the model put here.
    pub read: u64,
    /// Claims both put here.
    pub agreed: u64,
    /// The subject this one is most often read as instead, where they disagree. A subject
    /// read as one particular other subject is a boundary the taxonomy has not settled, and
    /// no amount of training settles it for the taxonomy.
    pub mistaken_for: Option<(&'static str, u64)>,
}

impl SubjectAgreement {
    #[must_use]
    #[expect(clippy::cast_precision_loss, reason = "label counts are small")]
    pub fn precision(&self) -> Option<f64> {
        (self.read > 0).then(|| self.agreed as f64 / self.read as f64)
    }

    #[must_use]
    #[expect(clippy::cast_precision_loss, reason = "label counts are small")]
    pub fn recall(&self) -> Option<f64> {
        (self.labelled > 0).then(|| self.agreed as f64 / self.labelled as f64)
    }

    #[must_use]
    pub fn f1(&self) -> Option<f64> {
        let (precision, recall) = (self.precision()?, self.recall()?);
        (precision + recall > 0.0).then(|| 2.0 * precision * recall / (precision + recall))
    }
}

/// How well the model and the labels agree over one game.
#[derive(Debug, Clone, Serialize)]
pub struct ClaimAgreement {
    pub app_id: u32,
    /// Labelled claims found in the readings at all.
    pub matched: u64,
    /// Of those, the ones the model would put a subject on.
    pub answered: u64,
    pub agreed: u64,
    /// Claims the model declined. Not wrong, and not right: it said nothing.
    pub declined: u64,
    pub polarity_answered: u64,
    pub polarity_agreed: u64,
    pub clear_answered: u64,
    pub clear_agreed: u64,
    pub contested_answered: u64,
    pub contested_agreed: u64,
    pub subjects: Vec<SubjectAgreement>,
}

impl ClaimAgreement {
    #[must_use]
    #[expect(clippy::cast_precision_loss, reason = "label counts are small")]
    pub fn rate(&self) -> Option<f64> {
        (self.answered > 0).then(|| self.agreed as f64 / self.answered as f64)
    }

    #[must_use]
    #[expect(clippy::cast_precision_loss, reason = "label counts are small")]
    pub fn declined_share(&self) -> Option<f64> {
        (self.matched > 0).then(|| self.declined as f64 / self.matched as f64)
    }

    #[must_use]
    #[expect(clippy::cast_precision_loss, reason = "label counts are small")]
    pub fn polarity_rate(&self) -> Option<f64> {
        (self.polarity_answered > 0)
            .then(|| self.polarity_agreed as f64 / self.polarity_answered as f64)
    }

    /// Macro F1 over the subjects a labeller actually used.
    ///
    /// Macro rather than weighted, because a corpus is mostly `verdict` and weighting by
    /// support would let one category the model finds easy carry the score for the rest.
    #[must_use]
    pub fn macro_f1(&self) -> Option<f64> {
        let scored: Vec<f64> = self
            .subjects
            .iter()
            .filter(|subject| subject.labelled > 0)
            .filter_map(SubjectAgreement::f1)
            .collect();
        #[expect(clippy::cast_precision_loss, reason = "at most a few dozen subjects")]
        (!scored.is_empty()).then(|| scored.iter().sum::<f64>() / scored.len() as f64)
    }
}

/// Scores one game's stored readings against its claim labels.
///
/// # Errors
///
/// Fails if the labels or the readings are missing or unreadable.
pub fn agreement(out_dir: &Path, app_id: u32, reference: &Path) -> Result<ClaimAgreement> {
    let labels: Vec<ClaimLabel> =
        serde_json::from_slice(&std::fs::read(reference.join("labels.json")).map_err(|_| {
            crate::Error::NoReferenceSet {
                path: reference.join("labels.json"),
            }
        })?)?;

    let snapshot = crate::embed::latest_snapshot(out_dir, app_id)?;
    let wanted: HashMap<(&str, u16), &ClaimLabel> = labels
        .iter()
        .map(|label| ((label.review_id.as_str(), label.index), label))
        .collect();

    let mut read: HashMap<(String, u16), (Option<String>, String)> = HashMap::new();
    crate::read::for_each_reading(
        &snapshot.join("readings.parquet"),
        |id, index, subject, _, polarity| {
            if wanted.contains_key(&(id, index)) {
                read.insert(
                    (id.to_owned(), index),
                    (subject.map(ToOwned::to_owned), polarity.to_owned()),
                );
            }
        },
    )?;

    let position = |id: &str| CORE_SPINE.iter().position(|category| category.id == id);
    let mut labelled = vec![0_u64; CORE_SPINE.len()];
    let mut said = vec![0_u64; CORE_SPINE.len()];
    let mut agreed_per = vec![0_u64; CORE_SPINE.len()];
    let mut confused: Vec<HashMap<usize, u64>> = vec![HashMap::new(); CORE_SPINE.len()];

    let mut found = ClaimAgreement {
        app_id,
        matched: 0,
        answered: 0,
        agreed: 0,
        declined: 0,
        polarity_answered: 0,
        polarity_agreed: 0,
        clear_answered: 0,
        clear_agreed: 0,
        contested_answered: 0,
        contested_agreed: 0,
        subjects: Vec::new(),
    };

    for label in &labels {
        let Some((subject, polarity)) = read.get(&(label.review_id.clone(), label.index)) else {
            continue;
        };
        found.matched += 1;
        let Some(truth) = position(&label.subject) else {
            continue;
        };
        labelled[truth] += 1;

        let Some(guessed) = subject.as_deref().and_then(position) else {
            found.declined += 1;
            continue;
        };
        found.answered += 1;
        said[guessed] += 1;

        if guessed == truth {
            found.agreed += 1;
            agreed_per[truth] += 1;
        } else {
            *confused[truth].entry(guessed).or_default() += 1;
        }

        if label.ambiguous {
            found.contested_answered += 1;
            found.contested_agreed += u64::from(guessed == truth);
        } else {
            found.clear_answered += 1;
            found.clear_agreed += u64::from(guessed == truth);
        }

        found.polarity_answered += 1;
        found.polarity_agreed += u64::from(polarity == &label.polarity);
    }

    found.subjects = CORE_SPINE
        .iter()
        .enumerate()
        .map(|(index, category)| SubjectAgreement {
            id: category.id,
            label: category.label,
            labelled: labelled[index],
            read: said[index],
            agreed: agreed_per[index],
            mistaken_for: confused[index]
                .iter()
                .max_by_key(|(_, count)| **count)
                .and_then(|(other, count)| {
                    CORE_SPINE.get(*other).map(|named| (named.label, *count))
                }),
        })
        .collect();

    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subject(labelled: u64, read: u64, agreed: u64) -> SubjectAgreement {
        SubjectAgreement {
            id: "gameplay",
            label: "Gameplay",
            labelled,
            read,
            agreed,
            mistaken_for: None,
        }
    }

    #[test]
    fn a_subject_nobody_labelled_scores_nothing_rather_than_perfectly() {
        let empty = subject(0, 0, 0);
        assert_eq!(empty.recall(), None);
        assert_eq!(empty.f1(), None);
    }

    #[test]
    fn precision_and_recall_are_the_two_ways_of_being_wrong() {
        // Twenty labelled, the model said forty, thirty of which were something else.
        let eager = subject(20, 40, 10);
        assert!((eager.precision().unwrap() - 0.25).abs() < 1e-9);
        assert!((eager.recall().unwrap() - 0.5).abs() < 1e-9);
        assert!((eager.f1().unwrap() - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn declining_is_neither_agreement_nor_disagreement() {
        let found = ClaimAgreement {
            app_id: 1,
            matched: 100,
            answered: 60,
            agreed: 45,
            declined: 40,
            polarity_answered: 60,
            polarity_agreed: 48,
            clear_answered: 50,
            clear_agreed: 40,
            contested_answered: 10,
            contested_agreed: 5,
            subjects: Vec::new(),
        };
        assert!((found.rate().unwrap() - 0.75).abs() < 1e-9);
        assert!(
            (found.declined_share().unwrap() - 0.4).abs() < 1e-9,
            "a score that hides what it refused to answer is not a score"
        );
    }
}
