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
    /// Labelled claims the model saw at all, here or elsewhere, answered or declined. The
    /// denominator a corrected rate needs, because the rate it corrects is over every claim
    /// in the corpus and not only the ones the model chose to answer.
    pub seen: u64,
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

    /// Of the labelled claims about this subject, the share the model filed here. Declined
    /// claims count against it, because they count against the corpus rate it corrects.
    #[must_use]
    #[expect(clippy::cast_precision_loss, reason = "label counts are small")]
    pub fn sensitivity(&self) -> Option<f64> {
        (self.labelled > 0).then(|| self.agreed as f64 / self.labelled as f64)
    }

    /// Of the labelled claims about anything else, the share the model filed here anyway.
    #[must_use]
    #[expect(clippy::cast_precision_loss, reason = "label counts are small")]
    pub fn false_positive_rate(&self) -> Option<f64> {
        let others = self.seen.saturating_sub(self.labelled);
        (others > 0).then(|| (self.read - self.agreed) as f64 / others as f64)
    }

    /// What the corpus rate would be if the model made no errors, from the errors it is
    /// measured to make.
    ///
    /// An observed rate is what the model said. It is too high by the claims about something
    /// else it filed here, and too low by the claims about this it filed elsewhere or
    /// declined. Both are measured on the labelled claims, and the correction is the one
    /// every prevalence study makes: subtract the false-positive rate and divide by the gap
    /// between sensitivity and false positives.
    ///
    /// `None` when the model finds this subject no more often in claims about it than in
    /// claims about anything else, because then the observed rate carries no information
    /// about the true one and any number produced from it would be invented. That is the
    /// honest answer for a subject with a handful of labels, and it is printed as such.
    #[must_use]
    pub fn corrected(&self, observed: f64) -> Option<f64> {
        let (sensitivity, false_positives) = (self.sensitivity()?, self.false_positive_rate()?);
        let gap = sensitivity - false_positives;
        if gap <= CORRECTABLE_GAP {
            return None;
        }
        Some(((observed - false_positives) / gap).clamp(0.0, 1.0))
    }
}

/// How much better than chance the model must find a subject before its rate is corrected.
///
/// Dividing by a gap near zero turns a rounding error in the measured rates into a corrected
/// prevalence of anything at all. A tenth is far enough from zero that a claim rate moves by
/// at most ten times its own measurement error, which is already a wide answer.
const CORRECTABLE_GAP: f64 = 0.1;

/// How well the model and the labels agree over one game.
#[derive(Debug, Clone, Serialize)]
pub struct ClaimAgreement {
    pub app_id: u32,
    /// Labelled claims found in the readings at all.
    pub matched: u64,
    /// Labelled claims the splitter in this build no longer produces as one claim, so no
    /// reading corresponds to them. A label names a span of a review; a splitter that cuts
    /// that review differently leaves the label pointing at nothing, and it is counted here
    /// rather than at whatever now sits at its old index.
    #[serde(default)]
    pub unjoined: u64,
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

    /// The range the agreement rate is entitled to claim, given how few claims it rests on.
    #[must_use]
    pub fn interval(&self) -> Option<(f64, f64)> {
        wilson(self.agreed, self.answered)
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

/// 95% Wilson score interval for a proportion.
///
/// Reference sets are small: a game contributes a few hundred labelled claims, where six
/// changing hands moves the headline six points. A bare percentage invites reading such a
/// swing as an improvement, so every rate reported anywhere in this tool carries the range it
/// is actually entitled to claim. Wilson rather than the textbook normal interval, which
/// misbehaves badly at these counts and happily returns bounds outside zero to one.
#[must_use]
pub fn wilson(part: u64, whole: u64) -> Option<(f64, f64)> {
    const Z: f64 = 1.959_963_985;
    let hits = rate(part, whole)?;
    #[expect(
        clippy::cast_precision_loss,
        reason = "reference sets are a few hundred claims"
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

#[expect(clippy::cast_precision_loss, reason = "counts are far below 2^53")]
fn rate(part: u64, whole: u64) -> Option<f64> {
    (whole > 0).then(|| part as f64 / whole as f64)
}

/// One figure over several games, by adding the counts rather than averaging the rates.
///
/// Averaging per-game rates would give a game with forty labelled claims the same weight as
/// one with six hundred. Adding the counts first is the figure a reader means by "how often is
/// it wrong", and the per-game numbers stay available beside it for the spread.
#[must_use]
pub fn pooled(games: &[ClaimAgreement]) -> ClaimAgreement {
    let mut total = ClaimAgreement {
        app_id: 0,
        matched: 0,
        unjoined: 0,
        answered: 0,
        agreed: 0,
        declined: 0,
        polarity_answered: 0,
        polarity_agreed: 0,
        clear_answered: 0,
        clear_agreed: 0,
        contested_answered: 0,
        contested_agreed: 0,
        subjects: CORE_SPINE
            .iter()
            .map(|category| SubjectAgreement {
                id: category.id,
                label: category.label,
                labelled: 0,
                read: 0,
                agreed: 0,
                seen: 0,
                mistaken_for: None,
            })
            .collect(),
    };

    for game in games {
        total.matched += game.matched;
        total.unjoined += game.unjoined;
        total.answered += game.answered;
        total.agreed += game.agreed;
        total.declined += game.declined;
        total.polarity_answered += game.polarity_answered;
        total.polarity_agreed += game.polarity_agreed;
        total.clear_answered += game.clear_answered;
        total.clear_agreed += game.clear_agreed;
        total.contested_answered += game.contested_answered;
        total.contested_agreed += game.contested_agreed;
        // By id, not by position. A game scored against an older build's spine has its
        // subjects in a different order, and adding those up by slot would file one subject's
        // claims under another's name without anything failing.
        for from in &game.subjects {
            if let Some(into) = total
                .subjects
                .iter_mut()
                .find(|subject| subject.id == from.id)
            {
                into.labelled += from.labelled;
                into.read += from.read;
                into.agreed += from.agreed;
                into.seen += from.seen;
            }
        }
    }

    // `mistaken_for` stays empty here. Which subject one game's `gameplay` is most often read
    // as instead says something; the same figure summed over games with different mixes of
    // subjects names whichever subject happened to be commonest, which is not a confusion.
    total
}

/// Where each labelled claim sits in the readings, by the span of text it names.
///
/// A label was made against a claim as some splitter cut it, and the readings were made
/// against claims as this build's splitter cuts them. The two agree on an index only while
/// the splitter is the same, and the splitter is meant to improve. What does not change is
/// the text: a label names a span of its review, so the claim it belongs to is whichever
/// claim of the current split covers exactly that span, wherever it now sits. A label whose
/// span no claim covers any more is left out, and counted.
fn join_by_span<'a>(
    snapshot: &Path,
    labels: &'a [ClaimLabel],
) -> Result<HashMap<(&'a str, u16), u16>> {
    let depth = std::fs::read(snapshot.join("reading.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<crate::read::ReadReport>(&bytes).ok())
        .map_or(crate::read::Depth::Deep, |found| found.depth);
    let ids: std::collections::HashSet<String> =
        labels.iter().map(|label| label.review_id.clone()).collect();
    let texts = crate::capture::texts_for(snapshot, &ids)?;

    let mut spans: HashMap<&str, HashMap<(u32, u32), u16>> = HashMap::new();
    for (id, text) in &texts {
        let by_span = depth
            .spans_of(text)
            .into_iter()
            .enumerate()
            .filter_map(|(index, span)| {
                let index = u16::try_from(index).ok()?;
                let (start, end) = (
                    u32::try_from(span.start).ok()?,
                    u32::try_from(span.end).ok()?,
                );
                Some(((start, end), index))
            })
            .collect();
        spans.insert(id.as_str(), by_span);
    }

    Ok(labels
        .iter()
        .filter_map(|label| {
            let now = spans
                .get(label.review_id.as_str())?
                .get(&(label.start, label.end))?;
            Some(((label.review_id.as_str(), label.index), *now))
        })
        .collect())
}

/// What the model said of one claim: its subject, or none where it declined, and its polarity.
type Said = (Option<String>, String);

/// The stored reading of each wanted claim.
fn readings_at(
    snapshot: &Path,
    wanted: &std::collections::HashSet<(&str, u16)>,
) -> Result<HashMap<(String, u16), Said>> {
    let mut read = HashMap::new();
    crate::read::for_each_reading(
        &snapshot.join("readings.parquet"),
        |id, index, subject, _, polarity| {
            if wanted.contains(&(id, index)) {
                read.insert(
                    (id.to_owned(), index),
                    (subject.map(ToOwned::to_owned), polarity.to_owned()),
                );
            }
        },
    )?;
    Ok(read)
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
    let joined = join_by_span(&snapshot, &labels)?;
    let wanted: std::collections::HashSet<(&str, u16)> = labels
        .iter()
        .filter_map(|label| {
            let now = joined.get(&(label.review_id.as_str(), label.index))?;
            Some((label.review_id.as_str(), *now))
        })
        .collect();
    let read = readings_at(&snapshot, &wanted)?;

    let position = |id: &str| CORE_SPINE.iter().position(|category| category.id == id);
    let mut labelled = vec![0_u64; CORE_SPINE.len()];
    let mut said = vec![0_u64; CORE_SPINE.len()];
    let mut agreed_per = vec![0_u64; CORE_SPINE.len()];
    let mut confused: Vec<HashMap<usize, u64>> = vec![HashMap::new(); CORE_SPINE.len()];

    let mut found = ClaimAgreement {
        app_id,
        matched: 0,
        unjoined: 0,
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
        let Some(&now) = joined.get(&(label.review_id.as_str(), label.index)) else {
            found.unjoined += 1;
            continue;
        };
        let Some((subject, polarity)) = read.get(&(label.review_id.clone(), now)) else {
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

    // Every labelled claim with a known subject, which is what a subject's false-positive rate
    // is measured over: the claims about anything else that the model could have filed here.
    let seen: u64 = labelled.iter().sum();
    found.subjects = CORE_SPINE
        .iter()
        .enumerate()
        .map(|(index, category)| SubjectAgreement {
            id: category.id,
            label: category.label,
            labelled: labelled[index],
            read: said[index],
            agreed: agreed_per[index],
            seen,
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
            seen: labelled * 5,
            mistaken_for: None,
        }
    }

    #[test]
    fn a_rate_is_corrected_by_the_errors_the_model_is_measured_to_make() {
        // A hundred labelled about gameplay of five hundred seen. The model found seventy of
        // them and filed forty claims about other things here too, so it reads gameplay at
        // 22% of claims when the truth is 20%. Sensitivity 0.7, false positives 0.1.
        let scored = SubjectAgreement {
            id: "gameplay",
            label: "Gameplay",
            labelled: 100,
            read: 110,
            agreed: 70,
            seen: 500,
            mistaken_for: None,
        };
        let corrected = scored.corrected(0.22).expect("a gap of 0.6 is correctable");
        assert!(
            (corrected - 0.2).abs() < 1e-9,
            "(0.22 - 0.1) / (0.7 - 0.1) is the true rate, got {corrected}"
        );
    }

    #[test]
    fn a_subject_the_model_cannot_find_gets_no_corrected_rate_rather_than_a_wild_one() {
        // Finds it in 12% of claims about it and files 10% of everything else here: barely
        // better than chance. Dividing by the 0.02 gap would turn 0.11 observed into 50%.
        let scored = SubjectAgreement {
            id: "gameplay",
            label: "Gameplay",
            labelled: 100,
            read: 52,
            agreed: 12,
            seen: 500,
            mistaken_for: None,
        };
        assert_eq!(scored.corrected(0.11), None);
    }

    #[test]
    fn a_corrected_rate_stays_a_rate() {
        // Observed below the false-positive rate, which a small corpus can produce by chance.
        // The arithmetic says negative; a rate cannot be.
        let scored = subject(100, 90, 80);
        assert_eq!(scored.corrected(0.0), Some(0.0));
        assert_eq!(scored.corrected(1.0), Some(1.0));
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
            unjoined: 0,
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

    #[test]
    fn a_rate_from_few_claims_admits_how_wide_it_is() {
        let (low, high) = wilson(9, 10).expect("ten claims is a proportion");
        assert!(low > 0.55 && low < 0.60, "lower bound was {low}");
        assert!(high > 0.97 && high < 0.99, "upper bound was {high}");

        let (tight_low, tight_high) = wilson(900, 1000).expect("a thousand claims too");
        assert!(
            tight_high - tight_low < high - low,
            "a hundred times the claims must narrow the range"
        );
    }

    #[test]
    fn an_interval_never_leaves_zero_to_one() {
        let (low, high) = wilson(0, 5).expect("nothing agreed is still a proportion");
        assert!(low >= 0.0 && high <= 1.0);
        let (low, high) = wilson(5, 5).expect("everything agreed too");
        assert!(low >= 0.0 && high <= 1.0);
    }

    #[test]
    fn nothing_measured_has_no_interval_rather_than_a_wide_one() {
        assert_eq!(wilson(0, 0), None);
    }

    #[test]
    fn pooling_adds_the_claims_rather_than_averaging_the_rates() {
        let small = ClaimAgreement {
            app_id: 1,
            matched: 10,
            unjoined: 0,
            answered: 10,
            agreed: 10,
            declined: 0,
            polarity_answered: 10,
            polarity_agreed: 10,
            clear_answered: 10,
            clear_agreed: 10,
            contested_answered: 0,
            contested_agreed: 0,
            subjects: vec![subject(10, 10, 10)],
        };
        let large = ClaimAgreement {
            app_id: 2,
            matched: 990,
            unjoined: 0,
            answered: 990,
            agreed: 495,
            declined: 0,
            polarity_answered: 990,
            polarity_agreed: 495,
            clear_answered: 990,
            clear_agreed: 495,
            contested_answered: 0,
            contested_agreed: 0,
            subjects: vec![subject(990, 990, 495)],
        };

        let both = pooled(&[small, large]);
        let rate = both.rate().expect("a thousand answered claims");
        assert!(
            (rate - 0.505).abs() < 1e-9,
            "pooling gave {rate}; averaging the two games' rates would have given 0.75, which \
             is a game of ten claims outvoting one of nine hundred and ninety"
        );
    }

    #[test]
    fn pooling_nothing_is_empty_rather_than_a_panic() {
        let none = pooled(&[]);
        assert_eq!(none.rate(), None);
        assert_eq!(none.subjects.len(), CORE_SPINE.len());
    }

    #[test]
    fn pooling_finds_a_subject_by_name_wherever_it_sits() {
        let last = CORE_SPINE.last().expect("the spine is not empty");
        let odd = ClaimAgreement {
            app_id: 3,
            matched: 4,
            unjoined: 0,
            answered: 4,
            agreed: 3,
            declined: 0,
            polarity_answered: 4,
            polarity_agreed: 4,
            clear_answered: 4,
            clear_agreed: 3,
            contested_answered: 0,
            contested_agreed: 0,
            subjects: vec![SubjectAgreement {
                id: last.id,
                label: last.label,
                labelled: 4,
                read: 4,
                agreed: 3,
                seen: 4,
                mistaken_for: None,
            }],
        };

        let both = pooled(&[odd]);
        let landed = both
            .subjects
            .iter()
            .find(|subject| subject.id == last.id)
            .expect("the spine has this subject");
        assert_eq!(
            (landed.labelled, landed.agreed),
            (4, 3),
            "a subject given on its own must land under its own name, not in the first slot"
        );
        assert!(
            both.subjects
                .iter()
                .filter(|subject| subject.id != last.id)
                .all(|subject| subject.labelled == 0),
            "and nowhere else"
        );
    }
}
