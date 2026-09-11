//! Two labellers on the same claims, and how far apart they are.
//!
//! Every figure this tool reports rests on labels one model wrote, and a set labelled once
//! cannot say anything about its own reliability. A tenth of it is labelled a second time by a
//! different labeller, and this is what reads the two together.
//!
//! Raw agreement is not enough on its own. A corpus is mostly `verdict` and `offtopic`, so two
//! labellers who never read a claim would still agree most of the time by landing on the
//! commonest subject. Cohen's kappa subtracts that: it asks how much of the agreement is more
//! than the two labellers' own habits would produce by chance.
//!
//! The fields are reported apart from each other because they fail differently. Subject is a
//! judgement about the claim; `ambiguous` is a judgement about the taxonomy, and the labelled
//! sets so far disagree about it far more than they disagree about anything else.

use std::collections::HashMap;

use serde::Serialize;

use crate::{Result, claimset::ClaimLabel};

/// How two labellers compare on one field.
#[derive(Debug, Clone, Serialize)]
pub struct FieldAgreement {
    pub field: &'static str,
    pub compared: u64,
    pub agreed: u64,
    /// What the two agreed on beyond what their own habits would produce by chance. Below
    /// zero means they agree less than two labellers picking at random from their own
    /// distributions, which is worse than it sounds and worth seeing.
    pub kappa: Option<f64>,
    /// The disagreement they make most often, as the two values and how many times.
    pub commonest_split: Option<(String, String, u64)>,
}

impl FieldAgreement {
    #[must_use]
    #[expect(clippy::cast_precision_loss, reason = "label counts are small")]
    pub fn rate(&self) -> Option<f64> {
        (self.compared > 0).then(|| self.agreed as f64 / self.compared as f64)
    }

    /// The range the rate is entitled to claim, given how few claims it rests on.
    #[must_use]
    pub fn interval(&self) -> Option<(f64, f64)> {
        crate::measure::wilson(self.agreed, self.compared)
    }
}

/// How two labellings of the same claims compare, field by field.
#[derive(Debug, Clone, Serialize)]
pub struct Reliability {
    /// Claims both labellers judged. A claim only one of them reached says nothing about
    /// either of them and is counted here rather than scored.
    pub overlap: u64,
    pub only_first: u64,
    pub only_second: u64,
    pub fields: Vec<FieldAgreement>,
    pub contested: Contested,
}

impl Reliability {
    /// The subject figure, which is the one worth quoting.
    #[must_use]
    pub fn subject(&self) -> Option<&FieldAgreement> {
        self.fields.iter().find(|field| field.field == "subject")
    }
}

/// Two labellings joined claim by claim, with how many claims each had that the other did not.
#[derive(Debug, Clone, Default)]
pub struct Paired {
    pub both: Vec<(ClaimLabel, ClaimLabel)>,
    pub only_first: u64,
    pub only_second: u64,
}

impl Paired {
    /// Adds another game's pairs to these, so several games can be read as one set.
    pub fn extend(&mut self, other: Self) {
        self.both.extend(other.both);
        self.only_first += other.only_first;
        self.only_second += other.only_second;
    }
}

/// The claims two labellings share, joined by review id and claim position.
///
/// # Errors
///
/// Fails if either file is missing or is not a list of labels.
pub fn paired(first: &std::path::Path, second: &std::path::Path) -> Result<Paired> {
    let read = |path: &std::path::Path| -> Result<Vec<ClaimLabel>> {
        let bytes = std::fs::read(path).map_err(|_| crate::Error::NoReferenceSet {
            path: path.to_path_buf(),
        })?;
        Ok(serde_json::from_slice(&bytes)?)
    };

    let (left, right) = (read(first)?, read(second)?);
    let mut theirs: HashMap<(String, u16), ClaimLabel> = right
        .into_iter()
        .map(|label| ((label.review_id.clone(), label.index), label))
        .collect();

    let mut found = Paired::default();
    for label in left {
        match theirs.remove(&(label.review_id.clone(), label.index)) {
            Some(other) => found.both.push((label, other)),
            None => found.only_first += 1,
        }
    }
    found.only_second = theirs.len() as u64;
    Ok(found)
}

/// Compares two labellings of the same claims.
///
/// # Errors
///
/// Fails if either file is missing or is not a list of labels.
pub fn compare(first: &std::path::Path, second: &std::path::Path) -> Result<Reliability> {
    Ok(over(&paired(first, second)?))
}

/// The same comparison over claims already paired, so several games can be pooled.
///
/// Pooled by adding the claims rather than averaging the games, for the same reason the
/// model's agreement is: a game with twenty-five claims read twice should not weigh as much as
/// one with eighty-seven.
#[must_use]
pub fn over(paired: &Paired) -> Reliability {
    let pairs: Vec<(&ClaimLabel, &ClaimLabel)> = paired.both.iter().map(|(a, b)| (a, b)).collect();
    let fields = [
        (
            "subject",
            (|label: &ClaimLabel| label.subject.clone()) as fn(&ClaimLabel) -> String,
        ),
        ("polarity", |label| label.polarity.clone()),
        ("ironic", |label| label.ironic.to_string()),
        ("ambiguous", |label| label.ambiguous.to_string()),
        ("split_wrong", |label| label.split_wrong.to_string()),
        ("confidence", |label| label.confidence.clone()),
    ];

    Reliability {
        overlap: pairs.len() as u64,
        only_first: paired.only_first,
        only_second: paired.only_second,
        fields: fields
            .iter()
            .map(|(name, of)| score(name, &pairs, *of))
            .collect(),
        contested: Contested::over(&pairs),
    }
}

/// Whether `ambiguous` means what the sheet says it means.
///
/// The sheet defines it as a property of the claim and the taxonomy: two subjects fit and the
/// rules do not settle which. If that is what labellers are recording, then a claim neither of
/// them called contested is one the rules do settle, and they should agree on its subject
/// almost always. If they do not, the field is recording something else, most likely how the
/// labeller felt, and it cannot be read the way the reports read it.
#[derive(Debug, Clone, Serialize)]
pub struct Contested {
    /// How often each labeller reached for the flag. Two labellers reading the same
    /// definition should reach for it at roughly the same rate.
    pub first_share: Option<f64>,
    pub second_share: Option<f64>,
    /// Claims neither labeller called contested, and how often they agreed on the subject.
    pub clear: u64,
    pub clear_agreed: u64,
    /// Claims either labeller called contested, and the same.
    pub flagged: u64,
    pub flagged_agreed: u64,
}

impl Contested {
    #[expect(clippy::cast_precision_loss, reason = "label counts are small")]
    fn over(pairs: &[(&ClaimLabel, &ClaimLabel)]) -> Self {
        let total = pairs.len() as f64;
        let mut found = Self {
            first_share: (!pairs.is_empty())
                .then(|| pairs.iter().filter(|(a, _)| a.ambiguous).count() as f64 / total),
            second_share: (!pairs.is_empty())
                .then(|| pairs.iter().filter(|(_, b)| b.ambiguous).count() as f64 / total),
            clear: 0,
            clear_agreed: 0,
            flagged: 0,
            flagged_agreed: 0,
        };
        for (a, b) in pairs {
            let same = u64::from(a.subject == b.subject);
            if a.ambiguous || b.ambiguous {
                found.flagged += 1;
                found.flagged_agreed += same;
            } else {
                found.clear += 1;
                found.clear_agreed += same;
            }
        }
        found
    }

    #[must_use]
    #[expect(clippy::cast_precision_loss, reason = "label counts are small")]
    pub fn clear_rate(&self) -> Option<f64> {
        (self.clear > 0).then(|| self.clear_agreed as f64 / self.clear as f64)
    }

    #[must_use]
    #[expect(clippy::cast_precision_loss, reason = "label counts are small")]
    pub fn flagged_rate(&self) -> Option<f64> {
        (self.flagged > 0).then(|| self.flagged_agreed as f64 / self.flagged as f64)
    }
}

fn score(
    field: &'static str,
    both: &[(&ClaimLabel, &ClaimLabel)],
    of: fn(&ClaimLabel) -> String,
) -> FieldAgreement {
    let mut agreed = 0_u64;
    let mut mine: HashMap<String, u64> = HashMap::new();
    let mut theirs: HashMap<String, u64> = HashMap::new();
    let mut splits: HashMap<(String, String), u64> = HashMap::new();

    for (left, right) in both {
        let (a, b) = (of(left), of(right));
        *mine.entry(a.clone()).or_default() += 1;
        *theirs.entry(b.clone()).or_default() += 1;
        if a == b {
            agreed += 1;
        } else {
            // Ordered, so "story read as gameplay" and the reverse are one disagreement
            // rather than two halves of one.
            let pair = if a < b { (a, b) } else { (b, a) };
            *splits.entry(pair).or_default() += 1;
        }
    }

    let compared = both.len() as u64;
    FieldAgreement {
        field,
        compared,
        agreed,
        kappa: kappa(agreed, compared, &mine, &theirs),
        commonest_split: splits
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|((a, b), count)| (a, b, count)),
    }
}

/// Cohen's kappa: agreement beyond what these two labellers' own habits would give by chance.
///
/// 1 is perfect, 0 is what two labellers drawing from their own distributions would manage
/// without reading anything, and below 0 is worse than that. `None` when the two agree on
/// everything and always would, where the statistic is undefined rather than perfect.
#[must_use]
#[expect(clippy::cast_precision_loss, reason = "label counts are small")]
fn kappa(
    agreed: u64,
    compared: u64,
    mine: &HashMap<String, u64>,
    theirs: &HashMap<String, u64>,
) -> Option<f64> {
    if compared == 0 {
        return None;
    }
    let total = compared as f64;
    let observed = agreed as f64 / total;
    let expected: f64 = mine
        .iter()
        .map(|(value, count)| {
            let other = theirs.get(value).copied().unwrap_or(0);
            (*count as f64 / total) * (other as f64 / total)
        })
        .sum();
    ((1.0 - expected).abs() > f64::EPSILON).then(|| (observed - expected) / (1.0 - expected))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn label(id: &str, index: u16, subject: &str, ambiguous: bool) -> ClaimLabel {
        ClaimLabel {
            review_id: id.to_owned(),
            index,
            app_id: 1,
            language: "english".to_owned(),
            subset: "random".to_owned(),
            start: 0,
            end: 1,
            subject: subject.to_owned(),
            polarity: "praise".to_owned(),
            ironic: false,
            confidence: "high".to_owned(),
            ambiguous,
            split_wrong: false,
        }
    }

    fn pairs(rows: &[(ClaimLabel, ClaimLabel)]) -> Vec<(&ClaimLabel, &ClaimLabel)> {
        rows.iter().map(|(a, b)| (a, b)).collect()
    }

    #[test]
    fn two_labellers_who_always_agree_score_one() {
        let rows: Vec<(ClaimLabel, ClaimLabel)> = ["bugs", "story", "bugs", "price"]
            .iter()
            .enumerate()
            .map(|(index, subject)| {
                (
                    label("1", u16::try_from(index).unwrap_or(0), subject, false),
                    label("1", u16::try_from(index).unwrap_or(0), subject, false),
                )
            })
            .collect();
        let scored = score("subject", &pairs(&rows), |label| label.subject.clone());
        assert_eq!(scored.rate(), Some(1.0));
        assert!(scored.kappa.unwrap() > 0.999);
        assert!(scored.commonest_split.is_none());
    }

    #[test]
    fn agreeing_only_as_often_as_chance_scores_zero_rather_than_well() {
        // Both call three in four claims `verdict`, independently. They agree 62.5% of the
        // time, and every point of it is the habit rather than the reading.
        let mut rows = Vec::new();
        for (index, (a, b)) in [
            ("verdict", "verdict"),
            ("verdict", "verdict"),
            ("verdict", "bugs"),
            ("bugs", "verdict"),
        ]
        .iter()
        .enumerate()
        {
            let index = u16::try_from(index).unwrap_or(0);
            rows.push((label("1", index, a, false), label("1", index, b, false)));
        }
        let scored = score("subject", &pairs(&rows), |label| label.subject.clone());
        assert_eq!(scored.agreed, 2);
        let kappa = scored.kappa.expect("two values in play");
        assert!(
            kappa.abs() < 0.35,
            "agreement this close to chance should not score well, got {kappa}"
        );
    }

    #[test]
    fn the_commonest_disagreement_is_named_once_rather_than_twice() {
        let rows = vec![
            (
                label("1", 0, "story", false),
                label("1", 0, "gameplay", false),
            ),
            (
                label("1", 1, "gameplay", false),
                label("1", 1, "story", false),
            ),
            (label("1", 2, "bugs", false), label("1", 2, "price", false)),
        ];
        let scored = score("subject", &pairs(&rows), |label| label.subject.clone());
        let (a, b, count) = scored.commonest_split.expect("they disagreed");
        assert_eq!((a.as_str(), b.as_str(), count), ("gameplay", "story", 2));
    }

    #[test]
    fn the_contested_flag_is_checked_against_what_it_claims_to_mean() {
        // Two claims neither labeller flagged, agreed on; two that one of them flagged, one
        // agreed and one not. If the flag means what the sheet says, the unflagged ones are
        // the ones the rules settle, and 100% on them is what "settled" looks like.
        let rows = vec![
            (label("1", 0, "bugs", false), label("1", 0, "bugs", false)),
            (label("1", 1, "story", false), label("1", 1, "story", false)),
            (label("1", 2, "genre", true), label("1", 2, "genre", false)),
            (
                label("1", 3, "verdict", false),
                label("1", 3, "genre", true),
            ),
        ];
        let found = over(&Paired {
            both: rows,
            only_first: 0,
            only_second: 0,
        });
        let contested = &found.contested;
        assert_eq!((contested.clear, contested.clear_agreed), (2, 2));
        assert_eq!((contested.flagged, contested.flagged_agreed), (2, 1));
        assert!((contested.first_share.unwrap() - 0.25).abs() < 1e-9);
        assert!((contested.second_share.unwrap() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn pooling_two_games_adds_their_claims() {
        let mut every = Paired {
            both: vec![(label("1", 0, "bugs", false), label("1", 0, "bugs", false))],
            only_first: 1,
            only_second: 0,
        };
        every.extend(Paired {
            both: vec![
                (label("2", 0, "story", false), label("2", 0, "story", false)),
                (label("2", 1, "story", false), label("2", 1, "price", false)),
            ],
            only_first: 0,
            only_second: 2,
        });
        let found = over(&every);
        assert_eq!(found.overlap, 3);
        assert_eq!((found.only_first, found.only_second), (1, 2));
        let subject = found.subject().expect("subject is always scored");
        assert_eq!((subject.compared, subject.agreed), (3, 2));
    }

    #[test]
    fn a_field_nobody_varies_has_no_kappa_rather_than_a_perfect_one() {
        let rows: Vec<(ClaimLabel, ClaimLabel)> = (0..4)
            .map(|index| {
                (
                    label("1", index, "verdict", false),
                    label("1", index, "verdict", false),
                )
            })
            .collect();
        let scored = score("ambiguous", &pairs(&rows), |label| {
            label.ambiguous.to_string()
        });
        assert_eq!(scored.rate(), Some(1.0));
        assert_eq!(
            scored.kappa, None,
            "a field with one value in it cannot have agreement beyond chance"
        );
    }
}
