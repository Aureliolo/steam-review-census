//! The claims a person adjudicates, and the page they do it on.
//!
//! Everything measured so far is a model agreeing with a model. The README says so and no
//! citation can rest on it. What turns a silver set into a gold one is a person reading the
//! claims themselves, and the only reason that has not happened is that there was nothing to
//! read them on.
//!
//! Two draws, because they answer different questions:
//!
//! - **Blind.** A random sample of frozen claims with no answer shown, which is the only kind
//!   of reading that produces an accuracy figure rather than a ratification.
//! - **Split.** The claims two labellers answered differently, with both answers shown, which
//!   is what settles a boundary rather than measuring one.
//!
//! The page is one self-contained file that fetches nothing and sends nothing. It holds review
//! text, so it is written outside the repository and never committed, exactly like a report.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    Result,
    claimset::{ClaimLabel, DrawnReview},
    measure::{Role, SPLIT_SEED, role},
    taxonomy::CORE_SPINE,
};

/// One claim put in front of a person, with everything needed to judge it and nothing else.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Question {
    pub app_id: u32,
    pub review_id: String,
    pub index: u16,
    /// The claim itself.
    pub claim: String,
    /// The review it came from, as the labeller was shown it.
    pub review: String,
    /// Where the claim starts in `review`, so the page can mark it.
    pub at: usize,
    pub language: String,
    /// What the labellers said, shown only for a split: `None` on a blind question, because an
    /// answer on the page is an answer in the reader's head.
    pub shown: Option<Vec<Answered>>,
}

/// One labeller's answer to a question, as it is shown on a split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Answered {
    pub subject: String,
    pub polarity: String,
    pub confidence: String,
    pub ambiguous: bool,
}

/// What a draw turned out to be.
#[derive(Debug, Clone, Copy, Default)]
pub struct GoldDraw {
    pub blind: usize,
    pub split: usize,
    pub games: usize,
    /// Claims the two labellers answered the same way, which are not worth a person's time:
    /// counted so the page can say what share of the set was never in question.
    pub agreed: usize,
}

/// Which games' disagreements to put in front of a person.
///
/// The blind sample is always frozen, because that is what makes it a measurement. The splits
/// measure nothing: they settle a boundary, and a boundary settled on one game is settled for
/// every game, so they may be drawn from anywhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Splits {
    None,
    Frozen,
    Everywhere,
}

/// Draws the claims a person should read.
///
/// Frozen because a gold figure has to be about games that chose nothing: a person adjudicating
/// the games the model trained on would produce a number that flatters every part of the chain
/// at once.
///
/// # Errors
///
/// Fails if a reference set cannot be read.
pub fn draw(
    reference: &Path,
    blind_wanted: usize,
    splits: Splits,
    seed: u64,
) -> Result<(Vec<Question>, GoldDraw)> {
    let mut blind: Vec<([u8; 32], Question)> = Vec::new();
    let mut split: Vec<Question> = Vec::new();
    let mut found = GoldDraw::default();

    for entry in std::fs::read_dir(reference)? {
        let dir = entry?.path();
        let Some(app_id) = dir
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| name.parse::<u32>().ok())
        else {
            continue;
        };
        let frozen = role(app_id, SPLIT_SEED) == Role::Frozen;
        if !frozen && splits != Splits::Everywhere {
            continue;
        }
        let Ok(sample) = std::fs::read(dir.join("sample.json")) else {
            continue;
        };
        let Ok(first) = std::fs::read(dir.join("labels.json")) else {
            continue;
        };
        let drawn: Vec<DrawnReview> = serde_json::from_slice(&sample)?;
        let first: Vec<ClaimLabel> = serde_json::from_slice(&first)?;
        let second: Vec<ClaimLabel> = std::fs::read(dir.join("second").join("labels.json"))
            .ok()
            .and_then(|raw| serde_json::from_slice(&raw).ok())
            .unwrap_or_default();
        found.games += usize::from(frozen);

        let around: std::collections::HashMap<&str, Rejoined<'_>> = drawn
            .iter()
            .map(|review| (review.id.as_str(), Rejoined::of(review)))
            .collect();

        let twice: std::collections::HashMap<(&str, u16), &ClaimLabel> = second
            .iter()
            .map(|label| ((label.review_id.as_str(), label.index), label))
            .collect();

        for label in &first {
            let Some(rejoined) = around.get(label.review_id.as_str()) else {
                continue;
            };
            let Some((at, text)) = rejoined.find(label.index) else {
                continue;
            };
            let question = Question {
                app_id,
                review_id: label.review_id.clone(),
                index: label.index,
                claim: text.to_owned(),
                review: rejoined.text.clone(),
                at,
                language: label.language.clone(),
                shown: None,
            };

            if let Some(other) = twice.get(&(label.review_id.as_str(), label.index)) {
                if other.subject == label.subject {
                    found.agreed += 1;
                } else if splits != Splits::None {
                    split.push(Question {
                        shown: Some(vec![answered(label), answered(other)]),
                        ..question
                    });
                }
                continue;
            }

            // A blind question is a measurement, and a measurement on a game the model trained
            // on measures nothing.
            if !frozen {
                continue;
            }
            blind.push((
                crate::bounded::rank(seed, "gold-blind", &format!("{app_id}#{}", label.review_id)),
                question,
            ));
        }
    }

    blind.sort_by_key(|(key, _)| *key);
    let mut questions: Vec<Question> = blind
        .into_iter()
        .take(blind_wanted)
        .map(|(_, question)| question)
        .collect();
    found.blind = questions.len();
    found.split = split.len();
    questions.extend(split);
    Ok((questions, found))
}

/// A review's claims joined back together, and where each one lands in the result.
///
/// The same view the labeller was shown and the same one the reader is trained against, so a
/// person adjudicating sees neither more nor less than either of them did.
struct Rejoined<'a> {
    text: String,
    /// Claim index, byte offset into `text`, and the claim itself.
    claims: Vec<(u16, usize, &'a str)>,
}

impl<'a> Rejoined<'a> {
    fn of(review: &'a DrawnReview) -> Self {
        let mut text = String::new();
        let mut claims = Vec::with_capacity(review.claims.len());
        for claim in &review.claims {
            if !text.is_empty() {
                text.push(' ');
            }
            claims.push((claim.index, text.len(), claim.text.as_str()));
            text.push_str(&claim.text);
        }
        Self { text, claims }
    }

    fn find(&self, index: u16) -> Option<(usize, &'a str)> {
        self.claims
            .iter()
            .find(|(at, _, _)| *at == index)
            .map(|&(_, at, text)| (at, text))
    }
}

fn answered(label: &ClaimLabel) -> Answered {
    Answered {
        subject: label.subject.clone(),
        polarity: label.polarity.clone(),
        confidence: label.confidence.clone(),
        ambiguous: label.ambiguous,
    }
}

const STYLE: &str = include_str!("gold.css");
const SCRIPT: &str = include_str!("gold.js");

/// Renders the page a person adjudicates on: one file, fetching nothing, sending nothing.
#[must_use]
pub fn render(questions: &[Question], found: GoldDraw) -> String {
    let categories: Vec<serde_json::Value> = CORE_SPINE
        .iter()
        .map(|category| {
            serde_json::json!({
                "id": category.id,
                "label": category.label,
                "description": category.description,
                "boundary": category.boundary,
            })
        })
        .collect();

    let data = serde_json::json!({
        "questions": questions,
        "categories": categories,
        "taxonomy": crate::CORE_SPINE_VERSION,
        "splitter": crate::claims::SPLITTER_VERSION,
        "blind": found.blind,
        "split": found.split,
        "games": found.games,
        "agreed": found.agreed,
    });

    format!(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <title>Adjudicate | SteamGauge</title>\n<style>{STYLE}</style>\n</head>\n<body>\n\
         <div id=\"app\"></div>\n\
         <script id=\"data\" type=\"application/json\">{}</script>\n\
         <script>{SCRIPT}</script>\n</body>\n</html>\n",
        serde_json::to_string(&data).unwrap_or_else(|_| "{}".to_owned())
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_page_carries_every_question_and_the_whole_sheet() {
        let questions = vec![Question {
            app_id: 1,
            review_id: "42".to_owned(),
            index: 0,
            claim: "Runs badly".to_owned(),
            review: "Runs badly and I love it".to_owned(),
            at: 0,
            language: "english".to_owned(),
            shown: None,
        }];
        let page = render(&questions, GoldDraw::default());
        assert!(page.contains("Runs badly"));
        for category in CORE_SPINE {
            assert!(page.contains(category.id), "{} is missing", category.id);
        }
        assert!(
            page.contains("<script id=\"data\""),
            "the questions travel as data rather than as markup"
        );
    }

    #[test]
    fn a_blind_question_carries_no_answer() {
        let questions = vec![Question {
            app_id: 1,
            review_id: "42".to_owned(),
            index: 0,
            claim: "Runs badly".to_owned(),
            review: "Runs badly".to_owned(),
            at: 0,
            language: "english".to_owned(),
            shown: None,
        }];
        let page = render(&questions, GoldDraw::default());
        assert!(
            page.contains("\"shown\":null"),
            "an answer on the page is an answer in the reader's head"
        );
    }
}
