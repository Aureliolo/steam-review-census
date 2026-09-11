//! The game in a paragraph, assembled from what was counted.
//!
//! A reader who will not read a table still wants to know what a game's players talk about
//! and which way they lean. Every clause here is a number from the reading with words around
//! it: which subjects are raised most and by what share of reviews, whether each is praised
//! or criticised more, and the words that stand out on each side. Nothing is inferred; a
//! subject nobody raises is not mentioned, and a side nothing stands out on says nothing.

use crate::{
    html::{percent, thousands},
    read::{ReadReport, SubjectCount},
    said::Term,
};

/// Subjects named in the paragraph. Three is what a sentence can carry.
const SUBJECTS_NAMED: usize = 3;

/// Terms quoted per side of a subject. Fewer than are shown under the row, because a
/// paragraph that lists eight words per side is a table with commas.
const TERMS_QUOTED: usize = 3;

/// A side has to outweigh the other by this much before the paragraph calls a subject
/// "mostly" one thing. Below it the honest word is "divided".
const MOSTLY: u64 = 2;

/// Subjects that are not what a game is about: a verdict names no aspect and off-topic names
/// no game, so neither belongs in a sentence about what players discuss.
const NOT_A_SUBJECT: [&str; 2] = ["verdict", "offtopic"];

/// What the reading found, in plain sentences. Empty when nothing was counted.
#[must_use]
pub fn in_short(reading: &ReadReport) -> String {
    if reading.reviews == 0 {
        return String::new();
    }
    let mut out = format!("Of {} reviews", thousands(reading.reviews));
    if let Some(positive) = share(reading.positive, reading.reviews) {
        let _ = std::fmt::Write::write_fmt(
            &mut out,
            format_args!(", {} recommend the game", percent(positive)),
        );
    }
    out.push('.');

    let mut raised: Vec<&SubjectCount> = reading
        .subjects
        .iter()
        .filter(|subject| !NOT_A_SUBJECT.contains(&subject.id.as_str()))
        .filter(|subject| subject.mention_reviews > 0)
        .collect();
    raised.sort_by(|left, right| {
        right
            .mention_reviews
            .cmp(&left.mention_reviews)
            .then_with(|| left.id.cmp(&right.id))
    });
    raised.truncate(SUBJECTS_NAMED);
    if raised.is_empty() {
        return out;
    }

    let named: Vec<String> = raised
        .iter()
        .filter_map(|subject| {
            share(subject.mention_reviews, reading.reviews).map(|rate| {
                format!(
                    "{} ({} of reviews)",
                    subject.label.to_lowercase(),
                    percent(rate)
                )
            })
        })
        .collect();
    let _ = std::fmt::Write::write_fmt(
        &mut out,
        format_args!(
            " The {} raised most {} {}.",
            if named.len() == 1 {
                "subject"
            } else {
                "subjects"
            },
            if named.len() == 1 { "is" } else { "are" },
            listed(&named)
        ),
    );

    for subject in raised {
        if let Some(sentence) = leaning(subject, reading) {
            out.push(' ');
            out.push_str(&sentence);
        }
    }
    out
}

/// Which way a subject's reviewers lean, and the words each side uses, as one sentence.
fn leaning(subject: &SubjectCount, reading: &ReadReport) -> Option<String> {
    let praised = subject.praised;
    let criticised = subject.criticised;
    if praised + criticised == 0 {
        return None;
    }
    let lean = if praised >= criticised.saturating_mul(MOSTLY) && praised > 0 {
        "is mostly praised"
    } else if criticised >= praised.saturating_mul(MOSTLY) && criticised > 0 {
        "is mostly criticised"
    } else {
        "divides opinion"
    };
    let mut sentence = format!("{} {lean}", capitalised(&subject.label));

    let said = reading.said.iter().find(|said| said.subject == subject.id);
    let praise = said.map_or(&[][..], |said| said.praised.as_slice());
    let complaint = said.map_or(&[][..], |said| said.criticised.as_slice());
    match (quoted(praise), quoted(complaint)) {
        (Some(praise), Some(complaint)) => {
            let _ = std::fmt::Write::write_fmt(
                &mut sentence,
                format_args!(": the praise says {praise}, the complaints {complaint}"),
            );
        }
        (Some(praise), None) => {
            let _ = std::fmt::Write::write_fmt(
                &mut sentence,
                format_args!("; the praise says {praise}"),
            );
        }
        (None, Some(complaint)) => {
            let _ = std::fmt::Write::write_fmt(
                &mut sentence,
                format_args!("; the complaints say {complaint}"),
            );
        }
        (None, None) => {}
    }
    sentence.push('.');
    Some(sentence)
}

/// The first few terms of a side, quoted, or nothing when the side has none.
fn quoted(terms: &[Term]) -> Option<String> {
    if terms.is_empty() {
        return None;
    }
    let words: Vec<String> = terms
        .iter()
        .take(TERMS_QUOTED)
        .map(|term| format!("\u{201c}{}\u{201d}", term.text))
        .collect();
    Some(listed(&words))
}

/// "a", "a and b", "a, b and c".
fn listed(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [only] => only.clone(),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    }
}

fn capitalised(label: &str) -> String {
    let mut chars = label.chars();
    chars.next().map_or_else(String::new, |first| {
        first.to_uppercase().chain(chars).collect()
    })
}

#[expect(
    clippy::cast_precision_loss,
    reason = "review counts are far below 2^53"
)]
fn share(part: u64, whole: u64) -> Option<f64> {
    (whole > 0).then(|| part as f64 / whole as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::said::{SaidAbout, Term};

    fn subject(
        id: &str,
        label: &str,
        mentions: u64,
        praised: u64,
        criticised: u64,
    ) -> SubjectCount {
        SubjectCount {
            id: id.to_owned(),
            label: label.to_owned(),
            mention_reviews: mentions,
            primary_reviews: mentions,
            claims: mentions,
            praised,
            criticised,
            mixed: 0,
            top_mention_reviews: 0,
            positive_mentions: 0,
        }
    }

    fn reading(subjects: Vec<SubjectCount>, said: Vec<SaidAbout>) -> ReadReport {
        ReadReport {
            app_id: 1,
            reviews: 1_000,
            corpus_reviews: 1_000,
            language: None,
            depth: crate::read::Depth::Deep,
            splitter: String::new(),
            claims: 3_000,
            unclassified_claims: 0,
            silent_reviews: 0,
            positive: 700,
            top_helpful: 50,
            model: String::new(),
            trained_on: String::new(),
            usual_declined: None,
            frozen: None,
            context: false,
            spine_version: String::new(),
            threshold: 0.5,
            device: String::new(),
            captured_unix: 0,
            subjects,
            said,
            languages: Vec::new(),
            months: Vec::new(),
            elapsed: std::time::Duration::ZERO,
        }
    }

    fn term(text: &str, reviews: u64) -> Term {
        Term {
            text: text.to_owned(),
            reviews,
        }
    }

    #[test]
    fn the_paragraph_names_the_subjects_raised_most_and_which_way_they_lean() {
        let found = reading(
            vec![
                subject("verdict", "Overall verdict only", 900, 0, 0),
                subject("gameplay", "Gameplay and mechanics", 410, 300, 50),
                subject("performance", "Performance", 180, 20, 120),
                subject("story", "Story and writing", 250, 100, 90),
                subject("audio", "Audio and music", 30, 30, 0),
            ],
            vec![
                SaidAbout {
                    subject: "gameplay".to_owned(),
                    praising: 300,
                    complaining: 50,
                    praised: vec![
                        term("combat", 90),
                        term("exploration", 60),
                        term("bosses", 40),
                        term("loot", 20),
                    ],
                    criticised: vec![term("grind", 30)],
                },
                SaidAbout {
                    subject: "performance".to_owned(),
                    praising: 20,
                    complaining: 120,
                    praised: Vec::new(),
                    criticised: vec![term("stutter", 80), term("crashes", 40)],
                },
            ],
        );
        let text = in_short(&found);
        assert_eq!(
            text,
            "Of 1,000 reviews, 70.0% recommend the game. The subjects raised most are gameplay \
             and mechanics (41.0% of reviews), story and writing (25.0% of reviews) and \
             performance (18.0% of reviews). Gameplay and mechanics is mostly praised: the \
             praise says \u{201c}combat\u{201d}, \u{201c}exploration\u{201d} and \
             \u{201c}bosses\u{201d}, the complaints \u{201c}grind\u{201d}. Story and writing \
             divides opinion. Performance is mostly criticised; the complaints say \
             \u{201c}stutter\u{201d} and \u{201c}crashes\u{201d}."
        );
    }

    #[test]
    fn a_verdict_is_not_a_subject_a_game_is_about() {
        let found = reading(
            vec![subject("verdict", "Overall verdict only", 900, 0, 0)],
            Vec::new(),
        );
        assert_eq!(
            in_short(&found),
            "Of 1,000 reviews, 70.0% recommend the game."
        );
    }

    #[test]
    fn nothing_counted_says_nothing() {
        let mut found = reading(Vec::new(), Vec::new());
        found.reviews = 0;
        assert_eq!(in_short(&found), "");
    }
}
