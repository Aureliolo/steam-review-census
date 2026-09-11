//! Renders an adjudication page from made-up claims, for the browser check.
//!
//! The real page is drawn from reference sets, which carry review text and are not in the
//! repository, so CI cannot make one. Everything the check drives is in the page's behaviour
//! rather than in the claims, so invented ones exercise it exactly as well: a blind question,
//! a split one with both labellers' answers, a multibyte one, and one whose claim appears
//! twice in its review so that marking the wrong copy would show.
//!
//! cargo run -p steamgauge-core --example sample-gold -- gold.html

use steamgauge_core::gold::{Answered, GoldDraw, Question, render};

fn question(id: &str, claim: &str, review: &str, shown: Option<Vec<Answered>>) -> Question {
    let at = review.find(claim).unwrap_or(0);
    Question {
        app_id: 1_091_500,
        review_id: id.to_owned(),
        index: 0,
        claim: claim.to_owned(),
        before: review[..at].to_owned(),
        after: review[at + claim.len()..].to_owned(),
        language: "english".to_owned(),
        shown,
    }
}

fn said(subject: &str, polarity: &str, confidence: &str, ambiguous: bool) -> Answered {
    Answered {
        subject: subject.to_owned(),
        polarity: polarity.to_owned(),
        confidence: confidence.to_owned(),
        ambiguous,
    }
}

fn main() {
    let to = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "gold.html".into());

    let questions = vec![
        question(
            "1",
            "The combat finally feels weighty.",
            "I put four hundred hours into this. The combat finally feels weighty. \
             Multiplayer is dead though and the servers are empty most nights.",
            None,
        ),
        // The same words twice in one review: marking the first copy when the label names the
        // second would put the reader's eye on the wrong sentence and nothing would look wrong.
        question(
            "2",
            "it runs badly",
            "On my old machine it runs badly. On the new one it runs badly too, which I did \
             not expect after the patch.",
            None,
        ),
        question(
            "3",
            "\u{6559}\u{5b66}\u{7eaf}\u{9760}\u{81ea}\u{5df1}\u{9886}\u{609f}",
            "\u{6559}\u{5b66}\u{7eaf}\u{9760}\u{81ea}\u{5df1}\u{9886}\u{609f}\u{3002}\
             \u{6218}\u{6597}\u{624b}\u{611f}\u{6781}\u{597d}\u{3002}",
            None,
        ),
        question(
            "4",
            "Worth every penny on sale.",
            "Bought it at 70% off. Worth every penny on sale. Would not pay full price.",
            Some(vec![
                said("price", "praise", "high", false),
                said("verdict", "praise", "medium", true),
            ]),
        ),
        question(
            "5",
            "same here",
            "The last reviewer said the tutorial explains nothing. same here",
            None,
        ),
    ];

    let found = GoldDraw {
        blind: 4,
        split: 1,
        games: 8,
        agreed: 244,
    };

    std::fs::write(&to, render(&questions, found)).expect("the page could not be written");
    println!("{to}");
}
