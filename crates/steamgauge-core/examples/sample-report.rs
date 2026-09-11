//! Renders a report from invented counts, so the page can be opened and driven without a
//! corpus, a model or a network.
//!
//! The scripting on that page depends on the shape of the markup and not on what the numbers
//! mean, so a fixture is enough to exercise all of it. Several games, because the cross-game
//! table only exists when there is something to compare, and one measured game and one
//! unmeasured, because they render differently.
//!
//! ```text
//! cargo run -p steamgauge-core --example sample-report -- page.html
//! cargo run -p steamgauge-core --example sample-report -- page.html --one-game
//! cargo run -p steamgauge-core --example sample-report -- page.html --games 36
//! ```

use std::path::PathBuf;

use steamgauge_core::{
    capture::CapturedReview,
    induced::Induced,
    measure::{ClaimAgreement, SubjectAgreement},
    read::{Depth, Month, ReadReport, SubjectCount},
    report::{AppReport, CrawlFacts, Example, InducedEvidence, Measurement, Report},
    said::{SaidAbout, Term},
    taxonomy::{CORE_SPINE, CORE_SPINE_VERSION},
};

/// Four, because a table of two fits on a phone and a real one does not, and what happens to
/// a table wider than the screen it is read on is the whole question.
const DEFAULT_GAMES: usize = 4;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let path: PathBuf = arguments
        .iter()
        .find(|argument| !argument.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "sample-report.html".to_owned())
        .into();
    // A report of one game has no cross-game table and no contents, and is a different page
    // to drive. Both shapes ship, so both are worth being able to render.
    let alone = arguments.iter().any(|argument| argument == "--one-game");
    let wanted = arguments
        .iter()
        .position(|argument| argument == "--games")
        .and_then(|slot| arguments.get(slot + 1))
        .map_or(DEFAULT_GAMES, |count| count.parse().expect("--games count"));

    let mut apps = vec![game(7, "A Measured Game <& Friends>", true)];
    if !alone {
        apps.push(game(11, "An Unmeasured Game", false));
        apps.push(game(13, "A Third Game With A Rather Long Name", true));
        apps.push(game(17, "A Fourth Game", true));
        // Beyond the four distinctive ones, filler: a census of a whole genre runs to dozens
        // of games, and every part of the page that grows with the number of them has to be
        // read at that size rather than at the size that fits in a screenshot.
        for extra in apps.len()..wanted {
            let id = 100 + u32::try_from(extra).expect("more games than a page could hold");
            apps.push(game(id, &format!("Filler Game {extra}"), true));
        }
    }
    let report = Report {
        generated_unix: 1_760_000_000,
        apps,
    };
    std::fs::write(&path, steamgauge_core::html::render(&report))?;
    println!("{}", path.display());
    Ok(())
}

fn game(app_id: u32, name: &str, measured: bool) -> AppReport {
    // Corpora differ in size by more than an order of magnitude in any real set, which is the
    // whole reason the page compares the games and not only the subjects, and the reason its
    // table can be reordered on any of its columns.
    let reviews = 2_000 + u64::from(app_id % 11) * 900;
    let positive = reviews * 7 / 10;
    let subjects = subjects(app_id, reviews);

    AppReport {
        crawl: CrawlFacts {
            app_id,
            name: name.to_owned(),
            review_score_desc: "Mostly Positive".to_owned(),
            rows_unique: reviews,
            valve_total_reviews: reviews + 20,
            valve_total_positive: positive,
            valve_total_negative: reviews - positive,
            // Never all of it: the one number a reader may take as a claim of completeness.
            coverage: 0.998,
            snapshot_unix: 1_759_000_000,
            swept_unix: Some(1_759_500_000),
            sweeps: 1,
            rows_swept: 40,
            shards: 4,
        },
        reading: ReadReport {
            app_id,
            reviews,
            corpus_reviews: reviews + 200,
            language: None,
            depth: Depth::Deep,
            splitter: steamgauge_core::claims::SPLITTER_VERSION.to_owned(),
            claims: reviews * 3,
            // Over half, as every real reading currently is, so the page renders the caveat
            // it puts above the table rather than the footnote it puts below.
            unclassified_claims: reviews * 3 * 3 / 5,
            silent_reviews: reviews / 50,
            positive,
            top_helpful: 50,
            model: "Alibaba-NLP/gte-multilingual-base".to_owned(),
            trained_on: "0123456789abcdef".to_owned(),
            usual_declined: Some(0.53),
            frozen: Some(steamgauge_core::reader::Frozen {
                games: 8,
                claims: 3769,
                coverage: 0.577,
                accuracy: 0.749,
                macro_f1: 0.504,
            }),
            context: true,
            spine_version: CORE_SPINE_VERSION.to_owned(),
            threshold: 0.77,
            device: "directml".to_owned(),
            captured_unix: 1_759_000_000,
            said: said_about(&subjects),
            subjects: subjects.clone(),
            languages: vec![
                ("english".to_owned(), reviews * 6 / 10),
                ("schinese".to_owned(), reviews * 3 / 10),
                ("russian".to_owned(), reviews / 10),
            ],
            months: months(app_id),
            elapsed: std::time::Duration::from_secs(90),
        },
        examples: examples(app_id, &subjects),
        top: vec![example(
            app_id,
            9,
            "The most helpful review in the corpus, which is what a reader skimming the store \
             page sees before anything this tool counted.",
            true,
        )],
        agreement: if measured {
            Measurement::Measured(Box::new(agreement(app_id)))
        } else {
            Measurement::Unlabelled
        },
        induced: induced(app_id),
    }
}

/// Every subject counted over one game.
fn subjects(app_id: u32, reviews: u64) -> Vec<SubjectCount> {
    CORE_SPINE
        .iter()
        .enumerate()
        .map(|(slot, category)| {
            // Descending counts, so sorting has something to reorder and the widest bar is
            // the first row rather than an accident. The last subject is raised by nobody,
            // which every real report has and which is the only thing that puts a dash in
            // the rightmost column of the table.
            let quiet = slot + 1 == CORE_SPINE.len();
            // Which game leads a subject is a claim the page only draws where one clears
            // every other in the figures it prints, so a fixture where every game reports
            // the same rate exercises none of it. The spread is distinct per game, except on
            // one subject where every game lands together and the page has to leave the row
            // unclaimed rather than outline a rounding difference.
            let tied = slot == 5;
            let spread = if tied {
                0
            } else {
                (u64::from(app_id) * 7_919 + slot as u64 * 31) % 300
            };
            let mentions = if quiet {
                0
            } else {
                (900 - (slot as u64 * 30) + spread).min(reviews)
            };
            SubjectCount {
                id: category.id.to_owned(),
                label: category.label.to_owned(),
                mention_reviews: mentions,
                primary_reviews: mentions / 2,
                claims: mentions * 2,
                praised: mentions / 3,
                criticised: mentions / 3,
                mixed: mentions / 6,
                top_mention_reviews: mentions / 40,
                positive_mentions: mentions * 2 / 3,
            }
        })
        .collect()
}

/// Three quoted claims for every subject anybody raised. A subject nobody raised gets none,
/// which is how a real report renders a row with a dash in it.
fn examples(app_id: u32, subjects: &[SubjectCount]) -> Vec<(String, Vec<Example>)> {
    subjects
        .iter()
        .filter(|subject| subject.mention_reviews > 0)
        .map(|subject| {
            let quoted = (0..3)
                .map(|which| {
                    example(
                        app_id,
                        which,
                        &format!(
                            "Something a reviewer said about {} that runs on long enough to \
                             wrap on a narrow screen, which is where a quotation stops being \
                             readable.",
                            subject.label.to_lowercase()
                        ),
                        which == 0,
                    )
                })
                .collect();
            (subject.id.clone(), quoted)
        })
        .collect()
}

/// The terms that separate praise from complaint on the first three subjects, which is what
/// the page shows under each side and what a reader clicks to narrow the evidence.
fn said_about(subjects: &[SubjectCount]) -> Vec<SaidAbout> {
    let term = |text: &str, reviews| Term {
        text: text.to_owned(),
        reviews,
    };
    subjects
        .iter()
        .take(3)
        .map(|subject| SaidAbout {
            subject: subject.id.clone(),
            praising: subject.praised,
            complaining: subject.criticised,
            praised: vec![term("best in years", 41), term("worth it", 22)],
            criticised: vec![
                term("save corruption", 37),
                term("crashes on launch", 29),
                term("unplayable", 11),
            ],
        })
        .collect()
}

/// Two years of months, so the timeline has a line to draw, a quiet month that carries no
/// rate, and a gap where nobody wrote anything at all.
fn months(app_id: u32) -> Vec<Month> {
    (0..24)
        .filter(|which| *which != 7)
        .map(|which| {
            let reviews = if which == 11 {
                5
            } else {
                100 + u64::from(app_id % 7) * 20 + which * 13
            };
            Month {
                label: format!("2024-{:02}", which % 12 + 1),
                reviews,
                positive: reviews * 2 / 3,
                subjects: CORE_SPINE
                    .iter()
                    .enumerate()
                    .map(|(slot, _)| reviews / (slot as u64 + 2))
                    .collect(),
            }
        })
        .collect()
}

/// A quoted claim and the review it came from.
///
/// The review is longer than the claim and longer than the page clips at, because that is
/// what a real one is: the claim is the one sentence in thirty that earned the count, and the
/// rest of it is behind a control a reader has to be able to open.
fn example(app_id: u32, which: u16, text: &str, from_the_top: bool) -> Example {
    let whole = format!(
        "{text} {}",
        "There is a great deal more in this review than the sentence that was counted, which \
         is the ordinary case and the reason the rest of it is offered rather than shown: a \
         reviewer with something to say says it at length, in paragraphs about other things. "
            .repeat(3)
    );
    Example {
        review: CapturedReview {
            id: format!("{app_id}{which:03}"),
            text: whole,
            language: "english".to_owned(),
            author_steamid: "76561190000000000".to_owned(),
            voted_up: which.is_multiple_of(2),
            votes_up: u32::from(which) * 7,
            votes_funny: u32::from(which),
            playtime_at_review_minutes: 600 + u32::from(which) * 90,
            created: 1_740_000_000 + i64::from(which) * 86_400,
        },
        claim: text.to_owned(),
        index: which,
        polarity: if which.is_multiple_of(2) {
            "praise".to_owned()
        } else {
            "complaint".to_owned()
        },
        confidence: 0.79 + f32::from(which) / 100.0,
        also: vec!["bugs".to_owned(), "performance".to_owned()],
        from_the_top,
    }
}

/// A game measured against a labelled set, including labels this build no longer cuts as
/// they were labelled, which the page has to account for rather than quietly drop.
fn agreement(app_id: u32) -> ClaimAgreement {
    let answered = 200 + u64::from(app_id % 5) * 30;
    let agreed = answered * 4 / 5;
    ClaimAgreement {
        app_id,
        matched: answered * 2,
        unjoined: 19,
        answered,
        agreed,
        declined: answered,
        polarity_answered: answered,
        polarity_agreed: answered * 9 / 10,
        clear_answered: answered * 3 / 4,
        clear_agreed: answered * 7 / 10,
        contested_answered: answered / 4,
        contested_agreed: answered / 8,
        subjects: CORE_SPINE
            .iter()
            .enumerate()
            .map(|(slot, category)| {
                let labelled = 60 - (slot as u64).min(55);
                SubjectAgreement {
                    id: category.id,
                    label: category.label,
                    labelled,
                    read: labelled + 3,
                    agreed: labelled * 2 / 3,
                    seen: labelled * 2,
                    // A subject read as one particular other subject is a boundary the
                    // taxonomy has not settled, and the page says so where it happens.
                    mistaken_for: (slot == 2).then_some(("gameplay", 7)),
                }
            })
            .collect(),
    }
}

/// Subjects this game's own players raise that the spine has no row for, with the reviews
/// that earned each one. Only some games have them, and a page with none renders differently.
fn induced(app_id: u32) -> Vec<InducedEvidence> {
    if app_id.is_multiple_of(2) {
        return Vec::new();
    }
    vec![InducedEvidence {
        subject: Induced {
            id: "mud-physics".to_owned(),
            label: "Mud and terrain deformation".to_owned(),
            description: "How the ground deforms under a wheel, and whether it behaves the \
                          way a driver expects; not the truck roster and not the scenery."
                .to_owned(),
            refines: Some("gameplay".to_owned()),
            evidence: (0..3).map(|which| format!("{app_id}{which:03}")).collect(),
        },
        reviews: (0..3)
            .map(|which| {
                example(
                    app_id,
                    which,
                    "A review that is mostly about the mud, at the length reviews about one \
                     mechanic tend to run to.",
                    false,
                )
                .review
            })
            .collect(),
    }]
}
