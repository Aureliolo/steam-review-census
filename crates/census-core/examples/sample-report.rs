//! Renders a report from invented counts, so the page can be opened and driven without a
//! corpus, a model or a network.
//!
//! The scripting on that page depends on the shape of the markup and not on what the numbers
//! mean, so a fixture is enough to exercise all of it. Two games, because the cross-game
//! matrix only exists when there is something to compare, and one measured game and one
//! unmeasured, because they render differently.
//!
//! ```text
//! cargo run -p census-core --example sample-report -- page.html
//! ```

use std::path::PathBuf;

use census_core::{
    capture::CapturedReview,
    evaluate::{AgreementReport, CategoryAgreement, Slice},
    report::{AppReport, CategoryCount, Classification, CrawlFacts, Example, Month, Report},
    taxonomy::CORE_SPINE,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "sample-report.html".to_owned())
        .into();
    // A report of one game has no cross-game matrix and no contents, and is a different page
    // to drive. Both shapes ship, so both are worth being able to render.
    let alone = std::env::args().any(|argument| argument == "--one-game");
    let mut apps = vec![game(7, "A Measured Game", true)];
    if !alone {
        // Four, because a matrix of two fits on a phone and a real one does not, and what
        // happens to a table wider than the screen it is read on is the whole question.
        apps.push(game(11, "An Unmeasured Game", false));
        apps.push(game(13, "A Third Game With A Rather Long Name", true));
        apps.push(game(17, "A Fourth Game", true));
    }
    let report = Report {
        generated_unix: 1_760_000_000,
        apps,
    };
    std::fs::write(&path, census_core::html::render(&report))?;
    println!("{}", path.display());
    Ok(())
}

fn game(app_id: u32, name: &str, measured: bool) -> AppReport {
    let categories: Vec<CategoryCount> = CORE_SPINE
        .iter()
        .enumerate()
        .map(|(slot, category)| {
            // Descending counts, so sorting has something to reorder and the widest bar is
            // the first row rather than an accident. The last category is raised by nobody,
            // which every real report has and which is the only thing that puts a dash in the
            // rightmost column of the table.
            let quiet = slot + 1 == CORE_SPINE.len();
            let mentions = if quiet { 0 } else { 900 - (slot as u64 * 40) };
            CategoryCount {
                id: category.id.to_owned(),
                label: category.label.to_owned(),
                primary_count: mentions / 3,
                mention_count: mentions,
                top_mention_count: if quiet { 0 } else { (slot as u64).min(9) },
                positive_mentions: mentions / 2,
            }
        })
        .collect();

    AppReport {
        crawl: CrawlFacts {
            app_id,
            name: name.to_owned(),
            review_score_desc: "Mostly Positive".to_owned(),
            rows_unique: 2_020,
            valve_total_reviews: 2_020,
            valve_total_positive: 1_400,
            valve_total_negative: 620,
            coverage: 1.0,
            snapshot_unix: 1_750_000_000,
            shards: 1,
        },
        classification: Classification {
            app_id,
            reviews: 2_000,
            positive: 1_400,
            unmatched: 0,
            top_helpful: 50,
            mention_margin: 0.015,
            spine_version: "core-3".to_owned(),
            model: "a-test-encoder".to_owned(),
            anchors_fitted_from: vec![app_id],
            months: months(&categories),
            languages: vec![("english".to_owned(), 1_500), ("schinese".to_owned(), 500)],
            top_reviews: Vec::new(),
            categories,
        },
        examples: CORE_SPINE
            .iter()
            .map(|category| {
                (
                    category.id.to_owned(),
                    vec![
                        quoted(app_id, category.id, false),
                        quoted(app_id, category.id, true),
                    ],
                )
            })
            .collect(),
        top: vec![quoted(app_id, CORE_SPINE[0].id, true)],
        agreement: measured.then(|| agreement(app_id)),
    }
}

fn months(categories: &[CategoryCount]) -> Vec<Month> {
    (1..=12)
        .map(|month| Month {
            label: format!("2024-{month:02}"),
            reviews: 100 + month * 7,
            positive: 60 + month * 4,
            categories: categories
                .iter()
                .enumerate()
                .map(|(slot, _)| (month * 3 + slot as u64) % 40)
                .collect(),
        })
        .collect()
}

fn quoted(app_id: u32, category: &str, from_the_top: bool) -> Example {
    Example {
        review: CapturedReview {
            id: format!("{app_id}-{category}-{from_the_top}"),
            // Long enough that the fold and its control are rendered.
            text: format!(
                "A review filed under {category}. {}",
                "It goes on at some length so that the page has something to fold away. ".repeat(8)
            ),
            language: "english".to_owned(),
            author_steamid: "76561190000000000".to_owned(),
            voted_up: !from_the_top,
            votes_up: if from_the_top { 900 } else { 3 },
            votes_funny: 1,
            playtime_at_review_minutes: 4_200,
            created: 1_700_000_000,
        },
        primary: category.to_owned(),
        mentions: vec![category.to_owned(), CORE_SPINE[0].id.to_owned()],
        from_the_top,
    }
}

/// A measured game, including one category the classifier is scored as barely finding, since
/// that row renders a warning nothing else on the page does.
fn agreement(app_id: u32) -> AgreementReport {
    let categories: Vec<CategoryAgreement> = CORE_SPINE
        .iter()
        .enumerate()
        .map(|(slot, category)| {
            let reference_mentions = 60 - (slot as u64 * 2);
            let found = if slot == 3 {
                reference_mentions / 10
            } else {
                reference_mentions * 2 / 3
            };
            CategoryAgreement {
                id: category.id,
                label: category.label,
                reference_primary: reference_mentions,
                predicted_primary: reference_mentions,
                primary_agreed: found,
                reference_mentions,
                predicted_mentions: reference_mentions,
                mention_agreed: found,
                taken_as: Vec::new(),
            }
        })
        .collect();

    AgreementReport {
        apps: vec![app_id],
        produced_by: "a fixture".to_owned(),
        compared: 240,
        unmatched: 0,
        primary_agreement: Some(0.6),
        anchors: vec![format!("fitted:{app_id}")],
        slices: vec![Slice {
            subset: "random".to_owned(),
            contested: None,
            compared: 100,
            agreed: 60,
        }],
        categories,
    }
}
