//! Rendering a report as one self-contained page.
//!
//! Self-contained is the whole constraint. No fonts, no scripts, no stylesheets from
//! anywhere: a corpus that never left the machine must not start leaving it the moment
//! somebody looks at it, and a page that phones out is also a page that stops working when
//! the network does or when the host it depended on goes away.
//!
//! The page answers one question in three widening steps. A sentence says what reading only
//! the top of the pile would have told you and what everyone actually said. A table gives
//! every category the same treatment. Every row opens onto the reviews it counted, each
//! linking back to Steam, because a rate nobody can check is just an assertion.

use std::fmt::Write as _;

use crate::{
    read::SubjectCount,
    report::{AppReport, Example, Report, coverage},
    taxonomy::CORE_SPINE,
};

/// Characters of a review shown before it is folded away behind a control.
const PREVIEW_CHARS: usize = 320;

/// Languages listed before the tail is summarised as a count.
const LANGUAGES_SHOWN: usize = 12;

/// The timeline's own coordinate space. The SVG scales to whatever width it is given, so
/// these are only the numbers the shapes are drawn in.
const WIDTH: f64 = 1000.0;
const HEIGHT: f64 = 160.0;
const SPARK_HEIGHT: f64 = 40.0;

/// Reviews a month needs before its rates are drawn. Below this a single review moves the
/// figure by tens of points, and one such month would set the scale for every month that has
/// something to say.
const ENOUGH_FOR_A_RATE: u64 = 30;

/// The numeric columns of a game's table, in order, each with what it counts.
///
/// One list rather than two, so a column cannot reach the table without saying what it means.
/// `{top}` is how many reviews Steam ranks as most helpful, which is a per-game setting.
const COLUMNS: [(&str, &str); 6] = [
    (
        "Mention rate",
        "The share of all reviews that say something about the category, with the number of \
         reviews beside it. These are the headline figures and they add up to more than 100%.",
    ),
    (
        "Main subject",
        "The share whose single main subject is that category. Unlike mention rates, these add \
         up to the number of reviews.",
    ),
    (
        "Top of the pile",
        "The mention rate over the {top} reviews Steam ranks as most helpful, which is roughly \
         what somebody sees before deciding.",
    ),
    (
        "Bias",
        "How much the top of the pile overstates a category. 0\u{d7} means none of those few \
         dozen reviews raised it, which is weak evidence rather than proof of absence.",
    ),
    (
        "Said about it",
        "Of the reviews raising a category, how many praise it, how many complain about it, \
         and how many do both. A subject half the players love and half hate, and one every \
         player has mixed feelings about, are different findings that a single share hides.",
    ),
    (
        "Recommended",
        "The share of the reviews raising a category that still recommended the game, against \
         this game's own baseline. Marked warm or cold only where the gap is wider than the \
         number of reviews behind it can explain.",
    ),
];

/// A rate below which no game can be said to talk about a subject more than another.
///
/// It is the point at which the page stops printing a number and prints "less than" instead,
/// which is the same judgement: under it there is a rate, and there is nothing to rank.
const TOO_SMALL_TO_RANK: f64 = 0.001;

/// Renders the whole report.
#[must_use]
pub fn render(report: &Report) -> String {
    let mut out = String::with_capacity(1 << 18);
    out.push_str("<!doctype html>\n<html lang=\"en\">\n<head>\n");
    out.push_str("<meta charset=\"utf-8\">\n");
    out.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
    let _ = writeln!(out, "<title>{}</title>", escape(&page_title(report)));
    let _ = writeln!(out, "<style>{STYLE}</style>");
    out.push_str("</head>\n<body>\n");

    out.push_str("<a class=\"skip\" href=\"#main\">Skip to the numbers</a>\n");
    page_header(&mut out, report);
    out.push_str("<main id=\"main\">\n");
    contents(&mut out, report);
    overview(&mut out, report);
    let several = report.apps.len() > 1;
    for app in &report.apps {
        game(&mut out, app, several);
    }
    out.push_str("</main>\n");
    page_footer(&mut out, report);
    let _ = writeln!(out, "<script>{SCRIPT}</script>");
    out.push_str("</body>\n</html>\n");
    out
}

fn page_title(report: &Report) -> String {
    match report.apps.as_slice() {
        [only] => format!("{} review census", only.crawl.title()),
        _ => "Steam review census".to_owned(),
    }
}

fn page_header(out: &mut String, report: &Report) {
    out.push_str("<header class=\"page\">\n<div class=\"wrap\">\n");
    let _ = writeln!(out, "<h1>{}</h1>", escape(&page_title(report)));
    out.push_str(
        "<p class=\"lede\">What every reviewer said, counted, against what the loudest \
         handful said.</p>\n",
    );
    // Named for the state it turns on rather than for the thing it changes: a control called
    // "Theme" announced as pressed or not pressed says nothing about which theme that is.
    out.push_str(
        "<button class=\"theme\" type=\"button\" data-theme-toggle aria-pressed=\"false\">\
         <span aria-hidden=\"true\">◐</span> Dark theme</button>\n",
    );
    filter(out, report);
    out.push_str("</div>\n</header>\n");
}

/// Narrows every table on the page to the categories whose name matches.
///
/// Hidden in the markup rather than by a stylesheet rule, so a reader without scripting is
/// never offered a box that does nothing. Two dozen categories across dozens of games is more
/// than anyone can scan for one subject.
fn filter(out: &mut String, report: &Report) {
    // Written once and handed to the script as well, so clearing the box puts back the
    // wording the page was rendered with rather than a second phrasing of the same thing.
    let showing = format!(
        "all {} of them{}",
        CORE_SPINE.len(),
        // Not "in every table": a report of several games also carries one comparing the
        // corpora themselves, which has no categories in it to narrow.
        if report.apps.len() > 1 {
            ", everywhere they appear"
        } else {
            ""
        }
    );
    let _ = writeln!(
        out,
        "<form class=\"filter\" role=\"search\" hidden data-filter \
         aria-label=\"Filter the report by category\">\n\
         <label for=\"filter-categories\">Show categories matching</label>\n\
         <input id=\"filter-categories\" type=\"search\" autocomplete=\"off\" spellcheck=\"false\" \
         placeholder=\"price, story, crashes\u{2026}\" data-filter-input>\n\
         <span class=\"filter-count\" role=\"status\" data-filter-count \
         data-showing-everything=\"{0}\">{0}</span>\n</form>",
        escape(&showing)
    );
}

/// A way to reach each game once there is more than one to reach.
fn contents(out: &mut String, report: &Report) {
    if report.apps.len() < 2 {
        return;
    }
    out.push_str("<nav class=\"contents\" id=\"games\" aria-label=\"Games in this report\">\n<div class=\"wrap\">\n<ul>\n");
    for app in &report.apps {
        let _ = writeln!(
            out,
            "<li><a href=\"#app-{}\"><span class=\"nav-name\">{}</span>\
             <span class=\"nav-count\">{}</span></a></li>",
            app.app_id(),
            escape(&app.crawl.title()),
            thousands(app.reading.reviews)
        );
    }
    out.push_str("</ul>\n</div>\n</nav>\n");
}

/// Every game against every other, which is the only view a census across games can give
/// and no single game's page ever can.
fn overview(out: &mut String, report: &Report) {
    if report.apps.len() < 2 {
        return;
    }
    let total: u64 = report
        .apps
        .iter()
        .map(|app| app.reading.reviews)
        .sum();

    out.push_str("<section class=\"overview\">\n<div class=\"wrap\">\n");
    out.push_str("<h2>Across these games</h2>\n");
    corpora(out, report);
    let _ = writeln!(
        out,
        "<p class=\"note\">Mention rates for {} reviews of {} games. The game that raises each \
         subject most is named beside it, and its cell outlined, where one clears every other \
         by more than rounding. Read down a column for one game and across a row to compare \
         them all; the table scrolls sideways. Deeper shading is a higher rate, and the rates \
         are not comparable to any other corpus.</p>",
        thousands(total),
        report.apps.len()
    );

    matrix(out, report);
    most_overstated(out, report);
    widest_apart(out, report);
    pooled_agreement(out, report);
    out.push_str("</div>\n</section>\n");
}

/// The corpora themselves, before the subjects in them.
///
/// The section compared what the games talk about and never the games themselves, so how big
/// each one is, how warmly it is reviewed and how well the classifier does on it were one
/// visit per game away from each other.
fn corpora(out: &mut String, report: &Report) {
    out.push_str("<div class=\"scroll\">\n<table class=\"corpora\">\n<thead><tr>");
    heading(out, "Game", false);
    for column in ["Reviews counted", "Recommended", "Agreement"] {
        heading(out, column, true);
    }
    out.push_str("</tr></thead>\n<tbody>\n");
    for app in &report.apps {
        let measured = app
            .agreement
            .report()
            .and_then(crate::measure::ClaimAgreement::rate);
        // Sorted on the number rather than on the text of it: "982,291" reads as less than
        // "2,000" to anything comparing strings, and a game with no reference set has to sort
        // as unmeasured rather than as zero agreement.
        let sortable = |value: Option<f64>| value.unwrap_or(-1.0);
        let _ = writeln!(
            out,
            "<tr class=\"row\"><th scope=\"row\"><a href=\"#app-{}\">{}</a></th>\
             <td class=\"num\" data-value=\"{}\">{}</td>\
             <td class=\"num\" data-value=\"{:.6}\">{}</td>\
             <td class=\"num\" data-value=\"{:.6}\">{}</td></tr>",
            app.app_id(),
            escape(&app.crawl.title()),
            app.reading.reviews,
            thousands(app.reading.reviews),
            sortable(app.positive_baseline()),
            app.positive_baseline()
                .map_or_else(|| nothing("no reviews"), percent),
            sortable(measured),
            measured.map_or_else(|| nothing(&unmeasured(&app.agreement)), percent)
        );
    }
    out.push_str("</tbody>\n</table>\n</div>\n");
    out.push_str(
        "<p class=\"note\">Agreement is measured over each game's labelled claims, and only \
         over the claims the model was willing to answer, so a game's figure carries a band \
         several points wide and the pooled one at the end of this section is the one worth \
         quoting. It counts agreement with a labeller, which is not the same as being \
         right.</p>\n",
    );
}

/// Why a game carries no agreement figure, in the few words a dash can be read out as.
fn unmeasured(measurement: &crate::report::Measurement) -> String {
    match measurement {
        crate::report::Measurement::OtherTaxonomy(version) => {
            format!("labelled against {version}")
        }
        _ => "no reference set".to_owned(),
    }
}

/// Every category against every game, loudest subject first.
///
/// A census of a genre runs to dozens of games, and a screen holds about eight of them. The
/// answer a row exists to give, which game raises this subject most, was carried only by an
/// outline around one cell, so on a wide table it was usually off the side of the screen and
/// a reader had no way of knowing it was there. Naming that game in a column beside the
/// category means the row still answers its question at any width.
fn matrix(out: &mut String, report: &Report) {
    let mut order: Vec<(&str, &str, u64)> = Vec::new();
    for category in CORE_SPINE {
        let pooled: u64 = report
            .apps
            .iter()
            .flat_map(|app| app.reading.subjects.iter())
            .filter(|c| c.id == category.id)
            .map(|c| c.mention_reviews)
            .sum();
        order.push((category.id, category.label, pooled));
    }
    order.sort_by_key(|(_, _, pooled)| std::cmp::Reverse(*pooled));

    out.push_str("<div class=\"scroll\">\n<table class=\"matrix\">\n<thead><tr>");
    out.push_str("<th scope=\"col\">Category</th><th scope=\"col\" class=\"leader\">Highest</th>");
    for app in &report.apps {
        let _ = write!(
            out,
            "<th scope=\"col\" class=\"num\"><a href=\"#app-{}\">{}</a></th>",
            app.app_id(),
            escape(&app.crawl.title())
        );
    }
    out.push_str("</tr></thead>\n<tbody>\n");

    for (id, label, pooled) in order {
        if pooled == 0 {
            continue;
        }
        let rates: Vec<Option<f64>> = report
            .apps
            .iter()
            .map(|app| {
                app.reading
                    .subjects
                    .iter()
                    .find(|c| c.id == id)
                    .and_then(|c| app.rate(c.mention_reviews))
            })
            .collect();
        let loudest = belongs_to(&rates);

        let _ = write!(
            out,
            "<tr data-name=\"{}\"><th scope=\"row\">{}</th>",
            escape(&label.to_lowercase()),
            escape(label)
        );
        match loudest.and_then(|index| Some((report.apps.get(index)?, rates[index]?))) {
            Some((app, rate)) => {
                let _ = write!(
                    out,
                    "<td class=\"leader\"><b>{}</b> <a href=\"#app-{}\">{}</a></td>",
                    percent(rate),
                    app.app_id(),
                    escape(&app.crawl.title())
                );
            }
            None => {
                let _ = write!(
                    out,
                    "<td class=\"leader\">{}</td>",
                    nothing("no game leads this")
                );
            }
        }
        for (index, rate) in rates.iter().enumerate() {
            match *rate {
                Some(rate) => {
                    // Square-rooted so the common categories do not wash every other cell
                    // out; the number is always there for anyone reading exactly.
                    let heat = (rate * 2.0).sqrt().min(1.0);
                    let top = if loudest == Some(index) {
                        " loudest"
                    } else {
                        ""
                    };
                    let _ = write!(
                        out,
                        "<td class=\"num heat{top}\" style=\"--heat:{heat:.3}\">{}{}</td>",
                        percent(rate),
                        if top.is_empty() {
                            ""
                        } else {
                            "<span class=\"read-aloud\">, the highest of these games</span>"
                        }
                    );
                }
                None => {
                    let _ = write!(out, "<td class=\"num\">{}</td>", nothing("no reviews"));
                }
            }
        }
        out.push_str("</tr>\n");
    }
    out.push_str("</tbody>\n</table>\n</div>\n");
}

/// The game a row belongs to, where the table can honestly say there is one.
///
/// Which game a subject belongs to is the question a reader brings to a row, and reading it
/// off six shades of the same colour is guesswork. Two things have to hold before the table
/// answers it. The rate has to be one the page prints as a number rather than as "less than":
/// three reviews in a million beating two is not a subject belonging to a game. And the
/// leader has to be told apart from the runner-up in the figures actually printed, because a
/// reader who sees the same number twice with one of them outlined is being shown a rounding
/// difference dressed as a finding.
fn belongs_to(rates: &[Option<f64>]) -> Option<usize> {
    let (best, highest) = rates
        .iter()
        .enumerate()
        .filter_map(|(index, rate)| rate.map(|rate| (index, rate)))
        .max_by(|a, b| a.1.total_cmp(&b.1))?;
    if highest < TOO_SMALL_TO_RANK {
        return None;
    }
    let leader = percent(highest);
    rates
        .iter()
        .enumerate()
        .all(|(index, rate)| index == best || rate.is_none_or(|rate| percent(rate) != leader))
        .then_some(best)
}

/// The claim the whole tool exists to make, over every game at once.
///
/// Each game's section opens with the same sentence about that game, and a single corpus can
/// always be answered with "that is just that game". Made of every game in the report, it
/// cannot be, and the reader is told how many corpora it took.
fn most_overstated(out: &mut String, report: &Report) {
    let Some((category, factor)) = report.worst_bias() else {
        return;
    };
    let Some(overall) = category.rate() else {
        return;
    };
    let _ = writeln!(
        out,
        "<p class=\"headline\">Across these {games} games the top of the pile overstates \
         <strong>{}</strong> by <strong>{factor:.1}\u{d7}</strong>: {} of the {} reviews at the \
         top of those {games} piles raise it, against {} of all {}.</p>",
        escape(&category.label.to_lowercase()),
        thousands(category.top_mentions),
        thousands(category.top_reviews),
        percent(overall),
        thousands(category.reviews),
        games = report.apps.len()
    );
}

/// The one finding only a table of several games can carry.
///
/// Every game's own section opens with a sentence; this one opened with a grid and left the
/// reader to find the interesting row. What a cross-game view is for is the subject that
/// belongs to one game and not the others, so that is what it now says.
fn widest_apart(out: &mut String, report: &Report) {
    let rate = |app: &AppReport, id: &str| {
        app.reading
            .subjects
            .iter()
            .find(|c| c.id == id)
            .and_then(|c| app.rate(c.mention_reviews))
    };

    let mut widest: Option<(f64, &str, &AppReport, f64, &AppReport, f64)> = None;
    for category in CORE_SPINE {
        let mut rates: Vec<(&AppReport, f64)> = report
            .apps
            .iter()
            .filter_map(|app| rate(app, category.id).map(|found| (app, found)))
            .collect();
        rates.sort_by(|a, b| a.1.total_cmp(&b.1));
        let (Some(low), Some(high)) = (rates.first(), rates.last()) else {
            continue;
        };
        let spread = high.1 - low.1;
        if widest.is_none_or(|(found, ..)| spread > found) {
            widest = Some((spread, category.label, high.0, high.1, low.0, low.1));
        }
    }

    let Some((spread, label, most, high, least, low)) = widest else {
        return;
    };
    if spread < TOO_SMALL_TO_RANK {
        return;
    }
    let _ = writeln!(
        out,
        "<p class=\"headline\">The subject these games disagree about most is \
         <strong>{}</strong>: {} of {} reviews raise it, against {} of {}. A rate is about \
         one corpus, and this is what that means.</p>",
        escape(&label.to_lowercase()),
        percent(high),
        escape(&most.crawl.title()),
        percent(low),
        escape(&least.crawl.title())
    );
}

/// What the model is measured to get wrong, over every game at once.
///
/// One game's few hundred labelled claims put a wide band around its own figure. The pooled
/// number is the one worth quoting, and it exists only when every game in the report has
/// labelled claims behind it: pooling the measured ones and reporting that as the set's
/// agreement would quietly be an average over whichever games happen to be labelled.
fn pooled_agreement(out: &mut String, report: &Report) {
    let measured: Vec<crate::measure::ClaimAgreement> = report
        .apps
        .iter()
        .filter_map(|app| app.agreement.report().cloned())
        .collect();
    if measured.len() != report.apps.len() || measured.len() < 2 {
        return;
    }
    agreement_note(out, &crate::measure::pooled(&measured));
}

fn game(out: &mut String, app: &AppReport, several: bool) {
    let _ = writeln!(
        out,
        "<section class=\"game\" id=\"app-{}\">\n<div class=\"wrap\">",
        app.app_id()
    );
    let _ = writeln!(out, "<h2>{}</h2>", escape(&app.crawl.title()));
    facts(out, app);
    headline(out, app);
    over_time(out, app);
    categories(out, app);
    induced(out, app);
    top_of_the_pile(out, app);
    languages(out, app);
    trust(out, app);
    // A section runs to a hundred rows and the evidence under them, so the way out of one is
    // worth stating rather than leaving to a scrollbar.
    let _ = writeln!(
        out,
        "<p class=\"back\"><a href=\"#{}\">{}</a></p>",
        if several { "games" } else { "main" },
        if several {
            "Back to the games"
        } else {
            "Back to the top"
        }
    );
    out.push_str("</div>\n</section>\n");
}

fn facts(out: &mut String, app: &AppReport) {
    let crawl = &app.crawl;
    out.push_str("<dl class=\"facts\">\n");
    // Two different numbers, and a reader who sees only the smaller one beside a coverage
    // figure taken over the larger one is left to work out why they disagree. Rates are over
    // what was counted; coverage is over what was captured, blank reviews included.
    fact(
        out,
        "Reviews counted",
        &thousands(app.reading.reviews),
    );
    fact(
        out,
        "Captured",
        &format!(
            "{} of the {} Valve reports ({})",
            thousands(crawl.rows_unique),
            thousands(crawl.valve_total_reviews),
            coverage(crawl.coverage)
        ),
    );
    if !crawl.review_score_desc.is_empty() {
        fact(out, "Steam calls it", &crawl.review_score_desc);
    }
    fact(out, "Snapshot", &crate::time::day(crawl.snapshot_unix));
    out.push_str("</dl>\n");
}

fn fact(out: &mut String, term: &str, value: &str) {
    let _ = writeln!(
        out,
        "<div><dt>{}</dt><dd>{}</dd></div>",
        escape(term),
        escape(value)
    );
}

/// The finding, in a sentence, before any table asks anyone to read a number.
fn headline(out: &mut String, app: &AppReport) {
    let Some((category, factor)) = app.worst_bias() else {
        return;
    };
    let Some(overall) = app.rate(category.mention_reviews) else {
        return;
    };

    // The sentence names a category and the reader's next question is always which reviews,
    // so the name is the way to them rather than something to go hunting for in the table.
    let named = if app.examples.iter().any(|(id, _)| *id == category.id) {
        format!(
            "<a href=\"#panel-{}-{}\"><strong>{}</strong></a>",
            app.app_id(),
            escape(&category.id),
            escape(&category.label.to_lowercase())
        )
    } else {
        format!(
            "<strong>{}</strong>",
            escape(&category.label.to_lowercase())
        )
    };

    // Counted rather than given as a share of the top of the pile. A few dozen reviews is a
    // number a reader can hold, and the point of the sentence is that a claim about a whole
    // corpus is being made out of a handful, which "10.0% of them" hides.
    out.push_str("<p class=\"headline\">\n");
    let _ = write!(
        out,
        "The top of the pile overstates {named} by <strong>{factor:.1}\u{d7}</strong>: {} of \
         the {} reviews Steam ranks most helpful raise it, against {} of all {}.",
        thousands(category.top_mention_reviews),
        thousands(app.reading.top_helpful),
        percent(overall),
        thousands(app.reading.reviews),
    );
    out.push_str("\n</p>\n");
}

/// The month the line falls furthest, as a sentence rather than as a shape to point at.
///
/// Months too small to carry a rate are left out for the same reason they are left out of the
/// sparkline: four reviews and one thumb up is 25% and is not the month a game was hated in.
///
/// A game whose launch month is both its busiest and its angriest, which is most of the ones
/// worth reporting on, would otherwise have that month and its size named twice in three
/// lines.
fn worst_month(months: &[crate::read::Month], busiest: Option<&crate::read::Month>) -> String {
    let coldest = months
        .iter()
        .filter(|month| month.reviews >= ENOUGH_FOR_A_RATE)
        .filter_map(|month| month.positive_share().map(|share| (month, share)))
        .min_by(|a, b| a.1.total_cmp(&b.1));
    coldest.map_or_else(String::new, |(month, share)| {
        if busiest.is_some_and(|peak| peak.label == month.label) {
            return format!(
                " It falls furthest in that same month, when {} of them recommended the game.",
                percent(share)
            );
        }
        format!(
            " It falls furthest in {}, when {} of {} reviews recommended the game.",
            escape(&crate::time::month_name(&month.label)),
            percent(share),
            thousands(month.reviews)
        )
    })
}

/// Reviews per month, and how many of them recommended the game.
///
/// A mention rate is one number for a corpus that took years to gather. A game review-bombed
/// for a fortnight and quiet since produces much the same rate as one grumbled about steadily
/// for a decade, and a reader shown only the rate cannot tell which they are looking at.
///
/// Drawn as inline SVG: a chart that fetched a plotting library would not be a self-contained
/// page, and this one is two shapes.
fn over_time(out: &mut String, app: &AppReport) {
    let months = &app.reading.months;
    if months.len() < 2 {
        return;
    }
    let first = crate::time::month_name(&months[0].label);
    let last = crate::time::month_name(&months[months.len() - 1].label);
    // The chart is drawn against its own busiest month and nothing else says how big that is,
    // which leaves every bar on it a shape with no size.
    let peak = months.iter().max_by_key(|month| month.reviews);
    let tallest = peak.map_or(1, |month| month.reviews).max(1);

    out.push_str("<h3>When it was said</h3>\n");
    let _ = writeln!(
        out,
        "<p class=\"note\">Reviews per month, {} to {}, against a busiest month of {} in {}. \
         The line is the share of each month that recommended the game, from none at the \
         bottom to all at the top; the dashed line is half.{} Point at a month to read \
         it.</p>",
        escape(&first),
        escape(&last),
        thousands(tallest),
        escape(&peak.map_or_else(String::new, |month| crate::time::month_name(&month.label))),
        // The dip in that line is what a reader looks for and the one thing on the chart a
        // pointer is needed to read. It costs a clause to say, and a pointer is a thing not
        // every reader has.
        worst_month(months, peak)
    );

    #[expect(
        clippy::cast_precision_loss,
        reason = "a corpus spans hundreds of months at most"
    )]
    let step = WIDTH / months.len() as f64;

    let _ = writeln!(
        out,
        "<figure class=\"timeline\">\n<svg viewBox=\"0 0 {WIDTH:.0} {HEIGHT:.0}\" \
         preserveAspectRatio=\"none\" role=\"img\" \
         aria-label=\"Reviews per month from {} to {}\">",
        escape(&first),
        escape(&last)
    );

    for (index, month) in months.iter().enumerate() {
        #[expect(
            clippy::cast_precision_loss,
            reason = "counts and month numbers are both far below 2^53"
        )]
        let (x, tall) = (
            index as f64 * step,
            month.reviews as f64 / tallest as f64 * HEIGHT,
        );
        let _ = writeln!(
            out,
            "<rect class=\"bar\" x=\"{x:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{tall:.2}\" />",
            HEIGHT - tall,
            (step * 0.82).max(0.5)
        );
    }

    // The share line is halfway up when half a month recommended the game, which is the
    // reading nobody can do off a line with nothing to measure it against.
    let _ = writeln!(
        out,
        "<line class=\"midline\" x1=\"0\" y1=\"{0:.2}\" x2=\"{WIDTH:.0}\" y2=\"{0:.2}\" />",
        HEIGHT / 2.0
    );

    let line: Vec<String> = months
        .iter()
        .enumerate()
        .filter_map(|(index, month)| {
            let share = month.positive_share()?;
            #[expect(
                clippy::cast_precision_loss,
                reason = "a corpus spans hundreds of months at most"
            )]
            let x = index as f64 * step + step / 2.0;
            Some(format!("{x:.2},{:.2}", (1.0 - share) * HEIGHT))
        })
        .collect();
    if line.len() > 1 {
        let _ = writeln!(
            out,
            "<polyline class=\"share\" points=\"{}\" />",
            line.join(" ")
        );
    }

    // Last, and the full height of the chart: a quiet month is a bar one pixel tall, which is
    // nothing to aim at. These are what the pointer actually finds.
    for (index, month) in months.iter().enumerate() {
        #[expect(
            clippy::cast_precision_loss,
            reason = "a corpus spans hundreds of months at most"
        )]
        let x = index as f64 * step;
        let _ = writeln!(
            out,
            "<rect class=\"hit\" x=\"{x:.2}\" y=\"0\" width=\"{:.2}\" height=\"{HEIGHT:.0}\">\
             <title>{}: {} reviews, {} recommended</title></rect>",
            step.max(0.5),
            escape(&crate::time::month_name(&month.label)),
            thousands(month.reviews),
            month
                .positive_share()
                .map_or_else(|| "no".to_owned(), percent)
        );
    }

    let _ = writeln!(
        out,
        // The two ends of the axis sit at the two ends of the caption, and read as one word
        // to anything that hears the page rather than seeing it.
        "</svg>\n<figcaption><span>{}</span><span class=\"read-aloud\"> to </span>\
         <span>{}</span></figcaption>\n</figure>",
        escape(&first),
        escape(&last)
    );
}

fn categories(out: &mut String, app: &AppReport) {
    out.push_str("<h3>What players talk about</h3>\n");
    if let Some(baseline) = app.positive_baseline() {
        let _ = writeln!(
            out,
            "<p class=\"note baseline\">{} of all {} reviews recommend this game. Every figure \
             in the last column is worth reading against that.</p>",
            percent(baseline),
            thousands(app.reading.reviews)
        );
    }
    out.push_str(
        "<p class=\"note\">A review counts towards every category it says something about, so \
         these add up to more than 100%. Select a row to read the reviews behind it.</p>\n",
    );
    legend(out, app);

    let mut rows: Vec<&SubjectCount> = app.reading.subjects.iter().collect();
    rows.sort_by_key(|category| std::cmp::Reverse(category.mention_reviews));
    let widest = rows.first().map_or(0, |c| c.mention_reviews).max(1);

    out.push_str("<div class=\"scroll\">\n<table class=\"categories\">\n<thead><tr>");
    heading(out, "Category", false);
    for (name, _) in COLUMNS {
        heading(out, name, true);
    }
    out.push_str("</tr></thead>\n<tbody>\n");

    for category in rows {
        category_row(out, app, category, widest);
    }
    out.push_str("</tbody>\n</table>\n</div>\n");
}

/// What each column means, folded away.
///
/// Five columns need five definitions and two of them were going unexplained, but a reader
/// who already knows them should not have to scroll past six lines of prose in every one of
/// six sections to reach the table. A native disclosure: it needs no scripting, and printing
/// opens it along with everything else folded.
fn legend(out: &mut String, app: &AppReport) {
    out.push_str(
        "<details class=\"legend\">\n\
         <summary>What the columns mean, and where the quoted reviews come from</summary>\n\
         <dl>\n",
    );
    let top = thousands(app.reading.top_helpful);
    for (name, meaning) in COLUMNS {
        let _ = writeln!(
            out,
            "<div><dt>{name}</dt><dd>{}</dd></div>",
            meaning.replace("{top}", &top)
        );
    }
    out.push_str("</dl>\n");

    // Otherwise the two identical columns on those rows look like a mistake.
    let alone: Vec<&str> = CORE_SPINE
        .iter()
        .filter(|category| category.alone)
        .map(|category| category.label)
        .collect();
    if !alone.is_empty() {
        let _ = writeln!(
            out,
            "<p>{} are claims that no aspect was named, so nothing else can be true of the \
             same review and their mention rate is their main-subject share.</p>",
            escape(&alone.join(" and "))
        );
    }
    out.push_str(
        "<p>The reviews behind a row are a couple from the top of the pile where the category \
         reaches it, and the rest drawn at random from everything filed there, so they are \
         evidence rather than a selection.</p>\n</details>\n",
    );
}

/// A column header that can reorder the table.
///
/// A button rather than a clickable cell, so it is reachable and announced without the page
/// reinventing what a button is. With scripting off it is an inert label and the table keeps
/// the order it was rendered in.
fn heading(out: &mut String, label: &str, numeric: bool) {
    let class = if numeric { " class=\"num\"" } else { "" };
    let _ = write!(
        out,
        "<th scope=\"col\"{class} aria-sort=\"none\">\
         <button type=\"button\" class=\"sort\" data-sort>{}\
         <span class=\"arrow\" aria-hidden=\"true\"></span></button></th>",
        escape(label)
    );
}

fn category_row(out: &mut String, app: &AppReport, category: &SubjectCount, widest: u64) {
    let examples = app
        .examples
        .iter()
        .find(|(id, _)| *id == category.id)
        .map(|(_, quoted)| quoted.as_slice())
        .unwrap_or_default();
    let has_examples = !examples.is_empty();
    let panel = format!("panel-{}-{}", app.app_id(), category.id);

    let _ = write!(
        out,
        "<tr class=\"row\" data-name=\"{}\"",
        escape(&category.label.to_lowercase())
    );
    // Named as expandable but not dressed as a control. A `role` here would take the row out
    // of the table for a screen reader, which is a worse trade than it sounds: the cells stop
    // being cells. The control that opens it is built by the script, which is also the only
    // circumstance in which there is anything to open.
    if has_examples {
        let _ = write!(out, " data-expands=\"{panel}\"");
    }
    out.push('>');

    let measured = app
        .agreement
        .report()
        .and_then(|report| report.subjects.iter().find(|s| s.id == category.id));

    let _ = write!(
        out,
        "<th scope=\"row\"><span class=\"name\">{}{}{}</span></th>",
        escape(&category.label),
        measured.map(thinly_measured).unwrap_or_default(),
        if has_examples {
            "<span class=\"chevron\" aria-hidden=\"true\"></span>"
        } else {
            ""
        }
    );

    #[expect(
        clippy::cast_precision_loss,
        reason = "review counts are far below 2^53"
    )]
    let share = category.mention_reviews as f64 / widest as f64;
    let _ = write!(
        out,
        "<td class=\"num bar-cell\" data-value=\"{}\">\
         <span class=\"bar\" style=\"--fill:{:.4}\"></span>\
         <span class=\"value\">{}</span><span class=\"count\">{}</span></td>",
        category.mention_reviews,
        share,
        app.rate(category.mention_reviews)
            .map_or_else(|| nothing("no reviews"), percent),
        thousands(category.mention_reviews)
    );
    rate_cell(out, app.rate(category.primary_reviews));
    rate_cell(out, app.top_rate(category.top_mention_reviews));
    bias_cell(out, app.bias(category));
    polarity_cell(out, category);
    verdict_cell(out, category, app.positive_baseline());
    out.push_str("</tr>\n");

    if has_examples {
        let _ = writeln!(out, "<tr class=\"panel\" id=\"{panel}\"><td colspan=\"7\">");
        if let Some(measured) = measured {
            // The share of all claims the model filed here, declined ones included, because
            // the labels the correction rests on were measured over declined claims too.
            let observed = (app.reading.claims > 0).then(|| {
                #[expect(clippy::cast_precision_loss, reason = "claim counts are far below 2^53")]
                let share = category.claims as f64 / app.reading.claims as f64;
                share
            });
            how_well_this_row_is_known(out, measured, observed);
        }
        sparkline(out, app, &category.id);
        what_they_said(out, app, category, examples);
        out.push_str("</td></tr>");
    }
}

/// Reviews the reference set must raise a category in before its recall means anything.
///
/// Below this the measurement is a handful of reviews and its own interval is wider than
/// any finding, so calling the row weak would be reading noise back as a warning.
const ENOUGH_TO_JUDGE_A_ROW: u64 = 10;

/// Recall below which the number in this row is standing on very little.
const THINLY_FOUND: f64 = 0.25;

/// A mark against a rate the model is measured to miss most of.
///
/// Only on the rows that earn it. Decorating every row with its own score would make the
/// table harder to read and the warning worth less exactly where it matters.
fn thinly_measured(measured: &crate::measure::SubjectAgreement) -> String {
    if measured.labelled < ENOUGH_TO_JUDGE_A_ROW {
        return String::new();
    }
    let Some(recall) = measured.recall() else {
        return String::new();
    };
    if recall >= THINLY_FOUND {
        return String::new();
    }
    format!(
        "<span class=\"thin\" title=\"Found in only {} of the claims measured to make it, \
         so this rate is a floor rather than a count\">\u{2757}\
         <span class=\"read-aloud\">measured to miss most of this subject</span></span>",
        percent(recall)
    )
}

/// What the reference set says about this row alone.
///
/// The page carries one figure for how often the model agrees overall, and that figure is no
/// guide at all to a particular row: the same run finds nine claims in ten of one subject and
/// one in twenty of another.
fn how_well_this_row_is_known(
    out: &mut String,
    measured: &crate::measure::SubjectAgreement,
    observed_claim_share: Option<f64>,
) {
    if measured.labelled == 0 {
        let _ = writeln!(
            out,
            "<p class=\"note\">No labelled claim is about this subject, so nothing here is \
             measured.</p>"
        );
        return;
    }
    let found = measured
        .recall()
        .map_or_else(|| "none of them".to_owned(), percent);
    let right = measured
        .precision()
        .map_or_else(|| "nothing it claimed".to_owned(), percent);
    let class = if measured.labelled >= ENOUGH_TO_JUDGE_A_ROW
        && measured.recall().is_some_and(|r| r < THINLY_FOUND)
    {
        "note warn"
    } else {
        "note"
    };
    let _ = writeln!(
        out,
        "<p class=\"{class}\">Measured on this row: of the {} labelled claims about it, the \
         model found {found}; of the claims it filed here, {right} were labelled that way. A \
         rate built on a subject it misses is a floor, not a count.{}</p>",
        thousands(measured.labelled),
        // Which subject the misses went to is the difference between a row that is merely hard
        // and a row whose claims are sitting under a neighbour's name.
        measured
            .mistaken_for
            .map_or_else(String::new, |(label, count)| {
                format!(
                    " Where it was read wrongly, it was most often read as \
                     <strong>{}</strong> ({count} of the labelled claims).",
                    escape(label)
                )
            })
    );

    // The measured errors are not only a warning, they are an estimate of the true rate. The
    // observed share is too high by what was filed here wrongly and too low by what was missed
    // or declined, and both are measured, so the arithmetic is the one every prevalence study
    // does. Only where the model finds the subject clearly better than chance: otherwise the
    // observed share carries no information about the true one and the figure is invented.
    if measured.labelled >= ENOUGH_TO_JUDGE_A_ROW
        && let Some(observed) = observed_claim_share
    {
        match measured.corrected(observed) {
            Some(corrected) => {
                let _ = writeln!(
                    out,
                    "<p class=\"note\">Corrected for those errors, the share of claims about \
                     this is about <strong>{}</strong>, against the {} the model read. That is \
                     an estimate from a few hundred labels, and it moves with them.</p>",
                    percent(corrected),
                    percent(observed)
                );
            }
            None => {
                let _ = writeln!(
                    out,
                    "<p class=\"note\">The model finds this subject little better than chance, \
                     so its rate cannot be corrected: the share it read says almost nothing \
                     about the share there is.</p>"
                );
            }
        }
    }
}

/// One category's mention rate month by month.
///
/// The same argument as the volume chart, one level down: a category at 5% of a corpus may
/// have been 40% of one month and absent since, and only the shape says which.
fn sparkline(out: &mut String, app: &AppReport, id: &str) {
    let Some(slot) = CORE_SPINE.iter().position(|c| c.id == id) else {
        return;
    };
    let months = &app.reading.months;
    if months.len() < 3 {
        return;
    }
    // A month with three reviews in it can be 100% of anything, and one such month would set
    // the scale for every month that has something to say.
    let rates: Vec<Option<f64>> = months
        .iter()
        .map(|month| {
            (month.reviews >= ENOUGH_FOR_A_RATE)
                .then(|| month.rate(slot))
                .flatten()
        })
        .collect();
    let peak = rates
        .iter()
        .flatten()
        .copied()
        .fold(0.0_f64, f64::max)
        .max(f64::EPSILON);

    #[expect(
        clippy::cast_precision_loss,
        reason = "a corpus spans hundreds of months at most"
    )]
    let step = WIDTH / (months.len() - 1).max(1) as f64;
    // Kept on the corpus's own axis rather than stretched across the months that survived, so
    // a line that stops halfway means the game went quiet there and not that the subject did.
    let drawn: Vec<usize> = rates
        .iter()
        .enumerate()
        .filter_map(|(index, rate)| rate.map(|_| index))
        .collect();
    let points: Vec<String> = drawn
        .iter()
        .filter_map(|&index| {
            let rate = rates[index]?;
            #[expect(
                clippy::cast_precision_loss,
                reason = "a corpus spans hundreds of months at most"
            )]
            let x = index as f64 * step;
            Some(format!("{x:.2},{:.2}", (1.0 - rate / peak) * SPARK_HEIGHT))
        })
        .collect();
    let (Some(&opens), Some(&closes)) = (drawn.first(), drawn.last()) else {
        return;
    };
    if points.len() < 3 {
        return;
    }

    let name = |index: usize| escape(&crate::time::month_name(&months[index].label));
    let _ = writeln!(
        out,
        "<figure class=\"spark\">\n<svg viewBox=\"0 0 {WIDTH:.0} {SPARK_HEIGHT:.0}\" \
         preserveAspectRatio=\"none\" role=\"img\" aria-label=\"Mention rate by month\">\
         <line class=\"axis\" x1=\"0\" y1=\"{SPARK_HEIGHT:.0}\" x2=\"{WIDTH:.0}\" \
         y2=\"{SPARK_HEIGHT:.0}\" />\
         <polyline points=\"{}\" /></svg>\n\
         <figcaption>Mention rate by month, peaking at {}. Drawn from {} to {} on an axis \
         running to {}: a month with fewer than {ENOUGH_FOR_A_RATE} reviews carries no \
         rate.</figcaption>\n</figure>",
        points.join(" "),
        percent(peak),
        name(opens),
        name(closes),
        name(months.len() - 1)
    );
}

/// A rate, carrying the number it was formatted from so the table can be reordered by it.
fn rate_cell(out: &mut String, rate: Option<f64>) {
    match rate {
        Some(rate) => {
            let _ = write!(
                out,
                "<td class=\"num\" data-value=\"{rate:.9}\">{}</td>",
                percent(rate)
            );
        }
        None => {
            let _ = write!(
                out,
                "<td class=\"num\" data-value=\"-1\">{}</td>",
                nothing("no reviews")
            );
        }
    }
}

/// Bias is the point, so it gets a bar that leaves the middle rather than a bare number.
fn bias_cell(out: &mut String, factor: Option<f64>) {
    let Some(factor) = factor else {
        let _ = write!(
            out,
            "<td class=\"num\" data-value=\"-1\">{}</td>",
            nothing("not measured")
        );
        return;
    };
    // Log scale: twice as often and half as often are the same distance from the middle,
    // which a linear scale would draw as wildly different.
    let offset = (factor.log2() / 3.0).clamp(-1.0, 1.0);
    let side = if offset >= 0.0 { "over" } else { "under" };
    let _ = write!(
        out,
        "<td class=\"num bias\" data-value=\"{factor:.6}\"><span class=\"gauge {side}\" style=\"--offset:{:.4}\"></span>\
         <span class=\"value\">{factor:.1}\u{d7}</span></td>",
        offset.abs()
    );
}

/// Whether a topic is raised by people recommending the game or refusing to.
///
/// Shown against the corpus baseline, because a category where 80% recommend the game is
/// only interesting once a reader knows whether 80% is high or low for that game.
///
/// Coloured against the interval the count is entitled to rather than against a fixed
/// distance from the baseline. Eight reviews all recommending the game is 100% and says
/// nothing; a fixed threshold paints it the same green as a thousand reviews at 98%, which
/// is the one reading the column exists to prevent.
/// What the reviews raising a subject actually say about it, as three shares in one bar.
///
/// The mixed share is the point of the column. A subject a third of players praise and a third
/// complain about is a fight; one where every review says both is a subject with a real
/// trade-off in it, and a single positive share renders those two identically.
///
/// Sorted on the complaint share, because a table of subjects is read looking for what is
/// wrong, and sorting on praise puts the answer at the bottom.
fn polarity_cell(out: &mut String, category: &SubjectCount) {
    let said = category.praised + category.criticised + category.mixed;
    if said == 0 {
        let _ = write!(
            out,
            "<td class=\"num\" data-value=\"-1\">{}</td>",
            nothing("none raised it")
        );
        return;
    }

    let share = |part: u64| share_of(part, said);
    let (praise, gripe, both) = (
        share(category.praised),
        share(category.criticised),
        share(category.mixed),
    );
    let _ = write!(
        out,
        "<td class=\"num said\" data-value=\"{gripe:.9}\">\
         <span class=\"mix\" aria-hidden=\"true\">\
         <span class=\"praise\" style=\"--part:{praise:.4}\"></span>\
         <span class=\"gripe\" style=\"--part:{gripe:.4}\"></span>\
         <span class=\"both\" style=\"--part:{both:.4}\"></span></span>\
         <span class=\"value\">{} praise</span>\
         <span class=\"count\">{} gripe, {} both</span></td>",
        percent(praise),
        percent(gripe),
        percent(both)
    );
}

fn verdict_cell(out: &mut String, category: &SubjectCount, baseline: Option<f64>) {
    let Some(share) = category.positive_share() else {
        let _ = write!(
            out,
            "<td class=\"num\" data-value=\"-1\">{}</td>",
            nothing("none raised it")
        );
        return;
    };
    let spread = crate::measure::wilson(category.positive_mentions, category.mention_reviews);
    let tone = match (baseline, spread) {
        (Some(baseline), Some((low, _))) if low > baseline => " warmer",
        (Some(baseline), Some((_, high))) if high < baseline => " colder",
        _ => "",
    };
    let _ = write!(
        out,
        "<td class=\"num verdict-share{tone}\" data-value=\"{share:.9}\">{}</td>",
        percent(share)
    );
}

/// What this game's players talk about that no game shares.
///
/// Shown apart from the table above, and without a rate, because these have no readings: a
/// subject was found by reading a hundred reviews chosen to be as unlike each other as the
/// corpus allows, and the model that counts the table has never been trained on it. What can
/// honestly be shown is the subject, a sentence on it, and the reviews it was found in.
fn induced(out: &mut String, app: &AppReport) {
    if app.induced.is_empty() {
        return;
    }
    out.push_str("<h3>What this game's players talk about that others' do not</h3>\n");
    let _ = writeln!(
        out,
        "<p class=\"note\">Found by reading reviews of this game chosen to be as unlike each \
         other as possible, and named only where at least three of them raise the same thing. \
         These rows carry no rate: the model that counts the table above has not been trained \
         on them, so what is shown is the finding and the reviews it rests on.</p>"
    );
    out.push_str("<dl class=\"induced\">\n");
    for found in &app.induced {
        let subject = &found.subject;
        let _ = write!(
            out,
            "<div class=\"found\"><dt>{}",
            escape(&subject.label)
        );
        if let Some(parent) = subject
            .refines
            .as_deref()
            .and_then(|id| CORE_SPINE.iter().find(|c| c.id == id))
        {
            let _ = write!(
                out,
                " <span class=\"chip\">a form of {}</span>",
                escape(parent.label)
            );
        }
        let _ = writeln!(
            out,
            "</dt>\n<dd><p>{}</p>",
            escape(&subject.description)
        );
        if !found.reviews.is_empty() {
            let _ = writeln!(
                out,
                "<details><summary>{} of the {} reviews it was found in</summary>",
                found.reviews.len(),
                subject.evidence.len()
            );
            out.push_str("<ol class=\"reviews\">\n");
            for review in &found.reviews {
                // The whole review, with no polarity and no confidence, because the
                // counting model never read it and the renderer shows neither for one it
                // did not.
                let example = Example {
                    review: review.clone(),
                    claim: review.text.clone(),
                    index: 0,
                    polarity: String::new(),
                    confidence: 0.0,
                    also: Vec::new(),
                    from_the_top: false,
                };
                self::review(out, app, &example);
            }
            out.push_str("</ol>\n</details>\n");
        }
        out.push_str("</dd></div>\n");
    }
    out.push_str("</dl>\n");
}

fn top_of_the_pile(out: &mut String, app: &AppReport) {
    if app.top.is_empty() {
        return;
    }
    out.push_str("<h3>The top of the pile</h3>\n");
    let _ = writeln!(
        out,
        "<p class=\"note\">The {} reviews Steam ranks as most helpful, which is roughly what a \
         reader sees before deciding. Every rate above is measured against the whole corpus \
         instead.</p>",
        thousands(app.reading.top_helpful)
    );
    // A native disclosure rather than a scripted one: it folds a long list away without the
    // page needing to work for it, and it still opens when scripting is off.
    // Counted from what is about to be listed rather than from what was measured, since a
    // review whose text has gone missing would otherwise leave the summary claiming one more
    // than the reader can find.
    let _ = writeln!(
        out,
        "<details class=\"pile\">\n<summary>Read all {} of them</summary>",
        thousands(app.top.len() as u64)
    );
    reviews(out, app, &app.top);
    out.push_str("</details>\n");
}

fn reviews(out: &mut String, app: &AppReport, examples: &[Example]) {
    out.push_str("<ol class=\"reviews\">\n");
    for example in examples {
        review(out, app, example);
    }
    out.push_str("</ol>\n");
}

/// The evidence behind a subject, grouped by what it says.
///
/// This is what stands in for a written summary until there is one. A reader opening a row
/// wants to know what people praise and what they complain about, and eight claims in a
/// single list make them work that out for themselves. Sorted into praise and complaint, with
/// the counts of each beside the heading, the list answers the question rather than
/// containing the answer.
fn what_they_said(out: &mut String, app: &AppReport, subject: &SubjectCount, examples: &[Example]) {
    if examples.is_empty() {
        return;
    }
    let sides: [(&str, &str, u64); 3] = [
        ("praise", "What they praise", subject.praised + subject.mixed),
        ("complaint", "What they complain about", subject.criticised + subject.mixed),
        ("neutral", "Said without judging", 0),
    ];
    for (polarity, heading, reviews) in sides {
        let shown: Vec<&Example> = examples
            .iter()
            .filter(|example| example.polarity == polarity)
            .collect();
        if shown.is_empty() {
            continue;
        }
        let _ = write!(out, "<h4 class=\"side {polarity}\">{}", escape(heading));
        if reviews > 0 {
            let _ = write!(
                out,
                " <span class=\"count\">{} reviews</span>",
                thousands(reviews)
            );
        }
        out.push_str("</h4>\n<ol class=\"reviews\">\n");
        for example in shown {
            review(out, app, example);
        }
        out.push_str("</ol>\n");
    }
}

fn review(out: &mut String, app: &AppReport, example: &Example) {
    let verdict = if example.review.voted_up {
        ("up", "Recommended")
    } else {
        ("down", "Not recommended")
    };
    out.push_str("<li class=\"review\">\n<div class=\"meta\">");
    let _ = write!(
        out,
        "<span class=\"verdict {}\">{}</span>",
        verdict.0, verdict.1
    );
    if example.review.votes_up > 0 {
        let _ = write!(
            out,
            "<span class=\"votes\">{} found this helpful</span>",
            thousands(u64::from(example.review.votes_up))
        );
    }
    if example.review.playtime_at_review_minutes > 0 {
        let _ = write!(
            out,
            "<span class=\"played\">{} played</span>",
            hours(example.review.playtime_at_review_minutes)
        );
    }
    if !example.review.language.is_empty() {
        let _ = write!(
            out,
            "<span class=\"lang\">{}</span>",
            escape(&example.review.language)
        );
    }
    if example.from_the_top {
        out.push_str("<span class=\"chip top\">top of the pile</span>");
    }
    // A claim that scraped past the threshold and one the model is certain of are not equally
    // good evidence, and a page that shows them identically is inviting the wrong conclusion.
    // A review the model never read carries no polarity and gets no confidence either: an
    // invented "100% sure" on it would be the one lie this line exists to prevent.
    if example.was_read() {
        let _ = write!(
            out,
            "<span class=\"sure\">{} sure</span>",
            percent(f64::from(example.confidence))
        );
    }
    out.push_str("</div>\n");

    // The page is in English and most of the reviews on it are not. Saying so is what lets a
    // screen reader pronounce a Chinese review as Chinese rather than as English, and what
    // lets an Arabic one be laid out the way it was written.
    let tagged = bcp47(&example.review.language).map_or_else(String::new, |tag| {
        let direction = if RIGHT_TO_LEFT.contains(&tag) {
            " dir=\"rtl\""
        } else {
            ""
        };
        format!(" lang=\"{tag}\"{direction}")
    });

    // The claim is what was counted, so the claim is what is quoted. The review it came from
    // follows only where it says something the claim does not, because a reader checking a
    // count should not have to find the one sentence in thirty that earned it.
    let claim = example.claim.trim();
    let _ = write!(
        out,
        "<div class=\"text\"><p class=\"claim\"{tagged}>{}</p></div>",
        escape(claim)
    );

    let whole = example.review.text.trim();
    if whole != claim {
        let long = whole.chars().count() > PREVIEW_CHARS;
        let _ = write!(
            out,
            "<div class=\"text whole{}\"><p{tagged}>{}</p></div>",
            if long { " long" } else { "" },
            escape(whole)
        );
        out.push_str(
            "<button class=\"more\" type=\"button\" data-expands-text>Show the whole \
             review</button>\n",
        );
    }

    out.push_str("<div class=\"filed\">");
    if example.was_read() {
        let _ = write!(
            out,
            "<span class=\"chip {}\">{}</span>",
            escape(&example.polarity),
            escape(&example.polarity)
        );
    }
    for id in &example.also {
        let label = CORE_SPINE
            .iter()
            .find(|c| c.id == *id)
            .map_or(id.as_str(), |c| c.label);
        let _ = write!(out, "<span class=\"chip\">{}</span>", escape(label));
    }
    if let Some(url) = example.url(app.app_id()) {
        let _ = write!(
            out,
            "<a class=\"source\" href=\"{}\" rel=\"noopener noreferrer\" target=\"_blank\">\
             On Steam</a>",
            escape(&url)
        );
    }
    out.push_str("</div>\n</li>\n");
}

/// What read the corpus, and how sure it had to be before it would answer.
///
/// The threshold is the promise the page is making. Two reports produced by the same model at
/// different thresholds are not comparable, and a reader given the numbers without it cannot
/// tell which they have.
fn built_from(app: &AppReport) -> String {
    let labels = if app.reading.trained_on.is_empty() {
        String::new()
    } else {
        format!(
            " trained on label set <code>{}</code>,",
            escape(&app.reading.trained_on)
        )
    };
    format!(
        "{},{labels} answering only above {:.2} confidence",
        escape(&app.reading.model),
        app.reading.threshold
    )
}

/// What the reader is entitled to conclude, next to the numbers rather than in a footnote.
/// What language the corpus is in, which Steam's own page cannot show a reader at all.
fn languages(out: &mut String, app: &AppReport) {
    if app.reading.languages.is_empty() {
        return;
    }
    let total: u64 = app.reading.languages.iter().map(|(_, n)| n).sum();
    let english = app
        .reading
        .languages
        .iter()
        .find(|(name, _)| name == "english")
        .map_or(0, |(_, count)| *count);

    let not_english = share_of(total - english, total);

    out.push_str("<h3>What language it was said in</h3>\n");
    let _ = writeln!(
        out,
        "<p class=\"note\">{} of these reviews are not in English. Steam shows a reader their \
         own language by default, so most of this argument is one they never see.</p>",
        percent(not_english)
    );

    let widest = app
        .reading
        .languages
        .first()
        .map_or(1, |(_, count)| *count)
        .max(1);
    out.push_str("<ul class=\"languages\">\n");
    for (name, count) in app.reading.languages.iter().take(LANGUAGES_SHOWN) {
        #[expect(
            clippy::cast_precision_loss,
            reason = "review counts are far below 2^53"
        )]
        let fill = *count as f64 / widest as f64;
        // A share as well as a count, the way every other table on the page reads. Six
        // thousand reviews is a different thing in a corpus of thirty thousand and in one of
        // a million, and nothing else on the row says which of those this is.
        let _ = writeln!(
            out,
            "<li><span class=\"lang-name\">{}</span>\
             <span class=\"bar\" style=\"--fill:{fill:.4}\"></span>\
             <span class=\"lang-share\">{}</span><span class=\"lang-count\">{}</span></li>",
            escape(&language_name(name)),
            percent(share_of(*count, total)),
            thousands(*count)
        );
    }
    out.push_str("</ul>\n");

    let tail: u64 = app
        .reading
        .languages
        .iter()
        .skip(LANGUAGES_SHOWN)
        .map(|(_, count)| count)
        .sum();
    let rest = app
        .reading
        .languages
        .len()
        .saturating_sub(LANGUAGES_SHOWN);
    if rest > 0 {
        let _ = writeln!(
            out,
            "<p class=\"note\">and {rest} more languages, {} of the corpus between them.</p>",
            percent(share_of(tail, total))
        );
    }
}

/// A share, where a denominator of nothing is nothing rather than a division by zero.
fn share_of(part: u64, whole: u64) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "review counts are far below 2^53"
    )]
    if whole > 0 {
        part as f64 / whole as f64
    } else {
        0.0
    }
}

fn trust(out: &mut String, app: &AppReport) {
    out.push_str("<h3>How far to trust this</h3>\n");
    out.push_str("<dl class=\"facts wide\">\n");

    fact(out, "Read by", &built_from(app));
    fact(out, "Taxonomy", &app.reading.spine_version);
    // Shallow numbers are not deep numbers with less work in them. A review about six things
    // read as one point is about none of them clearly, and a page that did not say which way
    // it was read would invite comparing the two.
    if app.reading.depth != crate::read::Depth::Deep {
        fact(
            out,
            "Depth",
            "shallow: each review read as one point, so anyone who wrote more than a sentence \
             is understated. Not comparable with a deep reading.",
        );
    }
    fact(
        out,
        "Claims read",
        &format!(
            "{} from {} reviews",
            thousands(app.reading.claims),
            thousands(app.reading.reviews)
        ),
    );

    // The single most important number on the page for reading every other one. A model
    // answering a fifth of the claims is not describing the corpus, it is describing the
    // fifth it was sure about, and no rate below can be read without knowing that.
    if let Some(share) = app.reading.unclassified_share() {
        fact(
            out,
            "Claims it would not answer",
            &format!(
                "{} ({}), counted as unclassified rather than filed under a best guess",
                thousands(app.reading.unclassified_claims),
                percent(share)
            ),
        );
    }
    if app.reading.silent_reviews > 0 {
        fact(
            out,
            "Reviews it said nothing about",
            &format!(
                "{} ({} of those read), where no point cleared the threshold",
                thousands(app.reading.silent_reviews),
                percent(share_of(app.reading.silent_reviews, app.reading.reviews))
            ),
        );
    }

    // The capture holds more rows than any rate is taken over, and a reader who subtracts
    // the two deserves the difference named rather than left to guess at it.
    let blank = app.crawl.rows_unique.saturating_sub(app.reading.reviews);
    if blank > 0 {
        let why = if app.reading.language.is_some() {
            "another language, or a rating and nothing else"
        } else {
            "a rating and nothing else"
        };
        fact(
            out,
            "Reviews not read",
            &format!("{} ({why}, counted in no rate)", thousands(blank)),
        );
    }
    out.push_str("</dl>\n");

    match &app.agreement {
        crate::report::Measurement::Measured(agreement) => agreement_note(out, agreement),
        crate::report::Measurement::Unlabelled => out.push_str(
            "<p class=\"warn\">No claims have been labelled for this game, so how often the \
             model is wrong here has not been measured. Treat every rate as provisional.</p>\n",
        ),
        // Not the same thing as nobody having labelled it, and telling a reader it is sends
        // them to do work that is already done.
        crate::report::Measurement::OtherTaxonomy(version) => {
            let _ = writeln!(
                out,
                "<p class=\"warn\">This game has a reference set, labelled against taxonomy \
                 {}, and these subjects are {}. Nothing here is measured against it, because \
                 that would score the model on subjects nobody labelling it was offered. Label \
                 the set again to measure this game.</p>",
                escape(version),
                escape(&app.reading.spine_version)
            );
        }
    }

    out.push_str(
        "<p class=\"note\">These rates are a census, not a survey: every review Valve serves \
         was counted, so there is no sampling error to report. What they do carry is \
         classifier error, which is what the agreement figure above measures.</p>\n",
    );
}

fn agreement_note(out: &mut String, agreement: &crate::measure::ClaimAgreement) {
    let (Some(rate), Some((low, high))) = (agreement.rate(), agreement.interval()) else {
        return;
    };
    let _ = writeln!(
        out,
        "<p class=\"warn\">Of the {} labelled claims this model was willing to answer, it \
         named the same subject a separate labeller did {} of the time, somewhere in [{}, {}] \
         with 95% confidence. The labeller was itself a language model, so that is \
         <strong>agreement, not accuracy</strong>: two models can be wrong together, most \
         easily on sarcasm and on claims that sit between subjects.</p>",
        thousands(agreement.answered),
        percent(rate),
        percent(low),
        percent(high)
    );

    // The figure above is computed over answered claims only, which is the honest way to score
    // a model that abstains and the dishonest way to describe what it did to a corpus. Both
    // numbers or neither.
    if let Some(declined) = agreement.declined_share() {
        let _ = writeln!(
            out,
            "<p class=\"note\">That figure covers the {} of those claims it answered. It \
             declined the other {}, which are counted as unclassified everywhere on this page \
             rather than being filed under a best guess.</p>",
            percent(1.0 - declined),
            percent(declined)
        );
    }

    // One figure for a whole taxonomy hides the shape of the error: the same run finds nine
    // claims in ten of one subject and one in twenty of another.
    let judged: Vec<&crate::measure::SubjectAgreement> = agreement
        .subjects
        .iter()
        .filter(|s| s.labelled >= ENOUGH_TO_JUDGE_A_ROW)
        .collect();
    if judged.is_empty() {
        return;
    }
    let thin = judged
        .iter()
        .filter(|s| s.recall().is_some_and(|recall| recall < THINLY_FOUND))
        .count();
    let _ = writeln!(
        out,
        "<p class=\"note\">Averaged over the subjects rather than over the claims, so a rare \
         one counts as much as a common one, that comes to <strong>{:.2}</strong> on a scale \
         where 1 is perfect agreement. {}</p>",
        agreement.macro_f1().unwrap_or(0.0),
        if thin == 0 {
            format!(
                "Every one of the {} subjects with enough labels to judge is found in at \
                 least a quarter of the claims making it.",
                judged.len()
            )
        } else if thin == 1 {
            format!(
                "One of the {} subjects with enough labels to judge is found in fewer than \
                 a quarter of the claims making it, and its row is marked: read that rate \
                 as a floor.",
                judged.len()
            )
        } else {
            format!(
                "{thin} of the {} subjects with enough labels to judge are found in fewer \
                 than a quarter of the claims making them, and their rows are marked: read \
                 those rates as floors.",
                judged.len()
            )
        }
    );

    // A subject read as one particular other subject is a boundary the taxonomy has not
    // settled, and no amount of training settles it for the taxonomy. Worth naming, because it
    // is the one kind of error a reader of this page can act on.
    let worst = agreement
        .subjects
        .iter()
        .filter(|s| s.labelled >= ENOUGH_TO_JUDGE_A_ROW)
        .filter_map(|s| s.mistaken_for.map(|(other, count)| (s, other, count)))
        .max_by_key(|(_, _, count)| *count);
    if let Some((subject, other, count)) = worst {
        let _ = writeln!(
            out,
            "<p class=\"note\">Where it disagrees most, {} claims the labeller called \
             {} were read as {}. That is a boundary between two subjects rather than a \
             mistake about one of them.</p>",
            thousands(count),
            escape(subject.label),
            escape(other)
        );
    }
}

fn page_footer(out: &mut String, report: &Report) {
    out.push_str("<footer class=\"page\">\n<div class=\"wrap\">\n");
    out.push_str("<h3>What this cannot tell you</h3>\n<ul>\n");
    for limit in FOOTER_LIMITS {
        let _ = writeln!(out, "<li>{limit}</li>");
    }
    out.push_str("</ul>\n");
    let _ = writeln!(
        out,
        "<p class=\"note\">Rendered on {} by steam-review-census, from captures taken on the \
         dates given above. Nothing on this page was sent anywhere to produce it.</p>",
        escape(&crate::time::day(report.generated_unix))
    );
    out.push_str("</div>\n</footer>\n");
}

const FOOTER_LIMITS: [&str; 4] = [
    "Every review Valve will serve is not every review ever written. Reviews from banned \
     accounts, deleted reviews, and reviews the API stops paging through are missing, and \
     nobody outside Valve can measure how many.",
    "A category assignment is a machine reading one review once. It has no idea whether a \
     joke is a joke, and sarcasm is exactly where it is weakest.",
    "Counting how often something is mentioned says nothing about whether the people saying \
     it are right, or whether the ones who never mentioned it disagree.",
    "The top of the pile is Steam's own ordering, which changes over time. A capture is a \
     photograph of it, not a permanent fact.",
];

/// Steam's own names for languages, the tags a browser understands, and what to call them
/// on screen.
///
/// Steam uses names of its own: "schinese", "koreana", "brazilian", "latam". A page that
/// repeats those tells a screen reader nothing and a reader not much more.
const LANGUAGES: [(&str, &str, &str); 31] = [
    ("english", "en", "English"),
    ("schinese", "zh-Hans", "Chinese (simplified)"),
    ("tchinese", "zh-Hant", "Chinese (traditional)"),
    ("japanese", "ja", "Japanese"),
    ("koreana", "ko", "Korean"),
    ("thai", "th", "Thai"),
    ("bulgarian", "bg", "Bulgarian"),
    ("czech", "cs", "Czech"),
    ("danish", "da", "Danish"),
    ("german", "de", "German"),
    ("greek", "el", "Greek"),
    ("spanish", "es", "Spanish"),
    ("latam", "es-419", "Spanish (Latin America)"),
    ("finnish", "fi", "Finnish"),
    ("french", "fr", "French"),
    ("hungarian", "hu", "Hungarian"),
    ("indonesian", "id", "Indonesian"),
    ("italian", "it", "Italian"),
    ("dutch", "nl", "Dutch"),
    ("norwegian", "no", "Norwegian"),
    ("polish", "pl", "Polish"),
    ("portuguese", "pt", "Portuguese"),
    ("brazilian", "pt-BR", "Portuguese (Brazil)"),
    ("romanian", "ro", "Romanian"),
    ("russian", "ru", "Russian"),
    ("swedish", "sv", "Swedish"),
    ("turkish", "tr", "Turkish"),
    ("ukrainian", "uk", "Ukrainian"),
    ("vietnamese", "vi", "Vietnamese"),
    ("arabic", "ar", "Arabic"),
    ("malay", "ms", "Malay"),
];

/// Tags whose script runs the other way, so a review in one is laid out the other way.
///
/// Arabic is the only one Steam offers as a review language, and two reviews of the three
/// million measured here are in it. Two reviews rendered backwards are still two reviews
/// rendered backwards, and the fix is one attribute.
const RIGHT_TO_LEFT: [&str; 1] = ["ar"];

/// The BCP 47 tag for a Steam language name.
///
/// An unknown name gets no tag rather than a guess: an element inheriting the page's English
/// is a smaller error than one claiming to be a language it is not.
fn bcp47(steam: &str) -> Option<&'static str> {
    LANGUAGES
        .iter()
        .find(|(name, _, _)| *name == steam)
        .map(|(_, tag, _)| *tag)
}

/// What to call a Steam language on screen, falling back to whatever Steam called it.
fn language_name(steam: &str) -> String {
    LANGUAGES
        .iter()
        .find(|(name, _, _)| *name == steam)
        .map_or_else(|| steam.to_owned(), |(_, _, display)| (*display).to_owned())
}

fn escape(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for character in raw.chars() {
        match character {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(character),
        }
    }
    out
}

/// A cell with no number behind it.
///
/// A dash reads as absence at a glance and as silence to a screen reader, so why the cell is
/// empty is spelled out for anyone who cannot see the column it sits in.
fn nothing(reason: &str) -> String {
    format!(
        "<span aria-hidden=\"true\">\u{2013}</span>\
         <span class=\"read-aloud\">{}</span>",
        escape(reason)
    )
}

fn percent(rate: f64) -> String {
    if rate > 0.0 && rate < 0.001 {
        return "<0.1%".to_owned();
    }
    format!("{:.1}%", rate * 100.0)
}

fn thousands(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

fn hours(minutes: u32) -> String {
    if minutes < 60 {
        return format!("{minutes} min");
    }
    format!("{} h", minutes / 60)
}

const STYLE: &str = include_str!("report.css");
const SCRIPT: &str = include_str!("report.js");

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_report(text: &str) -> Report {
        let category =
            |id: &str, label: &str, mentions: u64, top: u64| crate::read::SubjectCount {
                id: id.to_owned(),
                label: label.to_owned(),
                primary_reviews: mentions / 2,
                mention_reviews: mentions,
                claims: mentions * 2,
                praised: mentions / 3,
                criticised: mentions / 3,
                mixed: mentions / 6,
                top_mention_reviews: top,
                positive_mentions: mentions / 3,
            };
        let example = Example {
            review: crate::capture::CapturedReview {
                id: "42".to_owned(),
                text: text.to_owned(),
                language: "english".to_owned(),
                author_steamid: "7656119".to_owned(),
                voted_up: false,
                votes_up: 12,
                votes_funny: 0,
                playtime_at_review_minutes: 600,
                created: 1_700_000_000,
            },
            claim: text.to_owned(),
            index: 0,
            polarity: "complaint".to_owned(),
            confidence: 0.87,
            also: vec!["bugs".to_owned(), "performance".to_owned()],
            from_the_top: true,
        };
        Report {
            generated_unix: 1_700_000_000,
            apps: vec![AppReport {
                crawl: crate::report::CrawlFacts {
                    app_id: 7,
                    name: "A Game <& Friends>".to_owned(),
                    review_score_desc: "Mostly Positive".to_owned(),
                    rows_unique: 1_000,
                    valve_total_reviews: 1_000,
                    valve_total_positive: 700,
                    valve_total_negative: 300,
                    coverage: 1.0,
                    snapshot_unix: 1_700_000_000,
                    shards: 1,
                },
                reading: crate::read::ReadReport {
                    app_id: 7,
                    reviews: 1_000,
                    corpus_reviews: 1_000,
                    language: None,
                    depth: crate::read::Depth::Deep,
                    claims: 3_000,
                    unclassified_claims: 300,
                    silent_reviews: 3,
                    positive: 700,
                    top_helpful: 50,
                    spine_version: crate::CORE_SPINE_VERSION.to_owned(),
                    model: "test-reader".to_owned(),
                    trained_on: "0123456789abcdef".to_owned(),
                    threshold: 0.5,
                    device: "cpu".to_owned(),
                    subjects: vec![
                        category("bugs", "Bugs and crashes", 400, 30),
                        category("performance", "Performance", 100, 2),
                    ],
                    languages: vec![("english".to_owned(), 600), ("schinese".to_owned(), 400)],
                    months: vec![
                        crate::read::Month {
                            label: "2024-01".to_owned(),
                            reviews: 400,
                            positive: 320,
                            subjects: vec![40, 100],
                        },
                        crate::read::Month {
                            label: "2024-02".to_owned(),
                            reviews: 600,
                            positive: 380,
                            subjects: vec![60, 300],
                        },
                    ],
                    elapsed: std::time::Duration::ZERO,
                },
                examples: vec![("bugs".to_owned(), vec![example])],
                top: Vec::new(),
                agreement: crate::report::Measurement::Unlabelled,
                induced: Vec::new(),
            }],
        }
    }

    /// A game measured against labelled claims, agreeing on `agreed` of `answered`.
    fn measured(answered: u64, agreed: u64) -> crate::measure::ClaimAgreement {
        crate::measure::ClaimAgreement {
            app_id: 7,
            matched: answered,
            answered,
            agreed,
            declined: 0,
            polarity_answered: answered,
            polarity_agreed: agreed,
            clear_answered: answered,
            clear_agreed: agreed,
            contested_answered: 0,
            contested_agreed: 0,
            subjects: Vec::new(),
        }
    }

    /// The same report with a calendar of its own.
    fn with_months(months: Vec<crate::read::Month>) -> Report {
        let mut report = sample_report("ordinary text");
        report.apps[0].reading.months = months;
        report
    }

    fn month(label: &str, reviews: u64, bugs: u64) -> crate::read::Month {
        crate::read::Month {
            label: label.to_owned(),
            reviews,
            positive: reviews / 2,
            subjects: vec![0, bugs],
        }
    }

    #[test]
    fn every_month_gets_a_bar_however_quiet_it_was() {
        let months = vec![
            month("2024-01", 400, 40),
            month("2024-02", 5, 1),
            month("2024-03", 300, 30),
            month("2024-04", 200, 20),
        ];
        let page = render(&with_months(months));

        assert_eq!(
            page.matches("<rect class=\"bar\"").count(),
            4,
            "a quiet month is a fact about the game and belongs on the chart"
        );
        assert!(page.contains("Jan 2024"), "the first month should be named");
        assert!(page.contains("Apr 2024"), "the last month should be named");
    }

    #[test]
    fn a_month_too_small_to_carry_a_rate_stays_out_of_the_sparkline() {
        // One review in a five-review month is 20% and would set the scale for every month
        // that has something to say. The categories are the same in both cases; only the
        // month sizes differ, and only the second should reach the chart.
        let noisy = vec![
            month("2024-01", 100, 10),
            month("2024-02", 5, 5),
            month("2024-03", 100, 10),
            month("2024-04", 100, 10),
        ];
        let page = render(&with_months(noisy));

        assert!(
            page.contains("peaking at 10.0%"),
            "a five-review month set the scale"
        );
        assert!(page.contains("a month with fewer than 30 reviews carries no rate"));
        // The line stops where the rates stop, so the caption names where it stops and not
        // the last month of a corpus it never reached.
        let quietened = vec![
            month("2024-01", 100, 10),
            month("2024-02", 100, 10),
            month("2024-03", 100, 10),
            month("2024-04", 5, 1),
            month("2024-05", 5, 1),
        ];
        let page = render(&with_months(quietened));
        assert!(
            page.contains("Drawn from Jan 2024 to Mar 2024 on an axis running to May 2024"),
            "the caption claims months the line never reached"
        );
    }

    #[test]
    fn a_review_cannot_break_out_of_the_page_it_is_quoted_in() {
        // Review text is written by strangers and this one is trying. Nothing it contains
        // may reach the browser as markup.
        let hostile = "</p></td></tr></table><script>alert(1)</script><img src=x onerror=1>";
        let page = render(&sample_report(hostile));

        assert!(!page.contains("<script>alert(1)"), "a script tag survived");
        assert!(!page.contains("<img src=x"), "an image tag survived");
        assert!(
            page.contains("&lt;script&gt;alert(1)"),
            "the text itself is missing"
        );
        // The game's own name is equally untrusted, coming from the store.
        assert!(page.contains("A Game &lt;&amp; Friends&gt;"));
    }

    #[test]
    fn the_page_fetches_nothing_and_says_what_it_cannot_tell_you() {
        let page = render(&sample_report("ordinary text"));

        for fetching in ["<script src", "<link ", "<img ", "@import", "url("] {
            assert!(!page.contains(fetching), "the page would fetch: {fetching}");
        }
        assert!(page.contains("What this cannot tell you"));
        assert!(
            page.contains("No claims have been labelled"),
            "a report with no measured agreement must say so"
        );
    }

    #[test]
    fn nothing_offered_to_a_reader_without_scripting_does_nothing() {
        let page = render(&sample_report("ordinary text"));

        let filter = page
            .split_once("<form class=\"filter\"")
            .expect("the category filter is missing")
            .1;
        let opening = filter.split_once('>').expect("an unclosed form tag").0;
        assert!(
            opening.contains(" hidden"),
            "the filter is offered before scripting has said it works: {opening}"
        );
        assert!(
            SCRIPT.contains("form.hidden = false"),
            "nothing ever reveals the filter"
        );
    }

    /// A rate the model is measured to miss most of is not a count, and a reader scanning the
    /// table has no way to tell the two apart unless the page says so.
    #[test]
    fn a_row_the_model_barely_finds_is_marked_as_one() {
        let scored = |labelled, agreed| crate::measure::SubjectAgreement {
            id: "bugs",
            label: "Bugs and crashes",
            labelled,
            read: agreed,
            agreed,
            seen: labelled * 4,
            mistaken_for: None,
        };

        assert!(
            thinly_measured(&scored(100, 4)).contains("class=\"thin\""),
            "four found in a hundred is a floor, not a count"
        );
        assert!(
            thinly_measured(&scored(100, 90)).is_empty(),
            "a row the model finds should carry no warning"
        );
        assert!(
            thinly_measured(&scored(4, 0)).is_empty(),
            "four labelled claims cannot condemn a row"
        );

        let mut out = String::new();
        how_well_this_row_is_known(&mut out, &scored(100, 4), Some(0.05));
        assert!(out.contains("found 4.0%"), "{out}");
        assert!(out.contains("100 labelled claims"), "{out}");
        assert!(
            out.contains("note warn"),
            "a row this thin should look thin: {out}"
        );
        assert!(
            out.contains("cannot be corrected"),
            "a subject found in 4% of its claims is chance, and a corrected rate from it would \
             be invented: {out}"
        );

        let mut well = String::new();
        how_well_this_row_is_known(&mut well, &scored(100, 80), Some(0.2));
        assert!(
            well.contains("Corrected for those errors"),
            "a subject found four times in five earns a corrected rate: {well}"
        );

        let mut none = String::new();
        how_well_this_row_is_known(&mut none, &scored(0, 0), Some(0.1));
        assert!(none.contains("nothing here is measured"), "{none}");
    }

    /// The filter matches on this and nothing else, so a row without it silently stops
    /// being findable and a panel with it would be filtered away from its own row.
    #[test]
    fn every_row_a_reader_can_filter_carries_the_name_being_matched() {
        let page = render(&sample_report("ordinary text"));

        assert!(page.contains("data-name=\"bugs and crashes\""));
        assert!(page.contains("data-name=\"performance\""));
        assert_eq!(
            page.matches("data-name=").count(),
            2,
            "one game has two categories and no other row should claim a name"
        );
        for panel in page.split("<tr class=\"panel\"").skip(1) {
            let opening = panel.split_once('>').expect("an unclosed row").0;
            assert!(
                !opening.contains("data-name"),
                "a panel named itself: {opening}"
            );
        }
    }

    /// A capture holding more rows than the rates are taken over is normal and invites
    /// exactly one question, so the page answers it rather than leaving the reader to
    /// subtract two numbers and wonder.
    #[test]
    fn the_gap_between_what_was_captured_and_what_was_counted_is_named() {
        let mut report = sample_report("ordinary text");
        report.apps[0].crawl.rows_unique = 1_010;
        report.apps[0].reading.reviews = 1_000;

        let page = render(&report);
        assert!(
            page.contains("1,010 of the 1,000 Valve reports"),
            "the two numbers a reader has to reconcile are not both on the page"
        );
        assert!(
            page.contains("Reviews not read"),
            "the reviews that were never read go unmentioned"
        );
        assert!(
            page.contains("10 (a rating and nothing else"),
            "wrong count for what was captured but not read"
        );
        assert!(
            page.contains("Claims it would not answer"),
            "a page that does not say how much the model declined cannot be read at all"
        );

        report.apps[0].crawl.rows_unique = 1_000;
        let tidy = render(&report);
        assert!(
            !tidy.contains("Reviews not read"),
            "a capture with nothing missing should say nothing"
        );
    }

    /// Folded and filtered are two reasons a row is not on screen, and printing undoes only
    /// the first. Sharing one mechanism would print rows a reader had filtered away.
    #[test]
    fn printing_opens_what_was_folded_and_leaves_what_was_filtered() {
        assert!(
            SCRIPT.contains("classList.toggle('filtered-out'"),
            "the filter must not reach for the attribute that means folded"
        );
        let print = STYLE
            .split_once("@media print")
            .expect("nothing is written for paper")
            .1;
        assert!(
            print.contains("tr.panel[hidden]:not(.filtered-out)"),
            "printing would unfold rows the reader had filtered away"
        );
        assert!(
            STYLE.contains(".filtered-out {"),
            "the filter's class styles nothing"
        );
    }

    /// The finding names a category, and the next question is always which reviews. A link
    /// that lands on a folded row is worse than no link, so the panel has to open itself.
    #[test]
    fn the_finding_leads_to_the_reviews_behind_it() {
        let page = render(&sample_report("ordinary text"));
        let headline = page
            .split_once("<p class=\"headline\">")
            .expect("no finding")
            .1;
        let target = headline
            .split_once("<a href=\"#")
            .expect("the category is not a way to anything")
            .1
            .split_once('"')
            .expect("an unclosed href")
            .0
            .to_owned();

        assert!(
            page.contains(&format!("id=\"{target}\"")),
            "the finding points at {target}, which is not on the page"
        );
        assert!(
            SCRIPT.contains("hashchange"),
            "a second link to a second category would open nothing"
        );
        // 30 of the 50 Steam ranks most helpful raise bugs, against 40.0% of the corpus. The
        // reader is owed the count the claim is built on, not only the share it comes to.
        assert!(
            headline.contains("30 of the 50 reviews Steam ranks most helpful raise it"),
            "the finding hides how few reviews it rests on: {}",
            headline
                .split_once("</p>")
                .map_or(headline, |(head, _)| head)
        );
    }

    /// Every way through the page has to arrive somewhere, and no two things may answer to
    /// the same name. A link into the evidence is worthless if its target moved or doubled.
    #[test]
    fn nothing_on_the_page_points_at_something_that_is_not_there() {
        // Both shapes: the contents, the cross-game matrix and the way back out of a section
        // only exist once there is more than one game, and they are all links.
        for page in [
            render(&sample_report("ordinary text")),
            render(&two_games()),
        ] {
            no_dangling_links(&page);
        }
    }

    /// An induced subject's evidence was never read by the counting model, and the page must
    /// not dress it as though it had been.
    #[test]
    fn an_induced_subject_shows_its_reviews_without_inventing_a_reading() {
        let mut report = sample_report("ordinary text");
        let review = report.apps[0].examples[0].1[0].review.clone();
        report.apps[0].induced = vec![crate::report::InducedEvidence {
            subject: crate::induced::Induced {
                id: "mud-physics".to_owned(),
                label: "Mud physics".to_owned(),
                description: "How trucks behave in mud.".to_owned(),
                refines: Some("gameplay".to_owned()),
                evidence: vec!["42".to_owned(), "43".to_owned(), "44".to_owned()],
            },
            reviews: vec![review],
        }];
        let page = render(&report);

        let section = page
            .split_once("players talk about that others")
            .expect("no induced section")
            .1
            .split_once("<h3>")
            .map_or("", |(before, _)| before);
        assert!(section.contains("Mud physics"), "{section}");
        assert!(section.contains("a form of Gameplay"), "{section}");
        assert!(section.contains("1 of the 3 reviews"), "{section}");
        assert!(
            !section.contains("sure</span>"),
            "a review the model never read must not carry a confidence: {section}"
        );
        assert!(
            !section.contains("class=\"chip neutral\"") && !section.contains("class=\"chip praise\""),
            "nor a polarity: {section}"
        );

        let without = render(&sample_report("ordinary text"));
        assert!(
            !without.contains("players talk about that others"),
            "a game with nothing induced gets no empty heading"
        );
    }

    fn two_games() -> Report {
        let mut report = sample_report("ordinary text");
        let mut second = report.apps[0].clone();
        second.crawl.app_id = 9;
        second.crawl.name = "Another Game".to_owned();
        second.reading.app_id = 9;
        report.apps.push(second);
        report
    }

    fn no_dangling_links(page: &str) {
        let mut ids: Vec<&str> = attributes(page, "id=\"");
        let count = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(count, ids.len(), "two things answer to the same name");

        let mut targets: Vec<&str> = attributes(page, "aria-controls=\"");
        targets.extend(
            attributes(page, "href=\"#")
                .into_iter()
                .filter(|target| !target.is_empty()),
        );
        targets.extend(attributes(page, "for=\""));
        for target in targets {
            assert!(
                ids.binary_search(&target).is_ok(),
                "the page points at {target}, which is not on it"
            );
        }
    }

    fn attributes<'a>(page: &'a str, opening: &str) -> Vec<&'a str> {
        page.match_indices(opening)
            .filter_map(|(at, _)| page[at + opening.len()..].split_once('"').map(|(v, _)| v))
            .collect()
    }

    /// Every review raising a category recommending the game is 100% whether that is eight
    /// reviews or a thousand, and only one of the two is a warm subject.
    #[test]
    fn a_verdict_share_is_called_warm_only_when_the_count_can_carry_it() {
        let warmth = |mentions: u64, positive: u64| {
            let mut report = sample_report("ordinary text");
            let app = &mut report.apps[0];
            app.reading.reviews = 1_000;
            app.reading.positive = 700;
            app.reading.subjects[1].mention_reviews = mentions;
            app.reading.subjects[1].positive_mentions = positive;
            let row = render(&report)
                .split_once("data-name=\"performance\"")
                .expect("no performance row")
                .1
                .split_once("</tr>")
                .expect("the row never ends")
                .0
                .to_owned();
            assert!(row.contains("100.0%"), "the share itself is missing: {row}");
            (row.contains("warmer"), row.contains("colder"))
        };

        assert_eq!(
            warmth(8, 8),
            (false, false),
            "eight reviews were called a warm subject against a 70% baseline"
        );
        assert_eq!(
            warmth(400, 400),
            (true, false),
            "four hundred reviews were not"
        );
    }

    /// Bars drawn against the busiest month say nothing about size until that month does,
    /// and the dip in the share line is the one thing on the chart a pointer is needed for.
    #[test]
    fn the_chart_says_how_big_the_month_it_is_drawn_against_was() {
        let page = render(&sample_report("ordinary text"));
        assert!(
            page.contains("against a busiest month of 600 in Feb 2024"),
            "the chart has no scale on it"
        );
        // The busiest month is also the angriest here, as it is on most games worth reporting
        // on, and naming it and its size twice in three lines reads as a mistake.
        assert!(
            page.contains("falls furthest in that same month, when 63.3% of them recommended"),
            "the chart's low point can only be reached with a pointer"
        );
        let note = page
            .split_once("Reviews per month,")
            .expect("no chart")
            .1
            .split_once("</p>")
            .expect("the note never ends")
            .0;
        assert!(
            !note.contains("furthest in Feb 2024"),
            "the busiest month is named again as the angriest: {note}"
        );
        assert_eq!(
            note.matches("600").count(),
            1,
            "the size of that month is given twice: {note}"
        );

        // Four reviews and none of them a thumb up is 0% and is not the month a game was
        // hated in; thirty in a hundred is.
        let mut noisy = vec![
            month("2024-01", 100, 10),
            month("2024-02", 4, 1),
            month("2024-03", 100, 10),
        ];
        noisy[0].positive = 30;
        noisy[1].positive = 0;
        assert!(
            render(&with_months(noisy))
                .contains("falls furthest in Jan 2024, when 30.0% of 100 reviews recommended"),
            "a four-review month was called the low point"
        );
    }

    /// Marking a game as the one that talks about a subject most is a claim, and on a row
    /// nobody raises there is nothing to claim: three reviews in a million beat two.
    #[test]
    fn a_subject_nobody_raises_has_no_game_it_belongs_to() {
        let mut report = two_games();
        for (app, mentions) in report.apps.iter_mut().zip([3_u64, 2]) {
            app.reading.reviews = 100_000;
            // Performance is the row that exists only because five reviews in two hundred
            // thousand mention it.
            app.reading.subjects[1].mention_reviews = mentions;
        }
        // Bugs is loud on both games and loudest on one of them, so it keeps its outline.
        report.apps[1].reading.subjects[0].mention_reviews = 300;

        let matrix = |report: &Report| {
            render(report)
                .split_once("<table class=\"matrix\">")
                .expect("no matrix")
                .1
                .split_once("</table>")
                .expect("the matrix never ends")
                .0
                .to_owned()
        };
        let performance = |grid: &str| {
            grid.split_once("<th scope=\"row\">Performance</th>")
                .expect("no performance row")
                .1
                .split_once("</tr>")
                .expect("the row never ends")
                .0
                .to_owned()
        };

        let grid = matrix(&report);
        assert!(
            grid.contains("heat loudest"),
            "a rate worth ranking is not marked"
        );
        assert!(
            !performance(&grid).contains("loudest"),
            "three reviews in a hundred thousand were called a game's subject"
        );

        report.apps[0].reading.subjects[1].mention_reviews = 4_000;
        assert!(
            performance(&matrix(&report)).contains("loudest"),
            "a game that does raise the subject is not marked"
        );

        // Both print 4.0%, so which of them is ahead is a difference the reader cannot see.
        report.apps[1].reading.subjects[1].mention_reviews = 3_999;
        let tied = performance(&matrix(&report));
        assert_eq!(
            tied.matches("4.0%").count(),
            2,
            "the tie is not on the page"
        );
        assert!(
            !tied.contains("loudest"),
            "one review in a hundred thousand was drawn as a finding"
        );
    }

    /// Every taxonomy change puts every game into this state until its set is labelled again,
    /// so it is a state the page spends real time in. Telling a reader that nobody has
    /// labelled the game sends them to do work that is already done.
    #[test]
    fn a_set_labelled_against_another_taxonomy_is_not_reported_as_no_set_at_all() {
        let mut report = two_games();
        report.apps[0].agreement = crate::report::Measurement::OtherTaxonomy("core-3".to_owned());
        report.apps[0].reading.spine_version = "core-9".to_owned();
        let page = render(&report);

        let section = page
            .split_once("id=\"app-7\"")
            .expect("no section for the game")
            .1
            .split_once("</section>")
            .expect("the section never ends")
            .0;
        assert!(
            !section.contains("No claims have been labelled"),
            "a game that has been labelled is reported as never labelled"
        );
        assert!(
            section.contains("labelled against taxonomy core-3")
                && section.contains("these subjects are core-9"),
            "the page does not say which two taxonomies disagree: {section}"
        );

        let table = page
            .split_once("<table class=\"corpora\">")
            .expect("no corpora table")
            .1;
        assert!(
            table.contains("labelled against core-3"),
            "the dash beside the game is read out as no set at all"
        );
    }

    /// A set of thirty games is a table of thirty rows, and the questions a reader brings to
    /// it are which corpus is biggest and where the classifier is weakest. Both are a sort,
    /// and a sort on the printed text puts 982,291 below 2,000 and an unmeasured game above
    /// every measured one.
    #[test]
    fn the_corpora_can_be_reordered_on_what_they_hold_rather_than_on_the_text_of_it() {
        let mut report = two_games();
        report.apps[1].reading.reviews = 982_291;
        let page = render(&report);
        let table = page
            .split_once("<table class=\"corpora\">")
            .expect("no corpora table")
            .1
            .split_once("</table>")
            .expect("the table never ends")
            .0;

        let head = table.split_once("</thead>").expect("no head").0;
        assert_eq!(
            head.matches("data-sort").count(),
            4,
            "not every column of the corpora table can reorder it: {head}"
        );
        assert!(
            table.contains("data-value=\"982291\""),
            "a review count is left to be sorted as text: {table}"
        );
        // An unmeasured game is not a game measured at zero, and sorting has to keep them
        // apart or the worst-measured game in a set is one nobody measured.
        assert!(
            table.contains("data-value=\"-1.000000\""),
            "a game with no reference set sorts as zero agreement: {table}"
        );
        assert_eq!(
            table.matches("<tr class=\"row\">").count(),
            2,
            "the rows are not the ones the sort moves"
        );
    }

    /// The claim this tool exists to make can always be answered with "that is just that
    /// game" when it is made of one corpus. Made of every corpus in the report, it cannot be,
    /// so the sentence has to be counted over all of them rather than read off the first.
    #[test]
    fn the_bias_claim_across_games_is_counted_over_all_of_them() {
        let mut report = two_games();
        let sentence = |page: &str| {
            page.split_once("<p class=\"headline\">Across these ")
                .expect("no cross-game bias claim")
                .1
                .split_once("</p>")
                .expect("the claim never ends")
                .0
                .to_owned()
        };

        let claim = sentence(&render(&report));
        assert!(
            claim.contains(
                "2 games the top of the pile overstates <strong>bugs and crashes</strong>"
            ) && claim.contains("by <strong>1.5\u{d7}</strong>"),
            "the pooled claim is not the one the pooled counts support: {claim}"
        );
        assert!(
            claim.contains("60 of the 100 reviews at the top of those 2 piles")
                && claim.contains("against 40.0% of all 2,000"),
            "the claim is counted over one game rather than both: {claim}"
        );

        // A subject the second game's most-helpful reviews are full of and its corpus is not.
        // Reading either game alone still names bugs; only pooling moves the claim.
        report.apps[1].reading.subjects[1].top_mention_reviews = 45;
        let moved = sentence(&render(&report));
        assert!(
            moved.contains("<strong>performance</strong>"),
            "a subject only the pooled counts find was not found: {moved}"
        );
    }

    /// A census of a genre is dozens of games and a screen holds about eight, so a claim
    /// carried only by an outline around one cell is a claim most readers never see.
    #[test]
    fn a_row_names_the_game_that_leads_it_rather_than_only_outlining_the_cell() {
        let mut report = two_games();
        report.apps[1].reading.subjects[0].mention_reviews =
            report.apps[0].reading.subjects[0].mention_reviews * 2;

        let row = |page: &str| {
            page.split_once("<table class=\"matrix\">")
                .expect("no matrix")
                .1
                .split_once("<th scope=\"row\">Bugs and crashes</th>")
                .expect("no bugs row")
                .1
                .split_once("</tr>")
                .expect("the row never ends")
                .0
                .to_owned()
        };

        let leading = row(&render(&report));
        let named = leading
            .split_once("class=\"leader\">")
            .expect("the row does not say which game leads it")
            .1
            .split_once("</td>")
            .expect("the summary never ends")
            .0
            .to_owned();
        assert!(
            named.contains("Another Game"),
            "the leading game is not named beside the row: {named}"
        );
        assert!(
            named.contains("href=\"#app-9\""),
            "the named game is not a way of reaching it: {named}"
        );

        // Both games raise it at the same rate, so there is nothing the page can name.
        report.apps[1].reading.subjects[0].mention_reviews =
            report.apps[0].reading.subjects[0].mention_reviews;
        assert!(
            !row(&render(&report)).contains("Another Game"),
            "a tie was reported as a game leading the row"
        );
    }

    /// A section comparing what the games talk about that never compares the games
    /// themselves leaves how big each one is and how well it is measured a visit apart.
    #[test]
    fn the_cross_game_section_compares_the_corpora_and_not_only_the_subjects() {
        let mut report = two_games();
        report.apps[0].agreement = crate::report::Measurement::Measured(Box::new(measured(100, 62)));

        let table = render(&report)
            .split_once("<table class=\"corpora\">")
            .expect("no corpora table")
            .1
            .split_once("</table>")
            .expect("the table never ends")
            .0
            .to_owned();
        assert!(table.contains("href=\"#app-7\""), "no way into a game");
        assert!(table.contains("A Game &lt;&amp; Friends&gt;"));
        assert!(table.contains(">1,000<"), "no review count: {table}");
        assert!(table.contains(">70.0%<"), "no recommended share: {table}");
        assert_eq!(
            table.matches("no reference set").count(),
            1,
            "an unmeasured game is given a figure, or a measured one is not: {table}"
        );
    }

    /// A grid of six columns leaves the reader to find the interesting row. The section
    /// says which one it is, and a report of one game has no such row to name.
    #[test]
    fn a_report_of_several_games_says_what_they_disagree_about() {
        let mut report = two_games();
        // Bugs is level across both; performance is the row that separates them.
        report.apps[1].reading.subjects[1].mention_reviews = 20;

        let page = render(&report);
        let finding = page
            .split_once("<h2>Across these games</h2>")
            .expect("no cross-game section")
            .1
            .split_once("</section>")
            .expect("the section never ends")
            .0;
        assert!(
            finding.contains("disagree about most is <strong>performance</strong>"),
            "the wrong row was named: {finding}"
        );
        assert!(finding.contains("A Game &lt;&amp; Friends&gt;"));
        assert!(finding.contains("Another Game"));

        let alone = render(&sample_report("ordinary text"));
        assert!(
            !alone.contains("disagree about most"),
            "one game cannot disagree with itself"
        );
    }

    #[test]
    fn a_cell_with_no_number_says_so_out_loud() {
        let mut out = String::new();
        let unraised = crate::read::SubjectCount {
            id: "performance".to_owned(),
            label: "Performance".to_owned(),
            primary_reviews: 0,
            mention_reviews: 0,
            claims: 0,
            praised: 0,
            criticised: 0,
            mixed: 0,
            top_mention_reviews: 0,
            positive_mentions: 0,
        };
        rate_cell(&mut out, None);
        bias_cell(&mut out, None);
        verdict_cell(&mut out, &unraised, Some(0.8));

        assert!(
            !out.contains('\u{2014}'),
            "an em dash reached the page: {out}"
        );
        assert_eq!(
            out.matches("class=\"read-aloud\"").count(),
            3,
            "a dash was left with nothing to say to a screen reader: {out}"
        );
        assert!(out.contains("no reviews"));
        assert!(out.contains("not measured"));
        assert!(out.contains("none raised it"));
    }

    #[test]
    fn every_category_with_mentions_reaches_the_page_with_its_evidence() {
        let page = render(&sample_report("ordinary text"));

        assert!(page.contains("Bugs and crashes"));
        assert!(page.contains("Performance"));
        // 400 of 1,000 mention bugs, 30 of the top 50 do: 60% against 40%.
        assert!(page.contains("40.0%"), "the corpus rate is missing");
        assert!(
            page.contains("60.0%"),
            "the top-of-the-pile rate is missing"
        );
        assert!(page.contains("1.5\u{d7}"), "the bias factor is missing");
        assert!(
            page.contains("On Steam"),
            "the link back to the source is missing"
        );
    }

    /// The declarations inside a block, given the text that opens it.
    fn declared_in(marker: &str) -> Vec<&'static str> {
        let start = STYLE
            .find(marker)
            .unwrap_or_else(|| panic!("no {marker} in the stylesheet"));
        let open = start + STYLE[start..].find('{').expect("a block with no brace");
        let mut depth = 0_i32;
        let mut end = open;
        for (offset, character) in STYLE[open..].char_indices() {
            match character {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = open + offset;
                        break;
                    }
                }
                _ => {}
            }
        }
        STYLE[open..end]
            .lines()
            .filter_map(|line| line.trim().strip_prefix("--"))
            .filter_map(|line| line.split(':').next())
            .collect()
    }

    #[test]
    fn no_colour_goes_missing_when_the_page_is_read_in_the_dark() {
        // A token used but not defined for a theme is invisible text, and only in that
        // theme, which is exactly the kind of thing nobody notices until somebody else does.
        let light = declared_in(":root {");
        let media = declared_in("(prefers-color-scheme: dark)");
        let attribute = declared_in(":root[data-theme='dark']");

        // Set on individual elements by the markup rather than by the theme.
        let inline = ["fill", "heat", "offset", "part"];
        for used in STYLE.split("var(--").skip(1) {
            let name = used.split([')', ',', ' ']).next().unwrap_or_default();
            assert!(
                light.contains(&name) || inline.contains(&name),
                "--{name} is used and never defined"
            );
        }
        for name in &light {
            assert_eq!(
                media.contains(name),
                attribute.contains(name),
                "--{name} is themed by one dark rule and not the other"
            );
        }
        assert!(!media.is_empty(), "the dark theme defines nothing at all");
    }

    #[test]
    fn every_character_that_could_close_a_tag_is_escaped() {
        // Review text is arbitrary text written by strangers. A single unescaped angle
        // bracket in a million reviews is a broken page at best.
        assert_eq!(
            escape("<script>alert(\"x\" & 'y')</script>"),
            "&lt;script&gt;alert(&quot;x&quot; &amp; &#39;y&#39;)&lt;/script&gt;"
        );
    }

    /// A column with no definition is a number with no name for what it counts, and the
    /// panel a reader would look in has to hold one for every column the table draws.
    #[test]
    fn every_column_the_table_draws_is_defined_where_a_reader_looks() {
        let page = render(&sample_report("ordinary text"));
        let legend = page
            .split_once("<details class=\"legend\">")
            .expect("no legend")
            .1
            .split_once("</details>")
            .expect("the legend never ends")
            .0;
        for (name, _) in COLUMNS {
            assert!(
                page.contains(&format!(
                    "<button type=\"button\" class=\"sort\" data-sort>{name}"
                )),
                "{name} is defined and never drawn"
            );
            assert!(
                legend.contains(&format!("<dt>{name}</dt>")),
                "{name} is drawn and never defined"
            );
        }
        assert!(
            legend.contains("the 50 reviews Steam ranks"),
            "the definition still carries its placeholder: {legend}"
        );
    }

    /// Six thousand reviews is a different thing in a corpus of thirty thousand and in one
    /// of a million, and a bar says which is bigger but never how much of the whole it is.
    #[test]
    fn every_language_says_how_much_of_the_corpus_it_is() {
        let page = render(&sample_report("ordinary text"));
        let list = page
            .split_once("<ul class=\"languages\">")
            .expect("no languages")
            .1
            .split_once("</ul>")
            .expect("the list never ends")
            .0;
        // 600 English and 400 Chinese of a thousand.
        assert!(list.contains(">60.0%<"), "English has no share: {list}");
        assert!(list.contains(">40.0%<"), "Chinese has no share: {list}");
        assert!(list.contains(">600<") && list.contains(">400<"), "{list}");
    }

    #[test]
    fn steam_language_names_become_tags_a_browser_knows() {
        assert_eq!(bcp47("schinese"), Some("zh-Hans"));
        assert_eq!(bcp47("koreana"), Some("ko"));
        assert_eq!(bcp47("brazilian"), Some("pt-BR"));
        // A language Steam adds after this was written must not be guessed at.
        assert_eq!(bcp47("klingon"), None);
        assert_eq!(language_name("klingon"), "klingon");
        assert_eq!(language_name("latam"), "Spanish (Latin America)");

        // Every tag has to be one, and no two names may claim the same one.
        let mut tags: Vec<&str> = LANGUAGES.iter().map(|(_, tag, _)| *tag).collect();
        tags.sort_unstable();
        let count = tags.len();
        tags.dedup();
        assert_eq!(tags.len(), count, "two languages share a tag");
        for (name, tag, display) in LANGUAGES {
            assert!(!name.is_empty() && !display.is_empty(), "{name} is unnamed");
            assert!(
                tag.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'),
                "{tag} is not a tag"
            );
        }
        for tag in RIGHT_TO_LEFT {
            assert!(
                tags.contains(&tag),
                "{tag} runs the other way and is not a language the page knows"
            );
        }
    }

    /// A review written right to left, laid out left to right, is unreadable in a way the
    /// page has all the information to avoid.
    #[test]
    fn a_review_is_laid_out_the_way_it_was_written() {
        let mut report = sample_report("مراجعة عن اللعبة");
        report.apps[0].examples[0].1[0].review.language = "arabic".to_owned();
        assert!(
            render(&report).contains("<p class=\"claim\" lang=\"ar\" dir=\"rtl\">"),
            "an Arabic review is laid out as English"
        );

        let english = render(&sample_report("ordinary text"));
        assert!(
            english.contains("<p class=\"claim\" lang=\"en\">"),
            "an English review has no language on it"
        );
        assert!(
            !english.contains("dir=\"rtl\""),
            "an English review is laid out backwards"
        );
    }

    #[test]
    fn a_rate_too_small_to_round_is_not_shown_as_zero() {
        // 0.0% reads as "nobody said this", which is a different claim from "almost nobody".
        assert_eq!(percent(0.0004), "<0.1%");
        assert_eq!(percent(0.0), "0.0%");
        assert_eq!(percent(0.1234), "12.3%");
    }

    #[test]
    fn thousands_groups_from_the_right() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1_000), "1,000");
        assert_eq!(thousands(1_161_047), "1,161,047");
    }
}
