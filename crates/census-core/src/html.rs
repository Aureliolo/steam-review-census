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
    report::{AppReport, CategoryCount, Example, Report},
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
    out.push_str(
        "<button class=\"theme\" type=\"button\" data-theme-toggle aria-pressed=\"false\">\
         <span aria-hidden=\"true\">◐</span> Theme</button>\n",
    );
    filter(out, report);
    out.push_str("</div>\n</header>\n");
}

/// Narrows every table on the page to the categories whose name matches.
///
/// Hidden in the markup rather than by a stylesheet rule, so a reader without scripting is
/// never offered a box that does nothing. Twenty-one categories across several games is more
/// than anyone can scan for one subject.
fn filter(out: &mut String, report: &Report) {
    // Written once and handed to the script as well, so clearing the box puts back the
    // wording the page was rendered with rather than a second phrasing of the same thing.
    let showing = format!(
        "all {} of them{}",
        CORE_SPINE.len(),
        if report.apps.len() > 1 {
            ", in every table"
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
            thousands(app.classification.reviews)
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
        .map(|app| app.classification.reviews)
        .sum();

    out.push_str("<section class=\"overview\">\n<div class=\"wrap\">\n");
    out.push_str("<h2>Across these games</h2>\n");
    let _ = writeln!(
        out,
        "<p class=\"note\">Mention rates for {} reviews of {} games. Read down a column for \
         one game and across a row to see which games a subject belongs to. Deeper shading is \
         a higher rate and the outlined cell is the highest in its row; the rates are not \
         comparable to any other corpus.</p>",
        thousands(total),
        report.apps.len()
    );

    let mut order: Vec<(&str, &str, u64)> = Vec::new();
    for category in CORE_SPINE {
        let pooled: u64 = report
            .apps
            .iter()
            .flat_map(|app| app.classification.categories.iter())
            .filter(|c| c.id == category.id)
            .map(|c| c.mention_count)
            .sum();
        order.push((category.id, category.label, pooled));
    }
    order.sort_by_key(|(_, _, pooled)| std::cmp::Reverse(*pooled));

    out.push_str("<div class=\"scroll\">\n<table class=\"matrix\">\n<thead><tr>");
    out.push_str("<th scope=\"col\">Category</th>");
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
                app.classification
                    .categories
                    .iter()
                    .find(|c| c.id == id)
                    .and_then(|c| app.rate(c.mention_count))
            })
            .collect();
        // Which game a subject belongs to most is the question a reader brings to a row, and
        // reading it off six shades of the same colour is guesswork.
        let loudest = rates
            .iter()
            .enumerate()
            .filter_map(|(index, rate)| rate.map(|rate| (index, rate)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(index, _)| index);

        let _ = write!(
            out,
            "<tr data-name=\"{}\"><th scope=\"row\">{}</th>",
            escape(&label.to_lowercase()),
            escape(label)
        );
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
    pooled_agreement(out, report);
    out.push_str("</div>\n</section>\n");
}

/// What the classifier is measured to get wrong, over every game at once.
///
/// A hundred held-back reviews put a band twenty points wide around any one game's figure.
/// Six hundred is the number worth quoting, and it exists only when every game in the report
/// has a reference set behind it.
fn pooled_agreement(out: &mut String, report: &Report) {
    let measured: Vec<crate::AgreementReport> = report
        .apps
        .iter()
        .filter_map(|app| app.agreement.clone())
        .collect();
    if measured.len() != report.apps.len() || measured.len() < 2 {
        return;
    }
    agreement_note(out, &crate::evaluate::pooled(&measured));
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
    fact(out, "Reviews", &thousands(app.classification.reviews));
    fact(
        out,
        "Of Valve's total",
        &format!(
            "{} ({})",
            thousands(crawl.valve_total_reviews),
            coverage(crawl.coverage)
        ),
    );
    if !crawl.review_score_desc.is_empty() {
        fact(out, "Steam calls it", &crawl.review_score_desc);
    }
    fact(out, "Captured", &crate::time::day(crawl.snapshot_unix));
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
    let (Some(top), Some(overall)) = (
        app.top_rate(category.top_mention_count),
        app.rate(category.mention_count),
    ) else {
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

    out.push_str("<p class=\"headline\">\n");
    let _ = write!(
        out,
        "Reading only the {} most-upvoted reviews, {} of them talk about {named}. Across all \
         {} reviews, {} do: the top of the pile overstates it by \
         <strong>{factor:.1}\u{d7}</strong>.",
        thousands(app.classification.top_helpful),
        percent(top),
        thousands(app.classification.reviews),
        percent(overall),
    );
    out.push_str("\n</p>\n");
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
    let months = &app.classification.months;
    if months.len() < 2 {
        return;
    }
    let first = crate::time::month_name(&months[0].label);
    let last = crate::time::month_name(&months[months.len() - 1].label);
    let tallest = months.iter().map(|m| m.reviews).max().unwrap_or(1).max(1);

    out.push_str("<h3>When it was said</h3>\n");
    let _ = writeln!(
        out,
        "<p class=\"note\">Reviews per month, {} to {}. The line is the share of each month \
         that recommended the game, from none at the bottom to all at the top; the dashed \
         line is half. Point at a month to read it.</p>",
        escape(&first),
        escape(&last)
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
        "</svg>\n<figcaption><span>{}</span><span>{}</span></figcaption>\n</figure>",
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
            thousands(app.classification.reviews)
        );
    }
    out.push_str(
        "<p class=\"note\">A review counts towards every category it says something about, so \
         these add up to more than 100%. <strong>Bias</strong> is how much the top of the pile \
         overstates a category: 0\u{d7} means none of those few dozen reviews raised it, which \
         is weak evidence rather than proof of absence. <strong>Recommended</strong> is the \
         share of the reviews raising a category that still recommended the game, against \
         this game's own baseline. Select a row to read the reviews behind it.</p>\n",
    );
    // Otherwise the two identical columns on those rows look like a mistake.
    let alone: Vec<&str> = CORE_SPINE
        .iter()
        .filter(|category| category.alone)
        .map(|category| category.label)
        .collect();
    if !alone.is_empty() {
        let _ = writeln!(
            out,
            "<p class=\"note\">{} are claims that no aspect was named, so nothing else can be \
             true of the same review and their mention rate is their main-subject share.</p>",
            escape(&alone.join(" and "))
        );
    }

    let mut rows: Vec<&CategoryCount> = app.classification.categories.iter().collect();
    rows.sort_by_key(|category| std::cmp::Reverse(category.mention_count));
    let widest = rows.first().map_or(0, |c| c.mention_count).max(1);

    out.push_str("<div class=\"scroll\">\n<table class=\"categories\">\n<thead><tr>");
    heading(out, "Category", false);
    heading(out, "Mention rate", true);
    heading(out, "Main subject", true);
    heading(out, "Top of the pile", true);
    heading(out, "Bias", true);
    heading(out, "Recommended", true);
    out.push_str("</tr></thead>\n<tbody>\n");

    for category in rows {
        category_row(out, app, category, widest);
    }
    out.push_str("</tbody>\n</table>\n</div>\n");
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

fn category_row(out: &mut String, app: &AppReport, category: &CategoryCount, widest: u64) {
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
        .as_ref()
        .and_then(|report| report.categories.iter().find(|c| c.id == category.id));

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
    let share = category.mention_count as f64 / widest as f64;
    let _ = write!(
        out,
        "<td class=\"num bar-cell\" data-value=\"{}\">\
         <span class=\"bar\" style=\"--fill:{:.4}\"></span>\
         <span class=\"value\">{}</span><span class=\"count\">{}</span></td>",
        category.mention_count,
        share,
        app.rate(category.mention_count)
            .map_or_else(|| nothing("no reviews"), percent),
        thousands(category.mention_count)
    );
    rate_cell(out, app.rate(category.primary_count));
    rate_cell(out, app.top_rate(category.top_mention_count));
    bias_cell(out, app.bias(category));
    verdict_cell(out, category.positive_share(), app.positive_baseline());
    out.push_str("</tr>\n");

    if has_examples {
        let _ = writeln!(out, "<tr class=\"panel\" id=\"{panel}\"><td colspan=\"6\">");
        if let Some(measured) = measured {
            how_well_this_row_is_known(out, measured);
        }
        sparkline(out, app, &category.id);
        reviews(out, app, examples);
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

/// A mark against a rate the classifier is measured to miss most of.
///
/// Only on the rows that earn it. Decorating every row with its own score would make the
/// table harder to read and the warning worth less exactly where it matters.
fn thinly_measured(measured: &crate::evaluate::CategoryAgreement) -> String {
    if measured.reference_mentions < ENOUGH_TO_JUDGE_A_ROW {
        return String::new();
    }
    let Some(recall) = measured.recall() else {
        return String::new();
    };
    if recall >= THINLY_FOUND {
        return String::new();
    }
    format!(
        "<span class=\"thin\" title=\"Found in only {} of the reviews measured to raise it, \
         so this rate is a floor rather than a count\">\u{2757}\
         <span class=\"read-aloud\">measured to miss most of this category</span></span>",
        percent(recall)
    )
}

/// What the reference set says about this row alone.
///
/// The page carries one figure for how often the classifier agrees overall, and that figure
/// is no guide at all to a particular row: the same run finds nine mentions in ten of one
/// category and one in twenty of another.
fn how_well_this_row_is_known(out: &mut String, measured: &crate::evaluate::CategoryAgreement) {
    if measured.reference_mentions == 0 {
        let _ = writeln!(
            out,
            "<p class=\"note\">No labelled review raises this category, so nothing here is \
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
    let class = if measured.reference_mentions < ENOUGH_TO_JUDGE_A_ROW {
        "note"
    } else if measured.recall().is_some_and(|r| r < THINLY_FOUND) {
        "note warn"
    } else {
        "note"
    };
    let _ = writeln!(
        out,
        "<p class=\"{class}\">Measured on this row: of the {} labelled reviews that raise it, \
         the classifier found {found}; of the reviews it filed here, {right} were labelled \
         that way. A rate built on a category it misses is a floor, not a count.{}</p>",
        thousands(measured.reference_mentions),
        // Which category the misses went to is the difference between a row that is merely
        // hard and a row whose reviews are sitting under a neighbour's name.
        measured
            .mistaken_for()
            .map_or_else(String::new, |(label, count)| {
                format!(
                    " Where its main subject was read wrongly, it was most often read as \
                 <strong>{}</strong> ({count} of the labelled reviews).",
                    escape(label)
                )
            })
    );
}

/// One category's mention rate month by month.
///
/// The same argument as the volume chart, one level down: a category at 5% of a corpus may
/// have been 40% of one month and absent since, and only the shape says which.
fn sparkline(out: &mut String, app: &AppReport, id: &str) {
    let Some(slot) = CORE_SPINE.iter().position(|c| c.id == id) else {
        return;
    };
    let months = &app.classification.months;
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
    let points: Vec<String> = rates
        .iter()
        .enumerate()
        .filter_map(|(index, rate)| {
            let rate = (*rate)?;
            #[expect(
                clippy::cast_precision_loss,
                reason = "a corpus spans hundreds of months at most"
            )]
            let x = index as f64 * step;
            Some(format!("{x:.2},{:.2}", (1.0 - rate / peak) * SPARK_HEIGHT))
        })
        .collect();
    if points.len() < 3 {
        return;
    }

    let _ = writeln!(
        out,
        "<figure class=\"spark\">\n<svg viewBox=\"0 0 {WIDTH:.0} {SPARK_HEIGHT:.0}\" \
         preserveAspectRatio=\"none\" role=\"img\" aria-label=\"Mention rate by month\">\
         <polyline points=\"{}\" /></svg>\n\
         <figcaption>Mention rate by month, {} to {}, peaking at {}. Months with fewer than \
         {ENOUGH_FOR_A_RATE} reviews are left out.</figcaption>\n</figure>",
        points.join(" "),
        escape(&crate::time::month_name(&months[0].label)),
        escape(&crate::time::month_name(&months[months.len() - 1].label)),
        percent(peak)
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
fn verdict_cell(out: &mut String, share: Option<f64>, baseline: Option<f64>) {
    let Some(share) = share else {
        let _ = write!(
            out,
            "<td class=\"num\" data-value=\"-1\">{}</td>",
            nothing("none raised it")
        );
        return;
    };
    let tone = match baseline {
        Some(baseline) if share > baseline + 0.05 => " warmer",
        Some(baseline) if share < baseline - 0.05 => " colder",
        _ => "",
    };
    let _ = write!(
        out,
        "<td class=\"num verdict-share{tone}\" data-value=\"{share:.9}\">{}</td>",
        percent(share)
    );
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
        thousands(app.classification.top_helpful)
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
    out.push_str("</div>\n");

    let text = example.review.text.trim();
    let long = text.chars().count() > PREVIEW_CHARS;
    let _ = write!(
        out,
        "<div class=\"text{}\"><p{}>{}</p></div>",
        if long { " long" } else { "" },
        // The page is in English and most of the reviews on it are not. Saying so is what
        // lets a screen reader pronounce a Chinese review as Chinese rather than as English.
        bcp47(&example.review.language).map_or_else(String::new, |tag| format!(" lang=\"{tag}\"")),
        escape(text)
    );
    if long {
        out.push_str(
            "<button class=\"more\" type=\"button\" data-expands-text>Show the rest</button>\n",
        );
    }

    out.push_str("<div class=\"filed\">");
    for id in &example.mentions {
        let label = CORE_SPINE
            .iter()
            .find(|c| c.id == *id)
            .map_or(id.as_str(), |c| c.label);
        let primary = if *id == example.primary { " main" } else { "" };
        let _ = write!(
            out,
            "<span class=\"chip{primary}\">{}</span>",
            escape(label)
        );
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

/// Where the categories came from, and in particular whether this game was one of them.
///
/// Anchors fitted on other games and never on this one still work, and how well is the one
/// thing `census fit --leave-one-out` measures. A reader is entitled to know which of the two
/// they are looking at without going and reading a file.
fn built_from(app: &AppReport) -> String {
    let games = app.classification.anchors_fitted_from.len();
    if games == 0 {
        return "written category descriptions, fitted to no labels".to_owned();
    }
    let of_them = if games == 1 {
        "one game".to_owned()
    } else {
        format!("{games} games")
    };
    if app
        .classification
        .anchors_fitted_from
        .contains(&app.app_id())
    {
        format!("labelled reviews from {of_them}, this one among them")
    } else {
        format!("labelled reviews from {of_them}, none of them this one")
    }
}

/// What the reader is entitled to conclude, next to the numbers rather than in a footnote.
/// What language the corpus is in, which Steam's own page cannot show a reader at all.
fn languages(out: &mut String, app: &AppReport) {
    if app.classification.languages.is_empty() {
        return;
    }
    let total: u64 = app.classification.languages.iter().map(|(_, n)| n).sum();
    let english = app
        .classification
        .languages
        .iter()
        .find(|(name, _)| name == "english")
        .map_or(0, |(_, count)| *count);

    #[expect(
        clippy::cast_precision_loss,
        reason = "review counts are far below 2^53"
    )]
    let not_english = if total > 0 {
        (total - english) as f64 / total as f64
    } else {
        0.0
    };

    out.push_str("<h3>What language it was said in</h3>\n");
    let _ = writeln!(
        out,
        "<p class=\"note\">{} of these reviews are not in English. Steam shows a reader their \
         own language by default, so most of this argument is one they never see.</p>",
        percent(not_english)
    );

    let widest = app
        .classification
        .languages
        .first()
        .map_or(1, |(_, count)| *count)
        .max(1);
    out.push_str("<ul class=\"languages\">\n");
    for (name, count) in app.classification.languages.iter().take(LANGUAGES_SHOWN) {
        #[expect(
            clippy::cast_precision_loss,
            reason = "review counts are far below 2^53"
        )]
        let fill = *count as f64 / widest as f64;
        let _ = writeln!(
            out,
            "<li><span class=\"lang-name\">{}</span>\
             <span class=\"bar\" style=\"--fill:{fill:.4}\"></span>\
             <span class=\"lang-count\">{}</span></li>",
            escape(&language_name(name)),
            thousands(*count)
        );
    }
    out.push_str("</ul>\n");

    let rest = app
        .classification
        .languages
        .len()
        .saturating_sub(LANGUAGES_SHOWN);
    if rest > 0 {
        let _ = writeln!(
            out,
            "<p class=\"note\">and {rest} more languages with fewer reviews each.</p>"
        );
    }
}

fn trust(out: &mut String, app: &AppReport) {
    out.push_str("<h3>How far to trust this</h3>\n");
    out.push_str("<dl class=\"facts wide\">\n");

    fact(out, "Categories built from", &built_from(app));
    fact(out, "Encoder", &app.classification.model);
    fact(out, "Taxonomy", &app.classification.spine_version);
    if app.classification.unmatched > 0 {
        // Not the blank reviews: those never enter a count at all. These are reviews whose
        // text has no vector, which means the corpus was embedded before they arrived.
        fact(
            out,
            "Reviews with no vector",
            &format!(
                "{} (embed again to include them)",
                thousands(app.classification.unmatched)
            ),
        );
    }
    // The capture holds more rows than any rate is taken over, and a reader who subtracts
    // the two deserves the difference named rather than left to guess at it.
    let blank = app
        .crawl
        .rows_unique
        .saturating_sub(app.classification.reviews)
        .saturating_sub(app.classification.unmatched);
    if blank > 0 {
        fact(
            out,
            "Reviews with no text",
            &format!(
                "{} (a rating and nothing else, counted in no rate)",
                thousands(blank)
            ),
        );
    }
    out.push_str("</dl>\n");

    match app.agreement.as_ref() {
        Some(agreement) => agreement_note(out, agreement),
        None => out.push_str(
            "<p class=\"warn\">No reference set has been labelled for this game, so how often \
             the classifier is wrong here has not been measured. Treat every rate as \
             provisional.</p>\n",
        ),
    }

    out.push_str(
        "<p class=\"note\">These rates are a census, not a survey: every review Valve serves \
         was counted, so there is no sampling error to report. What they do carry is \
         classifier error, which is what the agreement figure above measures.</p>\n",
    );
}

fn agreement_note(out: &mut String, agreement: &crate::AgreementReport) {
    let random = agreement
        .slices
        .iter()
        .find(|slice| slice.subset == "random" && slice.contested.is_none());
    let Some(slice) = random else {
        return;
    };
    let (Some(rate), Some((low, high))) = (slice.agreement(), slice.interval()) else {
        return;
    };
    let _ = writeln!(
        out,
        "<p class=\"warn\">On {} held-back reviews this classifier put {} in the same \
         category a separate labeller did, somewhere in [{}, {}] with 95% confidence. The \
         labeller was itself a language model, so that is <strong>agreement, not \
         accuracy</strong>: two models can be wrong together, most easily on sarcasm and on \
         reviews that sit between categories.</p>",
        thousands(slice.compared),
        percent(rate),
        percent(low),
        percent(high)
    );

    // One figure for a whole taxonomy hides the shape of the error: the same run finds nine
    // mentions in ten of one category and one in twenty of another.
    let judged: Vec<&crate::evaluate::CategoryAgreement> = agreement
        .categories
        .iter()
        .filter(|c| c.reference_mentions >= ENOUGH_TO_JUDGE_A_ROW)
        .collect();
    let thin = judged
        .iter()
        .filter(|c| c.recall().is_some_and(|recall| recall < THINLY_FOUND))
        .count();
    if judged.is_empty() {
        return;
    }
    let _ = writeln!(
        out,
        "<p class=\"note\">Averaged over the categories rather than over the reviews, so a \
         rare one counts as much as a common one, that comes to <strong>{:.2}</strong> on a \
         scale where 1 is perfect agreement. {}</p>",
        agreement.macro_f1().unwrap_or(0.0),
        if thin == 0 {
            format!(
                "Every one of the {} categories with enough labels to judge is found in at \
                 least a quarter of the reviews raising it.",
                judged.len()
            )
        } else if thin == 1 {
            format!(
                "One of the {} categories with enough labels to judge is found in fewer than \
                 a quarter of the reviews raising it, and its row is marked: read that rate \
                 as a floor.",
                judged.len()
            )
        } else {
            format!(
                "{thin} of the {} categories with enough labels to judge are found in fewer \
                 than a quarter of the reviews raising them, and their rows are marked: read \
                 those rates as floors.",
                judged.len()
            )
        }
    );
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
const LANGUAGES: [(&str, &str, &str); 29] = [
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
];

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

/// A share of Valve's own total, which is the one figure a reader is entitled to read as a
/// claim of completeness.
///
/// Rounding is a courtesy everywhere else on the page and a lie here: a capture that reached
/// all but a hundred of a million reviews is not complete, and printing 100.0% says it is.
fn coverage(share: f64) -> String {
    if share < 1.0 && (share * 100.0).round() >= 100.0 {
        return ">99.9%".to_owned();
    }
    percent(share)
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
            |id: &str, label: &str, mentions: u64, top: u64| crate::report::CategoryCount {
                id: id.to_owned(),
                label: label.to_owned(),
                primary_count: mentions / 2,
                mention_count: mentions,
                top_mention_count: top,
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
            primary: "bugs".to_owned(),
            mentions: vec!["bugs".to_owned(), "performance".to_owned()],
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
                classification: crate::report::Classification {
                    app_id: 7,
                    reviews: 1_000,
                    positive: 700,
                    unmatched: 3,
                    top_helpful: 50,
                    mention_margin: 0.01,
                    spine_version: "core-3".to_owned(),
                    model: "test-encoder".to_owned(),
                    anchors_fitted_from: vec![7],
                    categories: vec![
                        category("bugs", "Bugs and crashes", 400, 30),
                        category("performance", "Performance", 100, 2),
                    ],
                    languages: vec![("english".to_owned(), 600), ("schinese".to_owned(), 400)],
                    months: vec![
                        crate::report::Month {
                            label: "2024-01".to_owned(),
                            reviews: 400,
                            positive: 320,
                            categories: vec![40, 100],
                        },
                        crate::report::Month {
                            label: "2024-02".to_owned(),
                            reviews: 600,
                            positive: 380,
                            categories: vec![60, 300],
                        },
                    ],
                    top_reviews: Vec::new(),
                },
                examples: vec![("bugs".to_owned(), vec![example])],
                top: Vec::new(),
                agreement: None,
            }],
        }
    }

    /// The same report with a calendar of its own.
    fn with_months(months: Vec<crate::report::Month>) -> Report {
        let mut report = sample_report("ordinary text");
        report.apps[0].classification.months = months;
        report
    }

    fn month(label: &str, reviews: u64, bugs: u64) -> crate::report::Month {
        crate::report::Month {
            label: label.to_owned(),
            reviews,
            positive: reviews / 2,
            categories: vec![0, bugs],
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
        assert!(page.contains("Months with fewer than 30 reviews are left out"));
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
            page.contains("No reference set has been labelled"),
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

    /// A rate the classifier is measured to miss most of is not a count, and a reader
    /// scanning the table has no way to tell the two apart unless the page says so.
    #[test]
    fn a_row_the_classifier_barely_finds_is_marked_as_one() {
        let scored = |reference_mentions, mention_agreed| crate::evaluate::CategoryAgreement {
            id: "bugs",
            label: "Bugs and crashes",
            reference_primary: 0,
            predicted_primary: 0,
            primary_agreed: 0,
            reference_mentions,
            predicted_mentions: mention_agreed,
            mention_agreed,
            taken_as: Vec::new(),
        };

        assert!(
            thinly_measured(&scored(100, 4)).contains("class=\"thin\""),
            "four found in a hundred is a floor, not a count"
        );
        assert!(
            thinly_measured(&scored(100, 90)).is_empty(),
            "a row the classifier finds should carry no warning"
        );
        assert!(
            thinly_measured(&scored(4, 0)).is_empty(),
            "four labelled reviews cannot condemn a row"
        );

        let mut out = String::new();
        how_well_this_row_is_known(&mut out, &scored(100, 4));
        assert!(out.contains("found 4.0%"), "{out}");
        assert!(out.contains("100 labelled reviews"), "{out}");
        assert!(
            out.contains("note warn"),
            "a row this thin should look thin: {out}"
        );

        let mut none = String::new();
        how_well_this_row_is_known(&mut none, &scored(0, 0));
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
        report.apps[0].classification.reviews = 1_000;
        report.apps[0].classification.unmatched = 4;

        let page = render(&report);
        assert!(
            page.contains("Reviews with no text"),
            "the blanks go unmentioned"
        );
        assert!(
            page.contains("6 (a rating and nothing else"),
            "wrong blank count"
        );
        assert!(page.contains("Reviews with no vector"));

        report.apps[0].crawl.rows_unique = 1_000;
        report.apps[0].classification.unmatched = 0;
        let tidy = render(&report);
        assert!(
            !tidy.contains("Reviews with no text"),
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

    fn two_games() -> Report {
        let mut report = sample_report("ordinary text");
        let mut second = report.apps[0].clone();
        second.crawl.app_id = 9;
        second.crawl.name = "Another Game".to_owned();
        second.classification.app_id = 9;
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

    /// The one number on the page a reader may read as a claim of completeness, so it is
    /// the one number that must never be rounded into one.
    #[test]
    fn a_capture_that_missed_something_never_claims_all_of_it() {
        assert_eq!(coverage(1.0), "100.0%");
        assert_eq!(
            coverage(0.999_9),
            ">99.9%",
            "a hundred reviews short of a million"
        );
        assert_eq!(coverage(0.999_949), ">99.9%");
        assert_eq!(
            coverage(0.994),
            "99.4%",
            "an honest figure needs no hedging"
        );
        assert_eq!(coverage(0.5), "50.0%");
    }

    #[test]
    fn a_cell_with_no_number_says_so_out_loud() {
        let mut out = String::new();
        rate_cell(&mut out, None);
        bias_cell(&mut out, None);
        verdict_cell(&mut out, None, Some(0.8));

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
        let media = declared_in("@media (prefers-color-scheme: dark)");
        let attribute = declared_in(":root[data-theme='dark']");

        // Set on individual elements by the markup rather than by the theme.
        let inline = ["fill", "heat", "offset"];
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
