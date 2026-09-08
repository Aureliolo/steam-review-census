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

    page_header(&mut out, report);
    out.push_str("<main>\n");
    contents(&mut out, report);
    overview(&mut out, report);
    for app in &report.apps {
        game(&mut out, app);
    }
    out.push_str("</main>\n");
    page_footer(&mut out);
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
    out.push_str("</div>\n</header>\n");
}

/// A way to reach each game once there is more than one to reach.
fn contents(out: &mut String, report: &Report) {
    if report.apps.len() < 2 {
        return;
    }
    out.push_str("<nav class=\"contents\" aria-label=\"Games in this report\">\n<div class=\"wrap\">\n<ul>\n");
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
         a higher rate; the rates are not comparable to any other corpus.</p>",
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
            "<th scope=\"col\" class=\"num\">{}</th>",
            escape(&app.crawl.title())
        );
    }
    out.push_str("</tr></thead>\n<tbody>\n");

    for (id, label, pooled) in order {
        if pooled == 0 {
            continue;
        }
        let _ = write!(out, "<tr><th scope=\"row\">{}</th>", escape(label));
        for app in &report.apps {
            let rate = app
                .classification
                .categories
                .iter()
                .find(|c| c.id == id)
                .and_then(|c| app.rate(c.mention_count));
            match rate {
                Some(rate) => {
                    // Square-rooted so the common categories do not wash every other cell
                    // out; the number is always there for anyone reading exactly.
                    let heat = (rate * 2.0).sqrt().min(1.0);
                    let _ = write!(
                        out,
                        "<td class=\"num heat\" style=\"--heat:{heat:.3}\">{}</td>",
                        percent(rate)
                    );
                }
                None => out.push_str("<td class=\"num\">\u{2014}</td>"),
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

fn game(out: &mut String, app: &AppReport) {
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
            percent(crawl.coverage)
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

    out.push_str("<p class=\"headline\">\n");
    let _ = write!(
        out,
        "Reading only the {} most-upvoted reviews, {} of them talk about \
         <strong>{}</strong>. Across all {} reviews, {} do: the top of the pile overstates it \
         by <strong>{factor:.1}\u{d7}</strong>.",
        thousands(app.classification.top_helpful),
        percent(top),
        escape(&category.label.to_lowercase()),
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
         that recommended the game, from none at the bottom to all at the top.</p>",
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
            "<rect class=\"bar\" x=\"{x:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{tall:.2}\">\
             <title>{}: {} reviews</title></rect>",
            HEIGHT - tall,
            (step * 0.82).max(0.5),
            escape(&crate::time::month_name(&month.label)),
            thousands(month.reviews)
        );
    }

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

    let _ = write!(out, "<tr class=\"row\"");
    if has_examples {
        let _ = write!(
            out,
            " tabindex=\"0\" role=\"button\" aria-controls=\"{panel}\" \
             data-expands=\"{panel}\""
        );
    }
    out.push('>');

    let _ = write!(
        out,
        "<th scope=\"row\"><span class=\"name\">{}{}</span></th>",
        escape(&category.label),
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
            .map_or("—".to_owned(), percent),
        thousands(category.mention_count)
    );
    rate_cell(out, app.rate(category.primary_count));
    rate_cell(out, app.top_rate(category.top_mention_count));
    bias_cell(out, app.bias(category));
    verdict_cell(out, category.positive_share(), app.positive_baseline());
    out.push_str("</tr>\n");

    if has_examples {
        let _ = writeln!(out, "<tr class=\"panel\" id=\"{panel}\"><td colspan=\"6\">");
        sparkline(out, app, &category.id);
        reviews(out, app, examples);
        out.push_str("</td></tr>");
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
        None => out.push_str("<td class=\"num\" data-value=\"-1\">—</td>"),
    }
}

/// Bias is the point, so it gets a bar that leaves the middle rather than a bare number.
fn bias_cell(out: &mut String, factor: Option<f64>) {
    let Some(factor) = factor else {
        out.push_str("<td class=\"num\" data-value=\"-1\">—</td>");
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
        out.push_str("<td class=\"num\" data-value=\"-1\">\u{2014}</td>");
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
    let _ = writeln!(
        out,
        "<details class=\"pile\">\n<summary>Read all {} of them</summary>",
        thousands(app.classification.top_helpful)
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
        "<div class=\"text{}\"><p>{}</p></div>",
        if long { " long" } else { "" },
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
            escape(name),
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
}

fn page_footer(out: &mut String) {
    out.push_str("<footer class=\"page\">\n<div class=\"wrap\">\n");
    out.push_str("<h3>What this cannot tell you</h3>\n<ul>\n");
    for limit in FOOTER_LIMITS {
        let _ = writeln!(out, "<li>{limit}</li>");
    }
    out.push_str("</ul>\n");
    out.push_str(
        "<p class=\"note\">Built by steam-review-census. Nothing on this page was sent \
         anywhere to produce it.</p>\n",
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
