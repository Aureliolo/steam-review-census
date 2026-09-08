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
    out.push_str("</tbody>\n</table>\n</div>\n</div>\n</section>\n");
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
    fact(out, "Captured", &date(crawl.snapshot_unix));
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
    out.push_str("<th scope=\"col\">Category</th>");
    out.push_str("<th scope=\"col\" class=\"num\">Mention rate</th>");
    out.push_str("<th scope=\"col\" class=\"num\">Main subject</th>");
    out.push_str("<th scope=\"col\" class=\"num\">Top of the pile</th>");
    out.push_str("<th scope=\"col\" class=\"num\">Bias</th>");
    out.push_str("<th scope=\"col\" class=\"num\">Recommended</th>");
    out.push_str("</tr></thead>\n<tbody>\n");

    for category in rows {
        category_row(out, app, category, widest);
    }
    out.push_str("</tbody>\n</table>\n</div>\n");
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
        "<td class=\"num bar-cell\"><span class=\"bar\" style=\"--fill:{:.4}\"></span>\
         <span class=\"value\">{}</span><span class=\"count\">{}</span></td>",
        share,
        app.rate(category.mention_count)
            .map_or("—".to_owned(), percent),
        thousands(category.mention_count)
    );
    let _ = write!(
        out,
        "<td class=\"num\">{}</td>",
        app.rate(category.primary_count)
            .map_or("—".to_owned(), percent)
    );
    let _ = write!(
        out,
        "<td class=\"num\">{}</td>",
        app.top_rate(category.top_mention_count)
            .map_or("—".to_owned(), percent)
    );
    bias_cell(out, app.bias(category));
    verdict_cell(out, category.positive_share(), app.positive_baseline());
    out.push_str("</tr>\n");

    if has_examples {
        let _ = writeln!(out, "<tr class=\"panel\" id=\"{panel}\"><td colspan=\"6\">");
        reviews(out, app, examples);
        out.push_str("</td></tr>");
    }
}

/// Bias is the point, so it gets a bar that leaves the middle rather than a bare number.
fn bias_cell(out: &mut String, factor: Option<f64>) {
    let Some(factor) = factor else {
        out.push_str("<td class=\"num\">—</td>");
        return;
    };
    // Log scale: twice as often and half as often are the same distance from the middle,
    // which a linear scale would draw as wildly different.
    let offset = (factor.log2() / 3.0).clamp(-1.0, 1.0);
    let side = if offset >= 0.0 { "over" } else { "under" };
    let _ = write!(
        out,
        "<td class=\"num bias\"><span class=\"gauge {side}\" style=\"--offset:{:.4}\"></span>\
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
        out.push_str("<td class=\"num\">\u{2014}</td>");
        return;
    };
    let tone = match baseline {
        Some(baseline) if share > baseline + 0.05 => " warmer",
        Some(baseline) if share < baseline - 0.05 => " colder",
        _ => "",
    };
    let _ = write!(
        out,
        "<td class=\"num verdict-share{tone}\">{}</td>",
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
        "<p class=\"note\">The {} reviews Steam ranks as most helpful, which is roughly what \
         a reader sees before deciding. Every rate above is measured against the whole corpus \
         instead.</p>",
        thousands(app.classification.top_helpful)
    );
    reviews(out, app, &app.top);
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

    let anchors = if app.classification.anchors_fitted_from.is_empty() {
        "written category descriptions, fitted to no labels".to_owned()
    } else {
        format!(
            "fitted on labelled reviews from {}",
            app.classification
                .anchors_fitted_from
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    fact(out, "Categories built from", &anchors);
    fact(out, "Encoder", &app.classification.model);
    fact(out, "Taxonomy", &app.classification.spine_version);
    if app.classification.unmatched > 0 {
        fact(
            out,
            "Reviews with no text",
            &format!(
                "{} (counted in no rate)",
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

/// A date without pulling in a calendar crate, which for a UTC day is arithmetic.
fn date(unix: i64) -> String {
    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    if unix <= 0 {
        return "unknown".to_owned();
    }
    let (year, month, day) = civil_from_days(unix.div_euclid(86_400));
    let name = MONTHS
        .get(usize::from(month).saturating_sub(1))
        .copied()
        .unwrap_or("?");
    format!("{day} {name} {year}")
}

/// Howard Hinnant's days-from-civil, inverted. Exact for every date this tool can hold.
fn civil_from_days(days: i64) -> (i64, u8, u8) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (
        if m <= 2 { y + 1 } else { y },
        u8::try_from(m).unwrap_or(1),
        u8::try_from(d).unwrap_or(1),
    )
}

const STYLE: &str = include_str!("report.css");
const SCRIPT: &str = include_str!("report.js");

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn dates_match_the_calendar() {
        assert_eq!(date(0), "unknown");
        assert_eq!(date(1_700_000_000), "14 November 2023");
        // A leap day, which off-by-one arithmetic gets wrong.
        assert_eq!(date(1_709_164_800), "29 February 2024");
    }
}
