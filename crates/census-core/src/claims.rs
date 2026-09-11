//! Splitting a review into the separate points it makes.
//!
//! A review is not one opinion. "Looks incredible, runs like a slideshow, and the story is
//! the best in the series" is three, about three different things, and a single vector for
//! the whole review is their average: a point in embedding space that belongs to none of
//! them. Every category a review touches has to be reachable, and averaging is what puts a
//! long multi-topic review nearest to nothing in particular.
//!
//! The split is deliberately mechanical. It knows about sentence terminators in the scripts
//! reviews are written in and nothing else: no grammar, no model, no language detection. A
//! wrong split costs one claim landing in the wrong category, which is measurable. A model
//! here would cost a second thing to keep honest.

use std::{collections::HashSet, path::Path, sync::Arc};

use arrow::{
    array::{ArrayRef, StringBuilder, UInt16Builder, UInt32Builder},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use parquet::{
    arrow::ArrowWriter,
    basic::{Compression, ZstdLevel},
    file::properties::WriterProperties,
};

use crate::Result;

/// Which set of splitting rules produced a set of claims.
///
/// Recorded with every drawn sample because it decides what a claim index refers to. Two sets
/// drawn under different versions are not the same claims, and a label from one applied to
/// the other is a label on whatever text happens to sit at that index now.
///
/// `claims-2` added the rules the first labellers asked for: bullet markers stripped rather
/// than kept, a heading ending in a colon joined to what it introduces, and no splitting
/// inside a quotation. `claims-3` added the rest of what they found: Steam's markup removed,
/// semicolons no longer ending a thought, numbered list markers, parenthetical asides, web
/// addresses and abbreviations. `claims-4` is what twenty-six labellers found after that: a
/// list of short comma-separated points is that many points, a question keeps its short
/// answer, a heading tag opened mid-sentence is emphasis rather than a heading, a line that
/// ends on a comma has not finished, and an emoticon belongs to the sentence before it.
/// `claims-5` is the copypasta: a ballot-box template is the boxes the reviewer ticked and
/// not the ones they left blank, and a drawing made of punctuation is one claim rather than
/// one per line.
pub const SPLITTER_VERSION: &str = "claims-5";

/// The marks a review template offers as options, ticked or left blank.
///
/// One review in seven hundred is one of these, and they hold 1.8% of every claim in the
/// reference set: a single Deep Rock Galactic review came back as fifty-eight claims, of
/// which fifty-three were options its author never chose.
const BALLOT_BOXES: [char; 6] = [
    '\u{2610}', '\u{2611}', '\u{2612}', '\u{25A1}', '\u{2713}', '\u{2714}',
];

/// The marks that count as ticked. A reviewer who fills a template in with an "x" is
/// answering it as surely as one who has a font with a tick in it.
const TICKED: [char; 5] = ['\u{2611}', '\u{2612}', '\u{2713}', '\u{2714}', 'x'];

/// How many lines of punctuation in a row are a picture rather than a sentence. Two could be
/// a shrug and a face; three is somebody drawing.
const LINES_OF_A_DRAWING: usize = 3;

/// The most a comma-separated part may weigh for a sentence of three or more of them to be
/// read as a list of points. "Stunning visual, calm music, epic story" is three of weight
/// fifteen and under; "The combat, which took a while to click, is superb" has a middle
/// twice that and stays one thought.
const SHORT_PART: usize = 18;

/// The least a part of such a list may weigh. A part names a thing and says something about
/// it, which is two words; "Yes, yes, yes." is three words of the same one and no list.
const LEAST_PART: usize = 6;

/// The most a piece may weigh and still be an answer to the question before it rather than a
/// point of its own. "Top left of the screen." answers "Want to see your objectives?"; a
/// paragraph after a question is a paragraph.
const SHORT_ANSWER: usize = 30;

/// Below this much of a piece it is not a point, it is the tail of one. "Yes." and "10/10."
/// are joined to what they qualify rather than counted as opinions of their own.
const MIN_CLAIM_WEIGHT: usize = 12;

/// What one character of a script that writes without spaces is worth against one Latin
/// character. A Japanese sentence is eight characters where its English translation is
/// thirty, so counting characters alone would file every CJK review as fragments.
const DENSE_CHARACTER: usize = 3;

/// Where a sentence can end, across the scripts Steam reviews arrive in. Latin and Cyrillic
/// share the first three; the rest are the full-width forms used in Chinese, Japanese and
/// Korean, Arabic's full stop, and the Devanagari danda.
/// A semicolon is deliberately absent. "It's not just a game; it's an experience" is one
/// thought with a hinge in it, and cutting there leaves two halves that each say nothing.
const TERMINATORS: [char; 9] = [
    '.', '!', '?', '\u{2026}', '\u{3002}', '\u{FF01}', '\u{FF1F}', '\u{06D4}', '\u{0964}',
];

/// The points a review makes, in the order it makes them.
///
/// Never empty for text with a word in it: a review that terminates nothing comes back as one
/// claim, which is the honest reading of a reviewer who wrote one long sentence. Empty for a
/// review that is only a drawing, which is a review that says nothing rather than one whose
/// single claim the reader has to decline.
#[must_use]
pub fn split(text: &str) -> Vec<std::borrow::Cow<'_, str>> {
    claims_of(text)
        .into_iter()
        .map(|(_, claim)| claim)
        .collect()
}

/// The same split, as byte ranges into the review.
///
/// Offsets rather than text is what a published dataset can carry: the labels point at spans
/// of reviews anyone can fetch from Steam themselves, so the labelling is shareable without
/// redistributing a word anybody wrote.
#[must_use]
pub fn spans(text: &str) -> Vec<std::ops::Range<usize>> {
    claims_of(text).into_iter().map(|(at, _)| at).collect()
}

/// Every claim, as where it sits in the review and what it says once markup is taken out.
///
/// Both together because they are two views of one answer and computing them separately
/// invites them to disagree, which would put a label on the wrong span.
#[must_use]
pub fn claims_of(text: &str) -> Vec<(std::ops::Range<usize>, std::borrow::Cow<'_, str>)> {
    let mut pieces: Vec<(usize, usize)> = Vec::new();
    let mut start = 0;
    let mut quoted = false;
    let mut open_brackets = 0_i32;
    let mut heading = false;
    let mut chars = text.char_indices().peekable();

    while let Some((at, ch)) = chars.next() {
        if ch == '['
            && let Some((after, kind)) = markup_at(text, at)
        {
            while chars.peek().is_some_and(|(next, _)| *next < after) {
                chars.next();
            }
            let cut = match kind {
                Markup::Break => Some(at),
                Markup::Skip => None,
                Markup::Heading { closing: false } => {
                    heading = opens_a_heading(text, start, at);
                    heading.then_some(at)
                }
                Markup::Heading { closing: true } => std::mem::take(&mut heading).then_some(after),
            };
            if let Some(cut) = cut
                && !says_nothing(&text[start..cut])
            {
                pieces.push((start, cut));
                start = cut;
            }
            continue;
        }
        if let Some(open) = quote_state(ch, quoted) {
            quoted = open;
        }
        match ch {
            '(' | '\u{FF08}' => open_brackets += 1,
            ')' | '\u{FF09}' => open_brackets = (open_brackets - 1).max(0),
            _ => {}
        }
        // A full stop inside a quotation or an aside ends that, not the reviewer's sentence.
        // "Так денег никто не даст. Давай по-новой" is one joke being retold, and cutting it
        // in half leaves two fragments that mean nothing apart.
        let ends = if ch == '\n' {
            // A line that ends on a comma was wrapped by hand mid-sentence, which is how a
            // reviewer who never types a full stop writes a long one.
            !ends_on_a_comma(&text[start..at])
        } else if quoted || open_brackets > 0 || inside_a_link(text, at) {
            false
        } else if ch == '.' {
            // The only ambiguous terminator. Every other one in TERMINATORS ends a thought
            // wherever it appears, including the full-width stops, which sit between
            // characters with no space anywhere near them.
            !numbers_a_list(&text[start..at])
                && !continues_a_number(text, at)
                && !abbreviates(text, at)
                && !continues_a_word(text, at + ch.len_utf8())
        } else {
            TERMINATORS.contains(&ch)
        };

        if !ends {
            continue;
        }

        // A run with no words in it, a blank line or a bare `[list]`, is not a piece; it
        // stays in front of the next one, which is where its markup belongs.
        let end = run_out(text, &mut chars, at + ch.len_utf8());
        if !says_nothing(&text[start..end]) {
            pieces.push((start, end));
            start = end;
        }
    }

    if !says_nothing(&text[start..]) {
        pieces.push((start, text.len()));
    }

    // A template has already decided what its claims are, and they are short on purpose: an
    // answer to a heading is "Beautiful", which the fragment joiner would glue to the next
    // answer and the list splitter would cut again.
    let filled_in = a_template_or_a_drawing(text);
    let pieces = filled_in.clone().unwrap_or_else(|| {
        join_the_fragments(text, pieces)
            .into_iter()
            .flat_map(|piece| listed_points(text, piece))
            .collect()
    });

    pieces
        .into_iter()
        .filter_map(|(from, to)| tidied(text, from, to))
        .filter_map(|at| {
            let piece = &text[at.clone()];
            let cleaned = without_markup(piece);
            // The span runs from the heading to the answer and passes over the options in
            // between, which are lines the reviewer declined. They are inside it because a
            // span is one range of bytes; they are not inside what anybody said.
            let cleaned = if filled_in.is_some() {
                without_the_unchosen(&cleaned)
            } else {
                cleaned
            };
            let cleaned = cleaned.trim();
            if cleaned.is_empty() {
                return None;
            }
            let claim = if cleaned.len() == piece.len() {
                std::borrow::Cow::Borrowed(piece)
            } else {
                std::borrow::Cow::Owned(cleaned.to_owned())
            };
            Some((at, claim))
        })
        .collect()
}

/// Drops the lines of a template the reviewer left blank, and the rules between sections.
fn without_the_unchosen(piece: &str) -> String {
    piece
        .lines()
        .filter(|line| {
            let bare = line.trim();
            !bare.is_empty() && option_mark(bare) != Some(false) && !is_a_drawn_line(line)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Cuts a review that is a filled-in template, or a picture, into what it actually says.
///
/// Steam reviews are full of a copypasta: a heading, then a column of options with one
/// ticked. Split by sentence it comes back as dozens of claims, nearly all of them options
/// the reviewer passed over, and four labellers in a row flagged every fragment. The ticked
/// boxes are the review; the blank ones are the ones somebody else would have ticked.
///
/// A drawing goes the same way: fifteen lines of braille or a hand made of brackets, cut into
/// a claim per line, none of which is words. Recognised by what a line is made of rather than
/// what it says, so it holds in every script.
///
/// `None` where the review is neither, which is all but one in seven hundred.
fn a_template_or_a_drawing(text: &str) -> Option<Vec<(usize, usize)>> {
    let lines: Vec<(usize, &str)> = text
        .split_inclusive('\n')
        .scan(0, |at, line| {
            let start = *at;
            *at += line.len();
            Some((start, line))
        })
        .collect();

    let boxed = lines
        .iter()
        .filter(|(_, line)| line.trim_start().starts_with(BALLOT_BOXES))
        .count();
    let drawn = longest_run(&lines, is_a_drawn_line);
    if boxed < LINES_OF_A_DRAWING && drawn < LINES_OF_A_DRAWING {
        return None;
    }

    // A heading above the options is what the ticked one is an answer to, so it stays in
    // front of it: "{ Graphics }" and "Beautiful" are one claim about graphics.
    let mut pieces: Vec<(usize, usize)> = Vec::new();
    let mut held: Option<usize> = None;
    for (at, line) in &lines {
        let end = at + line.len();
        let bare = line.trim();
        match option_mark(bare) {
            // An option nobody chose. Not a claim, and not a heading for the next one.
            Some(false) => {}
            Some(true) => {
                let from = held.take().unwrap_or(*at);
                pieces.push((from, end));
            }
            None => {
                if bare.is_empty() || is_a_drawn_line(line) {
                    continue;
                }
                // Anything that is not an option is a heading for the options under it,
                // unless nothing follows, in which case it is a claim of its own.
                held = Some(held.unwrap_or(*at));
            }
        }
    }
    if let Some(from) = held {
        pieces.push((from, text.len()));
    }
    // A review that is nothing but a drawing is one claim, which the reader declines. Falling
    // back to the sentence split here would hand back the picture a line at a time, which is
    // the shape this exists to stop.
    Some(if pieces.is_empty() {
        vec![(0, text.len())]
    } else {
        pieces
    })
}

/// Whether a line is one of a template's options, and whether the reviewer chose it.
///
/// A box is unambiguous in either state. A bare "x" is how somebody without a font full of
/// ticks answers the same template, and it counts only where a space follows it, so "x2
/// speed is great" stays a sentence.
fn option_mark(bare: &str) -> Option<bool> {
    let mut chars = bare.chars();
    let first = chars.next()?;
    if BALLOT_BOXES.contains(&first) {
        return Some(TICKED.contains(&first));
    }
    let crossed = first == 'x' || first == 'X';
    (crossed && chars.next().is_some_and(char::is_whitespace)).then_some(true)
}

/// Whether a line is part of a picture: it has something on it, and none of it is a letter
/// or a digit in any script.
fn is_a_drawn_line(line: &str) -> bool {
    let bare = line.trim();
    !bare.is_empty() && bare.chars().count() > 1 && !bare.chars().any(char::is_alphanumeric)
}

/// The longest run of consecutive lines the test holds for.
fn longest_run(lines: &[(usize, &str)], test: impl Fn(&str) -> bool) -> usize {
    let (mut longest, mut running) = (0, 0);
    for (_, line) in lines {
        running = if test(line) { running + 1 } else { 0 };
        longest = longest.max(running);
    }
    longest
}

/// Runs a boundary that begins at `end` out over any further terminators and the whitespace
/// after them, so "Wait... what?!" is one boundary rather than five, and over an emoticon
/// after those, which colours the sentence it follows: "Great fun. :D If you like Vermintide"
/// smiles about the fun. Not past a line break, where a lone dash is the next line's bullet.
fn run_out(
    text: &str,
    chars: &mut std::iter::Peekable<std::str::CharIndices<'_>>,
    mut end: usize,
) -> usize {
    let mut broke_the_line = false;
    loop {
        while let Some(&(next_at, next)) = chars.peek() {
            if next.is_whitespace() || TERMINATORS.contains(&next) {
                broke_the_line |= next == '\n';
                end = next_at + next.len_utf8();
                chars.next();
            } else {
                break;
            }
        }
        let token = text[end..].split(char::is_whitespace).next().unwrap_or("");
        if broke_the_line || !is_an_emoticon(token) {
            return end;
        }
        end += token.len();
        while chars.peek().is_some_and(|&(next_at, _)| next_at < end) {
            chars.next();
        }
    }
}

/// What a piece of Steam's markup does to the text around it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Markup {
    /// Starts a new point: a list item, a rule, a table cell.
    Break,
    /// Styling around text that carries on: bold, italic, spoiler, a link.
    Skip,
    /// A heading tag, which is a point of its own where it opens a line and emphasis where
    /// a reviewer opened it mid-sentence.
    Heading { closing: bool },
}

/// Whether a heading tag opened at `at` is a heading rather than emphasis.
///
/// On Steam a heading is a block, so a reviewer who opens one mid-sentence, after a word or
/// a comma, wanted bold text, and the sentence runs straight through it and through the
/// closing tag. Opened with nothing written since the line began or the last point ended,
/// or right after a sentence ended, it is the heading it looks like. Markup does not count
/// as written: a heading straight after `[/list]` is still the first thing on its line.
fn opens_a_heading(text: &str, start: usize, at: usize) -> bool {
    let line_start = text[..at].rfind('\n').map_or(0, |index| index + 1);
    let written = without_markup(&text[start.max(line_start)..at]);
    let written = written.trim_end();
    written.is_empty() || written.ends_with(TERMINATORS)
}

/// Tags that separate one point from the next. Everything else wraps a point without
/// interrupting it.
const BREAKING_TAGS: [&str; 9] = [
    "*", "hr", "list", "olist", "quote", "table", "tr", "td", "th",
];

/// Whether a piece ends on a comma, in any of the scripts that write one.
fn ends_on_a_comma(piece: &str) -> bool {
    piece.trim_end().ends_with([',', '\u{FF0C}', '\u{3001}'])
}

/// Cuts a sentence that is a list of short comma-separated points into those points.
///
/// "Stunning visual, calm music, epic story" is three subjects in one sentence, and a
/// labeller given it as one claim picks one and marks it contested. Three or more parts, each
/// short, is the shape of a list rather than of prose: a clause with a comma in it is longer
/// than any item anyone lists. Left alone where the piece holds a quotation or a bracket,
/// since a comma inside either is that thing's own.
///
/// Cut after the fragments are joined, not before, because the parts of a list are shorter
/// than a claim on purpose and joining them back together would undo the cut.
fn listed_points(text: &str, (from, to): (usize, usize)) -> Vec<(usize, usize)> {
    let piece = &text[from..to];
    if piece.contains([
        '"', '(', ')', '\u{201C}', '\u{201D}', '\u{FF08}', '\u{FF09}',
    ]) {
        return vec![(from, to)];
    }
    let mut parts: Vec<(usize, usize)> = Vec::new();
    let mut part_start = from;
    for (offset, ch) in piece.char_indices() {
        if matches!(ch, ',' | '\u{FF0C}' | '\u{3001}') {
            parts.push((part_start, from + offset));
            part_start = from + offset + ch.len_utf8();
        }
    }
    parts.push((part_start, to));

    let short = parts.len() >= 3
        && parts.iter().all(|&(start, end)| {
            (LEAST_PART..=SHORT_PART).contains(&weight(text[start..end].trim()))
        });
    if short { parts } else { vec![(from, to)] }
}

/// Recognises Steam's markup at `at`, returning where it ends and what it does.
///
/// Reviews are written with `BBCode` and the tags are not what anybody said. Left in, `[h3]`
/// and `[/list]` are tokens the model sees in every category and learns nothing from, and a
/// labeller was handed a heading tag and its closing tag as though they were a claim.
fn markup_at(text: &str, at: usize) -> Option<(usize, Markup)> {
    const LONGEST_TAG: usize = 200;

    let rest = &text[at + 1..];
    let end = rest
        .char_indices()
        .take(LONGEST_TAG)
        .find_map(|(offset, ch)| (ch == ']').then_some(offset))?;
    let inside = &rest[..end];
    if inside.is_empty() {
        return None;
    }

    let closing = inside.starts_with('/');
    let name = inside
        .split(['=', ' '])
        .next()
        .unwrap_or(inside)
        .trim_start_matches('/')
        .to_ascii_lowercase();
    // A tag name is letters, digits or the list bullet. Anything else is a bracket somebody
    // typed, and "[10/10]" is a claim rather than markup.
    if name.is_empty()
        || !name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '*')
    {
        return None;
    }

    let kind = if matches!(name.as_str(), "h1" | "h2" | "h3") {
        Markup::Heading { closing }
    } else if BREAKING_TAGS.contains(&name.as_str()) {
        Markup::Break
    } else {
        Markup::Skip
    };
    Some((at + 1 + end + 1, kind))
}

/// Whether a piece holds no words at all once its markup is taken out.
fn says_nothing(piece: &str) -> bool {
    without_markup(piece).trim().is_empty()
}

/// Removes Steam's markup from a claim, leaving what was written.
fn without_markup(piece: &str) -> String {
    let mut clean = String::with_capacity(piece.len());
    let mut at = 0;
    while at < piece.len() {
        let ch = piece[at..].chars().next().unwrap_or('\0');
        if ch == '['
            && let Some((after, _)) = markup_at(piece, at)
        {
            at = after;
            continue;
        }
        clean.push(ch);
        at += ch.len_utf8();
    }
    clean
}

/// Whether a terminator sits inside a web address, where "?" starts a query string and "."
/// separates a hostname rather than ending anything.
fn inside_a_link(text: &str, at: usize) -> bool {
    // Stepping over the whitespace by its own width rather than by one: reviews are full of
    // non-breaking spaces, and a byte past the start of one is not a character boundary.
    let token_start = text[..at]
        .char_indices()
        .rev()
        .find(|(_, ch)| ch.is_whitespace())
        .map_or(0, |(index, ch)| index + ch.len_utf8());
    let token = &text[token_start..at];
    token.contains("://") || token.contains("www.")
}

/// Whether a character opens or closes a quotation, given whether one is already open.
///
/// The straight double quote is both, so it toggles. The paired forms do not, which matters
/// for the languages that use them: a Chinese review full of 「」 would otherwise flip in and
/// out of quoted state on every mark.
fn quote_state(ch: char, quoted: bool) -> Option<bool> {
    match ch {
        '"' => Some(!quoted),
        '\u{201C}' | '\u{00AB}' | '\u{300C}' | '\u{300E}' => Some(true),
        '\u{201D}' | '\u{00BB}' | '\u{300D}' | '\u{300F}' => Some(false),
        _ => None,
    }
}

/// The span of a claim's words: a piece with the whitespace, the list marks and the markup
/// around its words taken off both ends.
///
/// Reviews are written with bullets, and "- 教学纯靠自己领悟" is a point about the tutorial
/// with a hyphen in front of it. Leaving the hyphen on gives the model a token that appears
/// in every category and means nothing in any of them. The markup goes for the same reason,
/// and so that a span names words: a label that points at `[*]` is a label on a tag.
///
/// Public because a label made against an older cut may carry the tag, and the join brings
/// it through here to meet the span this cut records.
#[must_use]
pub fn tidied(text: &str, from: usize, to: usize) -> Option<std::ops::Range<usize>> {
    // A comma in front of a point is the end of the point before it, left behind by a cut.
    const MARKERS: [char; 11] = [
        '-', '+', '*', '\u{2022}', '\u{00B7}', '\u{2013}', '\u{2014}', '>', ',', '\u{FF0C}',
        '\u{3001}',
    ];

    if !text.is_char_boundary(from) || !text.is_char_boundary(to) || from > to {
        return None;
    }

    let front = |mut start: usize, end: usize| loop {
        let trimmed = text[start..end].trim_start().trim_start_matches(MARKERS);
        let mut next = end - trimmed.len();
        if trimmed.starts_with('[')
            && let Some((after, _)) = markup_at(text, next)
        {
            next = after;
        }
        if next == start {
            return start;
        }
        start = next;
    };

    let mut start = front(from, to);

    // "1." and "2)" in front of a point are numbering, not what somebody said.
    let piece = &text[start..to];
    let digits = piece
        .char_indices()
        .take_while(|(_, ch)| ch.is_ascii_digit())
        .count();
    if digits > 0 && digits <= 3 {
        let after = &piece[digits..];
        if after.starts_with(['.', ')', ':']) && !after[1..].trim_start().is_empty() {
            start = front(start + digits + 1, to);
        }
    }

    let mut end = to;
    loop {
        let trimmed = text[start..end].trim_end();
        let stripped = trimmed.trim_end_matches(MARKERS).trim_end();
        // Only where something is left: a claim that is nothing but dashes is a divider,
        // and stripping it to nothing is the correct reading of one.
        let mut next = start
            + if stripped.is_empty() {
                trimmed.len()
            } else {
                stripped.len()
            };
        if let Some(open) = markup_ending_at(text, start, next) {
            next = open;
        }
        if next == end {
            break;
        }
        end = next;
    }

    (start < end).then_some(start..end)
}

/// Where the tag ending exactly at `end` opens, if the text there ends on one.
fn markup_ending_at(text: &str, from: usize, end: usize) -> Option<usize> {
    if !text[..end].ends_with(']') {
        return None;
    }
    let open = text[from..end].rfind('[')? + from;
    let (after, _) = markup_at(text, open)?;
    (after == end).then_some(open)
}

/// Whether everything before this full stop is just the number of a list item.
///
/// "1. The interface is unusable" is one point with a marker in front of it, and splitting at
/// the stop leaves "1." as a claim about nothing.
fn numbers_a_list(so_far: &str) -> bool {
    let trimmed = so_far.trim_start_matches(|ch: char| ch.is_whitespace() || ch == '(');
    !trimmed.is_empty() && trimmed.len() <= 3 && trimmed.chars().all(|ch| ch.is_ascii_digit())
}

/// Abbreviations that take a full stop without ending a sentence.
///
/// A list rather than a rule, because every rule general enough to catch "ca." also catches
/// "fun." A short list of the ones that actually appear in reviews costs nothing and is wrong
/// about nothing else. Reviews arrive in many languages, so this is not only English.
const ABBREVIATIONS: [&str; 42] = [
    "mr", "mrs", "ms", "dr", "prof", "vs", "etc", "eg", "ie", "approx", "max", "vol", "ch", "pp",
    "st", "inc", "ltd", "jr", "sr", "ca", "bzw", "evtl", "ggf", "usw", "zb", "dh", "uvm", "inkl",
    "bspw", "eig", "sog", "bzgl", "env", "ecc", "def", "ed", "ver", "vers", "esp", "resp", "orig",
    "hrs",
];

/// Whether the word before this stop is one of them, written with or without the stops
/// inside it: "i.e." and "z.B." are "ie" and "zb" with their dots taken out.
fn abbreviates(text: &str, at: usize) -> bool {
    let word_start = text[..at]
        .char_indices()
        .rev()
        .find(|(_, ch)| !ch.is_alphabetic() && *ch != '.')
        .map_or(0, |(index, ch)| index + ch.len_utf8());
    let word: String = text[word_start..at]
        .chars()
        .filter(|ch| *ch != '.')
        .map(|ch| ch.to_ascii_lowercase())
        .collect();
    ABBREVIATIONS.contains(&word.as_str())
}

/// Whether a full stop is a decimal point rather than the end of a thought, as in "9.5/10"
/// and "1.6 patch".
fn continues_a_number(text: &str, at: usize) -> bool {
    let before = text[..at].chars().next_back().is_some_and(char::is_numeric);
    let after = text[at + 1..].chars().next().is_some_and(char::is_numeric);
    before && after
}

/// Whether what follows a full stop reads as the middle of a sentence rather than the start
/// of one, which is what separates "e.g. this one" and "Mr. Freeman" from a real boundary.
///
/// Scripts without letter case, which is most of the ones this has to handle, have no
/// lowercase to find, so this only ever suppresses a split in the scripts that do.
fn continues_a_word(text: &str, from: usize) -> bool {
    let rest = text[from..].trim_start();
    if rest.len() == text[from..].len() && !rest.is_empty() {
        // No space at all after the stop: "www.example.com", "4.Great".
        return true;
    }
    rest.chars().next().is_some_and(char::is_lowercase)
}

/// How much a piece says, in Latin characters or their equivalent.
///
/// Markup weighs nothing: `[h1]Cons[/h1]` says exactly what "Cons" says, and both are a
/// heading that belongs to the point under it rather than a point of their own.
fn weight(piece: &str) -> usize {
    let mut total = 0;
    let mut at = 0;
    while at < piece.len() {
        let ch = piece[at..].chars().next().unwrap_or('\0');
        if ch == '['
            && let Some((after, _)) = markup_at(piece, at)
        {
            at = after;
            continue;
        }
        total += if writes_without_spaces(ch) {
            DENSE_CHARACTER
        } else {
            1
        };
        at += ch.len_utf8();
    }
    total
}

/// Whether a character belongs to a script that carries about a word per character and puts
/// no spaces between them.
pub(crate) fn writes_without_spaces(ch: char) -> bool {
    matches!(ch as u32,
        0x3040..=0x30FF   // Hiragana and Katakana
        | 0x3400..=0x4DBF // CJK unified ideographs, extension A
        | 0x4E00..=0x9FFF // CJK unified ideographs
        | 0xAC00..=0xD7AF // Hangul syllables
        | 0xF900..=0xFAFF // CJK compatibility ideographs
    )
}

/// Joins pieces too short to be a point of their own to the point they qualify.
///
/// Forward, because a short opener is nearly always a verdict on what follows ("Yes. Buy
/// it while it is on sale"), and a short piece with nothing after it has only one neighbour.
/// Backward in two shapes where the point is plainly the one before: a short answer after a
/// question ("Want to see your objectives?" "Top left of the screen."), and an emoticon,
/// which colours the sentence it follows and never the one it precedes.
fn join_the_fragments(text: &str, pieces: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    let mut joined: Vec<(usize, usize)> = Vec::with_capacity(pieces.len());
    let mut held: Option<(usize, usize)> = None;

    for (from, to) in pieces {
        let piece = text[from..to].trim();
        let answers = held.is_none()
            && joined
                .last()
                .is_some_and(|&(_, end)| asks_a_question(&text[..end]))
            && weight(piece) <= SHORT_ANSWER;
        if (answers || is_an_emoticon(piece))
            && let Some(last) = joined.last_mut()
            && held.is_none()
        {
            last.1 = to;
            continue;
        }

        let (from, to) = match held.take() {
            Some((earlier, _)) => (earlier, to),
            None => (from, to),
        };
        let piece = text[from..to].trim();
        // A piece ending in a colon introduces the next one rather than saying anything
        // itself. "Spoiler zur Spieldynamik:" on its own line is a heading, and on its own
        // it is a claim about nothing. Read without its markup, since a reviewer who bolds
        // a heading closes the tag after the colon.
        let written = without_markup(piece);
        let introduces = written.trim_end().ends_with([':', '\u{FF1A}']);
        if introduces || weight(piece) < MIN_CLAIM_WEIGHT {
            held = Some((from, to));
        } else {
            joined.push((from, to));
        }
    }

    if let Some((from, to)) = held {
        match joined.last_mut() {
            Some(last) => last.1 = to,
            None => joined.push((from, to)),
        }
    }

    joined
}

/// Whether text ends on a question mark, once any trailing whitespace is ignored.
fn asks_a_question(so_far: &str) -> bool {
    so_far.trim_end().ends_with(['?', '\u{FF1F}'])
}

/// Whether a token is an emoticon or a flourish rather than words: ":D", "<3", "^^", ":)".
/// One letter at most, because "D:" is an emoticon and "ok" is a verdict, and never a letter
/// alone, because "A" after a full stop is the next sentence starting. "xD" is the one
/// two-letter emoticon common enough to name.
fn is_an_emoticon(token: &str) -> bool {
    let length = token.chars().count();
    let letters = token.chars().filter(|ch| ch.is_alphanumeric()).count();
    (1..=4).contains(&length)
        && ((letters <= 1 && letters < length) || token.eq_ignore_ascii_case("xd"))
}

/// What splitting a corpus produced.
#[derive(Debug, Clone)]
pub struct ClaimReport {
    pub app_id: u32,
    pub reviews: u64,
    /// Reviews with nothing in them to split, which are stored but say nothing at all.
    pub empty: u64,
    pub claims: u64,
    /// Claims whose text nothing else in the corpus repeats. This is what has to be embedded,
    /// and on a corpus full of "Great game." it is far below the claim count.
    pub distinct: u64,
}

impl ClaimReport {
    /// Claims per review, which is how much a whole-review vector was averaging away.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "corpus counts are far below 2^53"
    )]
    pub fn per_review(&self) -> f64 {
        if self.reviews == 0 {
            return 0.0;
        }
        self.claims as f64 / self.reviews as f64
    }

    /// Share of claims that something else in the corpus says in the same words.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "corpus counts are far below 2^53"
    )]
    pub fn repeated(&self) -> Option<f64> {
        (self.claims > 0).then(|| 1.0 - self.distinct as f64 / self.claims as f64)
    }
}

fn claim_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("recommendationid", DataType::Utf8, false),
        Field::new("appid", DataType::UInt32, false),
        Field::new("claim_index", DataType::UInt16, false),
        // Byte offsets into the review as captured, so a claim can be recovered from the
        // corpus rather than stored twice.
        Field::new("start", DataType::UInt32, false),
        Field::new("end", DataType::UInt32, false),
        Field::new("language", DataType::Utf8, false),
        Field::new("text_sha256", DataType::Utf8, false),
    ]))
}

/// Splits every review in the most recent capture into the points it makes.
///
/// Writes `claims.parquet` beside the capture: one row per claim, carrying offsets rather
/// than text, so the corpus is not stored twice and a published label set can point at spans
/// of reviews without redistributing them.
///
/// # Errors
///
/// Fails if there is no capture, or if reading or writing fails.
pub fn extract_corpus(
    out_dir: &Path,
    app_id: u32,
    mut on_progress: impl FnMut(u64, u64),
) -> Result<ClaimReport> {
    let snapshot = crate::embed::latest_snapshot(out_dir, app_id)?;
    let path = snapshot.join("claims.parquet");
    let schema = claim_schema();
    let mut writer = ArrowWriter::try_new(
        std::fs::File::create(&path)?,
        Arc::clone(&schema),
        Some(
            WriterProperties::builder()
                .set_compression(Compression::ZSTD(ZstdLevel::default()))
                .build(),
        ),
    )?;

    let mut seen: HashSet<[u8; 32]> = HashSet::new();
    let mut report = ClaimReport {
        app_id,
        reviews: 0,
        empty: 0,
        claims: 0,
        distinct: 0,
    };
    let mut pending = ClaimBatch::default();

    crate::capture::for_each_body(&snapshot, |id, language, text| {
        report.reviews += 1;
        let found = claims_of(text);
        if found.is_empty() {
            report.empty += 1;
        }
        for (index, (at, claim)) in found.into_iter().enumerate() {
            seen.insert(crate::embed::sha256_bytes(&claim));
            report.claims += 1;
            pending.push(id, app_id, index, &at, language, &claim);
        }
        if pending.len() >= 16_384 {
            writer.write(&pending.take(&schema)?)?;
            on_progress(report.reviews, report.claims);
        }
        Ok(())
    })?;

    if pending.len() > 0 {
        writer.write(&pending.take(&schema)?)?;
    }
    writer.close()?;
    report.distinct = seen.len() as u64;
    Ok(report)
}

#[derive(Default)]
struct ClaimBatch {
    ids: Vec<String>,
    appids: Vec<u32>,
    indexes: Vec<u16>,
    starts: Vec<u32>,
    ends: Vec<u32>,
    languages: Vec<String>,
    digests: Vec<String>,
}

impl ClaimBatch {
    fn len(&self) -> usize {
        self.ids.len()
    }

    fn push(
        &mut self,
        id: &str,
        app_id: u32,
        index: usize,
        at: &std::ops::Range<usize>,
        language: &str,
        claim: &str,
    ) {
        self.ids.push(id.to_owned());
        self.appids.push(app_id);
        self.indexes.push(u16::try_from(index).unwrap_or(u16::MAX));
        self.starts
            .push(u32::try_from(at.start).unwrap_or(u32::MAX));
        self.ends.push(u32::try_from(at.end).unwrap_or(u32::MAX));
        self.languages.push(language.to_owned());
        self.digests.push(crate::embed::sha256_hex(claim));
    }

    fn take(&mut self, schema: &Arc<Schema>) -> Result<RecordBatch> {
        let mut ids = StringBuilder::new();
        let mut appids = UInt32Builder::new();
        let mut indexes = UInt16Builder::new();
        let mut starts = UInt32Builder::new();
        let mut ends = UInt32Builder::new();
        let mut languages = StringBuilder::new();
        let mut digests = StringBuilder::new();

        for row in 0..self.len() {
            ids.append_value(&self.ids[row]);
            appids.append_value(self.appids[row]);
            indexes.append_value(self.indexes[row]);
            starts.append_value(self.starts[row]);
            ends.append_value(self.ends[row]);
            languages.append_value(&self.languages[row]);
            digests.append_value(&self.digests[row]);
        }
        self.ids.clear();
        self.appids.clear();
        self.indexes.clear();
        self.starts.clear();
        self.ends.clear();
        self.languages.clear();
        self.digests.clear();

        let columns: Vec<ArrayRef> = vec![
            Arc::new(ids.finish()),
            Arc::new(appids.finish()),
            Arc::new(indexes.finish()),
            Arc::new(starts.finish()),
            Arc::new(ends.finish()),
            Arc::new(languages.finish()),
            Arc::new(digests.finish()),
        ];
        Ok(RecordBatch::try_new(Arc::clone(schema), columns)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_review_about_three_things_is_three_claims() {
        let claims = split(
            "Looks incredible. Runs like a slideshow on my machine. The story is the best in the series.",
        );
        assert_eq!(claims.len(), 3);
        assert_eq!(claims[0], "Looks incredible.");
        assert!(claims[2].starts_with("The story"));
    }

    #[test]
    fn a_score_is_not_a_sentence_boundary() {
        assert_eq!(
            split("Solid 9.5 out of 10 for the soundtrack alone."),
            vec!["Solid 9.5 out of 10 for the soundtrack alone."]
        );
    }

    #[test]
    fn full_width_stops_split_where_there_are_no_spaces() {
        let claims = split(
            "\u{753B}\u{9762}\u{304C}\u{7DBA}\u{9E97}\u{3067}\u{3059}\u{3002}\u{3067}\u{3082}\u{5024}\u{6BB5}\u{304C}\u{9AD8}\u{3059}\u{304E}\u{307E}\u{3059}\u{3002}",
        );
        assert_eq!(claims.len(), 2);
    }

    #[test]
    fn a_trailing_verdict_joins_what_it_qualifies() {
        // "Buy it." on its own is not a point about anything; attached, it is the verdict on
        // the point before it.
        let claims = split("The combat finally feels weighty and fast. Buy it.");
        assert_eq!(claims.len(), 1);
    }

    #[test]
    fn an_opening_fragment_joins_what_follows_it() {
        let claims = split("Yes. It is worth every penny at full price.");
        assert_eq!(claims, vec!["Yes. It is worth every penny at full price."]);
    }

    #[test]
    fn one_long_sentence_is_one_claim() {
        let claims =
            split("i played this for six hundred hours and i still have no idea what is going on");
        assert_eq!(claims.len(), 1);
    }

    #[test]
    fn newlines_end_a_point_even_without_punctuation() {
        let claims = split("Pros\nThe driving model is superb\nCons\nThe menus are a disaster");
        assert_eq!(claims.len(), 2);
        assert!(claims[0].contains("driving model"));
    }

    #[test]
    fn runs_of_punctuation_are_one_boundary() {
        let claims = split("Wait... what?! The ending was cut out of the retail release.");
        assert_eq!(claims.len(), 2);
    }

    #[test]
    fn an_abbreviation_does_not_end_a_point() {
        let claims = split(
            "Bring a friend, e.g. someone who likes being shouted at, and it is a great time.",
        );
        assert_eq!(claims.len(), 1);
    }

    #[test]
    fn text_with_nothing_in_it_makes_no_claims() {
        assert!(split("   \n\n  ").is_empty());
        assert!(split("").is_empty());
    }

    /// Every case below was flagged by a labeller reading real reviews, which is the only
    /// evidence any splitting rule here has.
    #[test]
    fn a_bullet_is_not_part_of_the_point_it_introduces() {
        let claims = split("- The tutorial explains nothing at all\n- The interface is a mess");
        assert_eq!(claims.len(), 2);
        assert!(claims[0].starts_with("The tutorial"), "got {:?}", claims[0]);
        assert!(
            claims[1].starts_with("The interface"),
            "got {:?}",
            claims[1]
        );
    }

    #[test]
    fn trailing_dashes_used_as_a_divider_are_not_part_of_the_claim() {
        let claims = split("The translation is full of mistakes, --\nEverything else is fine.");
        assert_eq!(claims[0], "The translation is full of mistakes");
    }

    #[test]
    fn a_heading_belongs_to_what_it_introduces() {
        let claims = split("Spoiler about the endgame:\nThe last chapter undoes the whole story.");
        assert_eq!(claims.len(), 1, "got {claims:?}");
        assert!(claims[0].contains("last chapter"));
    }

    #[test]
    fn a_sentence_inside_a_quotation_is_not_the_reviewers_own() {
        let claims = split(
            "The devs keep saying \"we hear you. we are working on it.\" and nothing changes.",
        );
        assert_eq!(claims.len(), 1, "got {claims:?}");
    }

    #[test]
    fn a_row_of_dashes_is_a_divider_rather_than_a_claim() {
        let claims = split("Great game.\n-----\nWould buy again at that price honestly.");
        assert!(
            claims
                .iter()
                .all(|claim| !claim.chars().all(|ch| ch == '-')),
            "got {claims:?}"
        );
    }

    #[test]
    fn steam_markup_is_not_something_anybody_said() {
        let claims = split("[b]Great[/b] combat and [i]awful[/i] menus throughout the game.");
        assert_eq!(claims.len(), 1);
        assert_eq!(
            claims[0],
            "Great combat and awful menus throughout the game."
        );
    }

    #[test]
    fn a_heading_tag_starts_a_new_point() {
        let claims = split(
            "[h3]Combat[/h3]\nThe parrying is the best in years.\n[h3]Sound[/h3]\nMuffled and thin.",
        );
        assert_eq!(claims.len(), 2, "got {claims:?}");
        assert!(claims[0].contains("parrying"), "got {:?}", claims[0]);
        assert!(claims[1].contains("Muffled"), "got {:?}", claims[1]);
        assert!(claims.iter().all(|claim| !claim.contains("[h3]")));
    }

    /// The copypasta four labellers in a row flagged, in the shape they actually found it.
    #[test]
    fn a_template_is_the_boxes_the_reviewer_ticked() {
        let claims = split(
            "{ Graphics }\n\u{2610} You forget what reality is\n\u{2611} Beautiful\n\u{2610} Good\n\u{2610} MS-DOS\n{ Gameplay }\n\u{2610} Very good\n\u{2611} Good\n\u{2610} Mehh\n",
        );
        assert_eq!(claims.len(), 2, "got {claims:?}");
        assert!(claims[0].contains("Graphics"), "got {:?}", claims[0]);
        assert!(claims[0].contains("Beautiful"), "got {:?}", claims[0]);
        assert!(
            !claims[0].contains("MS-DOS"),
            "an option nobody ticked is not a claim: {:?}",
            claims[0]
        );
        assert!(claims[1].contains("Gameplay"), "got {:?}", claims[1]);
    }

    /// Reviewers without a font that has a tick in it type an "x". Recognising the template
    /// is still the blank boxes' work, since nobody writes one of those by accident.
    #[test]
    fn a_template_filled_in_with_an_x_is_still_filled_in() {
        let claims = split(
            "Audience\n\u{2610} Kids\nx Everyone\n\u{2610} Veterans\nSound\n\u{2610} Tinny\nX Superb\n\u{2610} Deafening\n",
        );
        assert_eq!(claims.len(), 2, "got {claims:?}");
        assert!(claims[0].contains("Everyone"), "got {:?}", claims[0]);
        assert!(claims[1].contains("Superb"), "got {:?}", claims[1]);
        assert!(
            !claims.iter().any(|claim| claim.contains("Veterans")),
            "got {claims:?}"
        );
    }

    #[test]
    fn a_drawing_is_one_claim_rather_than_one_per_line() {
        let claims = split(
            "Best game ever.\n( \u{0361}\u{00B0} \u{035C}\u{0296} \u{0361}\u{00B0})\n/|\\ /|\\\n_/ \\_\n| |\nBuy it now.",
        );
        assert!(claims.len() <= 2, "got {claims:?}");
        assert!(
            claims.iter().any(|claim| claim.contains("Best game")),
            "got {claims:?}"
        );
    }

    #[test]
    fn an_ordinary_review_is_not_a_template() {
        // Two lines of punctuation is a shrug and a face, not somebody drawing, and a review
        // with no boxes in it must go through the ordinary path untouched.
        let claims = split(
            "The combat is superb.\n:)\n\u{00AF}\\_(\u{30C4})_/\u{00AF}\nThe menus are a disaster.",
        );
        assert!(claims.len() >= 2, "got {claims:?}");
        assert!(claims[0].contains("combat"), "got {:?}", claims[0]);
    }

    #[test]
    fn a_bold_heading_before_a_list_belongs_to_its_first_item() {
        let claims = split(
            "[b][u]The Good[/u][/b]:\n[list]\n[*]Mission variety keeps every run fresh.\n[*]The sound design is superb.\n[/list]",
        );
        assert_eq!(claims.len(), 2, "got {claims:?}");
        assert!(claims[0].starts_with("The Good:"), "got {:?}", claims[0]);
        assert!(claims[0].contains("Mission variety"), "got {:?}", claims[0]);
        assert!(claims[1].starts_with("The sound"), "got {:?}", claims[1]);
    }

    #[test]
    fn a_heading_glued_to_a_sentence_end_is_still_a_heading() {
        let claims = split(
            "[h3]Combat[/h3]The parrying is the best in years.[h3]Sound[/h3]Muffled and thin.",
        );
        assert_eq!(claims.len(), 2, "got {claims:?}");
        assert!(claims[1].starts_with("Sound"), "got {:?}", claims[1]);
    }

    #[test]
    fn a_heading_tag_opened_mid_sentence_is_emphasis() {
        assert_eq!(
            split("The biggest improvement might be [h1]EVERYONE WALK FAST[/h1], really nice"),
            vec!["The biggest improvement might be EVERYONE WALK FAST, really nice"]
        );
        let claims = split(
            "\u{91CD}\u{5934}\u{620F}\u{6765}\u{4E86}\u{FF1A}[h2] \u{6574}\u{4E2A}UI\u{8BBE}\u{8BA1} [/h2]\u{FF0C}\u{5C24}\u{5176}\u{662F}\u{5546}\u{5E97}\u{754C}\u{9762}\u{5B9E}\u{5728}\u{592A}\u{4E71}\u{4E86}",
        );
        assert_eq!(claims.len(), 1, "got {claims:?}");
    }

    #[test]
    fn a_heading_after_a_list_opens_its_own_line() {
        let claims = split(
            "[list][*]Runs well on old hardware[*]Looks great at night[/list][h1]Cons[/h1]\nThe menus are a disaster.",
        );
        assert_eq!(claims.len(), 3, "got {claims:?}");
        assert!(claims[2].starts_with("Cons"), "got {:?}", claims[2]);
        assert!(claims[2].contains("menus"), "got {:?}", claims[2]);
    }

    #[test]
    fn a_list_of_short_parts_is_a_point_per_part() {
        let claims = split("Stunning visual, calm music, epic story");
        assert_eq!(claims.len(), 3, "got {claims:?}");
        assert_eq!(claims[1], "calm music");
        let claims = split(
            "\u{753B}\u{9762}\u{7F8E}\u{3057}\u{3044}\u{3001}\u{97F3}\u{697D}\u{6700}\u{9AD8}\u{3001}\u{7269}\u{8A9E}\u{6DF1}\u{3044}",
        );
        assert_eq!(claims.len(), 3, "got {claims:?}");
    }

    #[test]
    fn a_clause_set_off_by_commas_is_not_a_list() {
        assert_eq!(
            split("The combat, which took a while to click, is superb").len(),
            1
        );
        assert_eq!(
            split("Runs well on my machine, looks great at night, and the story kept me going until three in the morning").len(),
            1
        );
    }

    #[test]
    fn a_short_answer_belongs_to_its_question() {
        assert_eq!(
            split("Want to see your objectives? Top left of the screen."),
            vec!["Want to see your objectives? Top left of the screen."]
        );
        let claims = split("Is this Frost Punk? No. Is this a phenomenal game? Yes.");
        assert_eq!(claims.len(), 2, "got {claims:?}");
        assert_eq!(claims[0], "Is this Frost Punk? No.");
    }

    #[test]
    fn a_long_sentence_after_a_question_is_its_own_point() {
        let claims = split(
            "Is the story any good? The writing is sharp for the first ten hours and falls apart after that.",
        );
        assert_eq!(claims.len(), 2, "got {claims:?}");
    }

    #[test]
    fn an_emoticon_colours_the_sentence_before_it() {
        let claims = split("Great fun with friends. :D If you like Vermintide you will like this.");
        assert_eq!(claims.len(), 2, "got {claims:?}");
        assert_eq!(claims[0], "Great fun with friends. :D");
        assert!(claims[1].starts_with("If you"), "got {:?}", claims[1]);
    }

    #[test]
    fn a_line_ending_on_a_comma_continues_on_the_next() {
        let claims = split(
            "\u{7D4C}\u{6E08}\u{306F}\u{3068}\u{3066}\u{3082}\u{53B3}\u{3057}\u{304F}\u{3001}\n\u{5E8F}\u{76E4}\u{306F}\u{91D1}\u{304C}\u{5168}\u{304F}\u{8DB3}\u{308A}\u{306A}\u{3044}",
        );
        assert_eq!(claims.len(), 1, "got {claims:?}");
        let claims = split(
            "The economy is brutal at the start,\nand the tutorial never explains why.\nThe art is lovely though.",
        );
        assert_eq!(claims.len(), 2, "got {claims:?}");
        assert!(claims[0].contains("tutorial"), "got {:?}", claims[0]);
    }

    #[test]
    fn an_edition_is_an_abbreviation() {
        assert_eq!(
            split("Since I've been playing the Def. Editions for a while, the changes stand out.")
                .len(),
            1
        );
    }

    #[test]
    fn a_list_of_points_is_a_list_of_claims() {
        let claims =
            split("[list][*]The interface is unusable[*]The tutorial explains nothing[/list]");
        assert_eq!(claims.len(), 2, "got {claims:?}");
        assert!(claims[0].contains("interface"));
        assert!(claims[1].contains("tutorial"));
    }

    #[test]
    fn a_span_names_the_words_and_not_the_markup_around_them() {
        let text = "[list][*]The interface is unusable[*]The tutorial explains nothing[/list]";
        let spans: Vec<&str> = spans(text).into_iter().map(|at| &text[at]).collect();
        assert_eq!(
            spans,
            vec!["The interface is unusable", "The tutorial explains nothing"]
        );
        // A span recorded before the tags were trimmed comes to the same words.
        assert_eq!(tidied(text, 6, 34), Some(9..34));
        assert_eq!(tidied(text, 34, text.len()), Some(37..66));
        assert_eq!(tidied("[b]1. [i]Point[/i][/b]", 0, 22), Some(9..14));
    }

    #[test]
    fn a_score_in_brackets_is_a_claim_rather_than_markup() {
        let claims = split("[10/10] would freeze to death again in this wonderful city builder.");
        assert!(claims[0].contains("10/10"), "got {claims:?}");
    }

    #[test]
    fn a_web_address_is_not_two_thoughts() {
        let claims =
            split("Compare the charts at https://example.com/a?b=1&c=2 before you buy it.");
        assert_eq!(claims.len(), 1, "got {claims:?}");
    }

    /// Reviews are full of non-breaking spaces, and stepping over one by a single byte lands
    /// in the middle of a character.
    #[test]
    fn a_semicolon_joins_rather_than_ends() {
        let claims = split("It is not just a game; it is an experience worth having twice.");
        assert_eq!(claims.len(), 1, "got {claims:?}");
    }

    #[test]
    fn a_numbered_list_is_numbered_points_rather_than_numbers_and_points() {
        let claims = split("1. The interface is unusable.\n2. The tutorial explains nothing.");
        assert_eq!(claims.len(), 2, "got {claims:?}");
        assert!(
            claims[0].starts_with("The interface"),
            "got {:?}",
            claims[0]
        );
        assert!(claims[1].starts_with("The tutorial"), "got {:?}", claims[1]);
    }

    #[test]
    fn an_abbreviation_in_any_language_does_not_end_a_point() {
        assert_eq!(
            split("Es dauert ca. 40 Stunden bis zum Ende der Kampagne.").len(),
            1
        );
        assert_eq!(
            split("Roughly 40 hours, vs. 20 for the first one, which is generous.").len(),
            1
        );
        assert_eq!(
            split("Play with a friend, i.e. Someone patient, and it is a great time.").len(),
            1
        );
        assert_eq!(
            split("Manche Missionen dauern z.B. 40 Minuten ohne Speicherpunkt.").len(),
            1
        );
    }

    #[test]
    fn a_short_word_before_a_stop_is_still_a_sentence_ending() {
        let claims = split("The combat is genuinely fun. 10 out of 10 from me, no notes at all.");
        assert_eq!(claims.len(), 2, "got {claims:?}");
    }

    #[test]
    fn an_aside_in_brackets_does_not_end_the_sentence_around_it() {
        let claims = split("The campaign (which took me 40 hrs. and change) is the best part.");
        assert_eq!(claims.len(), 1, "got {claims:?}");
    }

    #[test]
    fn a_multi_byte_space_before_a_boundary_does_not_panic() {
        let claims =
            split("The soundtrack is superb.\u{a0}The mixing is not. It sits far too low.");
        assert!(claims.len() >= 2, "got {claims:?}");
    }

    #[test]
    fn every_claim_is_a_slice_of_what_went_in() {
        let review = "Great port. Runs at 4k60 on a 3060. Steam Deck verified too.";
        for claim in split(review) {
            assert!(review.contains(claim.as_ref()));
        }
    }
}
