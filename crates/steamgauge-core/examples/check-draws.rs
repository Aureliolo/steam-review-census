//! Does a drawn handout still name the claims this build cuts?
//!
//! A draw is made once and labelled days later, and the splitter can move in between. A label
//! carries the byte span of its claim, so a span this build still cuts as one claim finds its
//! reading wherever that claim now sits; a span it no longer cuts is dropped from every
//! measurement. That is safe and it is not free: a handout mostly made of spans this build no
//! longer cuts is a labelling run whose answers cannot be scored, and the quota it costs is the
//! scarcest thing in this project.
//!
//! So before spending one, ask. The spans are in each game's `sample.json`, which is committed,
//! and they are ranges into the review as captured, so they are checked against the capture. A
//! game with no capture on this machine falls back to the copy of the text in `batches/`, which
//! is the claims rejoined rather than the review as written, and reads as a claim that moved.
//!
//!     cargo run --release -p steamgauge-core --example check-draws -- reference/claims data

use std::collections::HashMap;

use steamgauge_core::read::Depth;

#[derive(serde::Deserialize)]
struct Drawn {
    id: String,
    claims: Vec<Span>,
}

#[derive(serde::Deserialize)]
struct Span {
    start: usize,
    end: usize,
    text: String,
}

#[derive(serde::Deserialize)]
struct Handed {
    review_id: String,
    review: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "reference/claims".to_owned());
    let library =
        std::path::PathBuf::from(std::env::args().nth(2).unwrap_or_else(|| "data".to_owned()));
    let mut games: Vec<std::path::PathBuf> = std::fs::read_dir(&root)?
        .filter_map(|entry| entry.ok().map(|found| found.path()))
        .filter(|path| path.is_dir())
        .collect();
    games.sort();

    let mut total = 0_usize;
    let mut stale = 0_usize;
    let mut retidied = 0_usize;
    let mut unread = 0_usize;
    for game in games {
        let sample = game.join("sample.json");
        if !sample.is_file() {
            continue;
        }
        let drawn: Vec<Drawn> = serde_json::from_slice(&std::fs::read(&sample)?)?;
        // The capture rather than the handout beside it: a span is a range into the review as
        // captured, and that is what the ingest resolves it against.
        let wanted: std::collections::HashSet<&str> =
            drawn.iter().map(|review| review.id.as_str()).collect();
        let texts = match captured(&library, &game, &wanted) {
            Ok(found) if !found.is_empty() => found,
            _ => handed_out(&game.join("batches"))?,
        };
        let mut held = 0_usize;
        let mut moved = 0_usize;
        let mut tidied = 0_usize;
        let mut missing = 0_usize;
        for review in &drawn {
            let Some(text) = texts.get(&review.id) else {
                missing += review.claims.len();
                continue;
            };
            // A drawn span is a byte range into the review as captured, which is what a label
            // carries and what the ingest resolves against. The claim's text beside it is the
            // tidied piece, which is the same words with markup and bullets taken out, so the
            // span and the text are checked apart: a span this build no longer cuts is a label
            // that cannot be scored, and a span it cuts whose text reads differently is only
            // the tidying having changed.
            let spans = Depth::Deep.spans_of(text);
            for claim in &review.claims {
                held += 1;
                if !spans
                    .iter()
                    .any(|span| span.start == claim.start && span.end == claim.end)
                {
                    moved += 1;
                } else if text.get(claim.start..claim.end) != Some(claim.text.as_str()) {
                    tidied += 1;
                }
            }
        }
        total += held;
        stale += moved;
        retidied += tidied;
        unread += missing;
        let name = game.file_name().unwrap_or_default().to_string_lossy();
        if moved > 0 {
            println!("{name}: {moved} of {held} claims are not what this build cuts there");
        }
    }

    println!("\n{total} drawn claims checked, {stale} the splitter no longer cuts as drawn");
    if retidied > 0 {
        println!("{retidied} are cut in the same place and tidied differently, which scores fine");
    }
    if unread > 0 {
        println!("{unread} could not be checked: the text they index is not on this machine");
    }
    Ok(())
}

/// The reviews a draw names, as the capture holds them.
fn captured(
    library: &std::path::Path,
    game: &std::path::Path,
    wanted: &std::collections::HashSet<&str>,
) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
    let app_id = game.file_name().unwrap_or_default().to_string_lossy();
    let mut snapshots: Vec<std::path::PathBuf> =
        std::fs::read_dir(library.join(format!("appid={app_id}")))?
            .filter_map(|entry| entry.ok().map(|found| found.path()))
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("snapshot="))
            })
            .collect();
    snapshots.sort();
    let Some(snapshot) = snapshots.pop() else {
        return Ok(HashMap::new());
    };
    let mut texts = HashMap::new();
    steamgauge_core::capture::for_each_row(&snapshot, |row, text| {
        if wanted.contains(row.recommendationid.as_str()) {
            texts.insert(row.recommendationid, text.to_owned());
        }
        Ok(())
    })?;
    Ok(texts)
}

/// The review text each handed-out batch carries, by review id.
fn handed_out(
    batches: &std::path::Path,
) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
    let mut texts = HashMap::new();
    let Ok(entries) = std::fs::read_dir(batches) else {
        return Ok(texts);
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.extension().is_none_or(|kind| kind != "json") {
            continue;
        }
        let handed: Vec<Handed> = serde_json::from_slice(&std::fs::read(&path)?)?;
        for review in handed {
            texts.insert(review.review_id, review.review);
        }
    }
    Ok(texts)
}
