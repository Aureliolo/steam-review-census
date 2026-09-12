//! What the processor half of a reading costs, and where inside it the time goes.
//!
//! A reading is a card and a processor taking turns: the card runs the graph, the processor
//! finds each claim's window and turns the pair into token ids. Which of the two a reading is
//! waiting on decides what is worth changing, and the answer has moved twice already, so it is
//! measured here rather than argued about.
//!
//!     cargo run --release -p steamgauge-core --example encoder-cost -- \
//!         data models/claim-reader 245170

use std::sync::Arc;
use std::time::Instant;

use steamgauge_core::{
    read::Depth,
    reader::{Asked, ClaimReader},
};

/// One claim with the review it sits in, owned, as the reading pass queues them.
struct Queued {
    claim: String,
    review: Arc<str>,
    at: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let out = args.next().unwrap_or_else(|| "data".to_owned());
    let model = args
        .next()
        .unwrap_or_else(|| "models/claim-reader".to_owned());
    let app_id: u32 = args
        .next()
        .and_then(|id| id.parse().ok())
        .unwrap_or(245_170);
    let batch: usize = args.next().and_then(|n| n.parse().ok()).unwrap_or(128);
    let wanted: usize = args
        .next()
        .and_then(|n| n.parse().ok())
        .unwrap_or(16_384)
        .max(batch);

    let reader = ClaimReader::load(std::path::Path::new(&model))?;
    let encoder = reader.encoder();
    let snapshot = steamgauge_core::embed::latest_snapshot(std::path::Path::new(&out), app_id)?;

    // The same window the reading pass fills, taken from the front of the corpus, and sorted
    // by length the way the pass sorts it: batches of claims of a size are what the card
    // actually sees, and a batch of mixed lengths pads to its longest member.
    let mut queued: Vec<Queued> = Vec::with_capacity(wanted);
    steamgauge_core::capture::for_each_row(&snapshot, |_, text| {
        if queued.len() >= wanted {
            return Ok(());
        }
        let claims = Depth::Deep.claims_of(text);
        let review: Arc<str> = Arc::from(claims.join(" "));
        let mut at = 0;
        for claim in &claims {
            queued.push(Queued {
                claim: claim.to_string(),
                review: Arc::clone(&review),
                at,
            });
            at += claim.len() + 1;
        }
        Ok(())
    })?;
    queued.truncate(wanted);
    queued.sort_unstable_by_key(|one| one.claim.len() + one.review.len());

    let asked: Vec<Asked<'_>> = queued
        .iter()
        .map(|one| Asked {
            claim: &one.claim,
            review: &one.review,
            at: one.at,
        })
        .collect();

    println!(
        "{} claims from app {app_id}, batches of {batch}, on {}",
        asked.len(),
        reader.device()
    );

    let started = Instant::now();
    let mut windows = 0_usize;
    for chunk in asked.chunks(batch) {
        windows += encoder.windows_for(chunk).len();
    }
    let cutting = started.elapsed();

    let started = Instant::now();
    for chunk in asked.chunks(batch) {
        encoder.prepare(chunk)?;
    }
    let preparing = started.elapsed();

    #[expect(
        clippy::cast_precision_loss,
        reason = "a window of claims is thousands, not quadrillions"
    )]
    let rate = |taken: std::time::Duration| asked.len() as f64 / taken.as_secs_f64();
    println!(
        "  windows   {:>6.2}s  {:>7.0} claims a second ({windows} cut)",
        cutting.as_secs_f64(),
        rate(cutting)
    );
    println!(
        "  prepare   {:>6.2}s  {:>7.0} claims a second (windows, then token ids)",
        preparing.as_secs_f64(),
        rate(preparing)
    );
    println!(
        "  of which tokenising {:>6.2}s",
        preparing.saturating_sub(cutting).as_secs_f64()
    );
    Ok(())
}
