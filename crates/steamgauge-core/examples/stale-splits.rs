//! How much of what labellers called a bad split this build still splits that way.
//!
//! A labeller marks a claim `split_wrong` when the piece they were handed makes more than one
//! point, or less than one. Those marks were made against the splitter of the day, and five
//! versions of rules have landed since the first of them. A rule already fixed still shows up
//! in the training summary as a mis-split claim, which makes the splitter look worse than it
//! is and hides whatever is still wrong with it.
//!
//! This re-cuts each flagged claim's review with the splitter this build has and asks whether
//! the piece the labeller objected to is still one piece.
//!
//!     cargo run --release -p steamgauge-core --example stale-splits -- training/data/claims.jsonl

use std::collections::HashMap;

use steamgauge_core::read::Depth;

#[derive(serde::Deserialize)]
struct Row {
    text: String,
    review: Option<String>,
    split_wrong: Option<bool>,
    app_id: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "training/data/claims.jsonl".to_owned());
    let show: usize = std::env::var("SHOW")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(0);

    let mut flagged = 0_usize;
    let mut still = 0_usize;
    let mut gone = 0_usize;
    let mut missing = 0_usize;
    let mut by_game: HashMap<u32, (usize, usize)> = HashMap::new();
    let mut shown = 0_usize;

    for line in std::fs::read_to_string(&path)?.lines() {
        let row: Row = serde_json::from_str(line)?;
        if row.split_wrong != Some(true) {
            continue;
        }
        flagged += 1;
        let Some(review) = row.review.as_deref() else {
            missing += 1;
            continue;
        };
        let pieces = Depth::Deep.claims_of(review);
        let entry = by_game.entry(row.app_id).or_default();
        if pieces.iter().any(|piece| piece.as_ref() == row.text) {
            still += 1;
            entry.0 += 1;
            if shown < show {
                shown += 1;
                println!("still one piece: {:?}", row.text);
            }
        } else {
            gone += 1;
            entry.1 += 1;
        }
    }

    println!("{flagged} claims a labeller called badly split");
    if missing > 0 {
        println!("  {missing} carry no review text, so this build cannot re-cut them");
    }
    let judged = still + gone;
    if judged > 0 {
        #[expect(
            clippy::cast_precision_loss,
            reason = "a reference set is thousands of claims, not quadrillions"
        )]
        let share = |n: usize| 100.0 * n as f64 / judged as f64;
        println!("  {still} are still one piece ({:.1}%)", share(still));
        println!("  {gone} are cut differently now ({:.1}%)", share(gone));
    }
    let mut worst: Vec<(u32, (usize, usize))> = by_game.into_iter().collect();
    worst.sort_by_key(|(_, (still, _))| std::cmp::Reverse(*still));
    for (app_id, (still, gone)) in worst.into_iter().take(8) {
        println!("  app {app_id}: {still} still, {gone} fixed");
    }
    Ok(())
}
