//! What each fishing line is actually catching, before any of it is labelled.
//!
//! A probe list is a guess about vocabulary and the cheapest way to find out it was a bad
//! guess is to read what it caught. "Accessible" looked like an `accessibility` probe until
//! this printed five claims about a game being easy to get into; "licence" looked like a
//! `licensing` probe until it returned a dwarf buying a beer licence.
//!
//! Prints a sample per line and, for each, how often the line fires across the capture, which
//! is the other half of the question: a line nobody trips is as useless as one everybody does.
//!
//!     cargo run --release -p steamgauge-core --example mine-check -- 548430 --show 6

use std::collections::HashMap;

use steamgauge_core::{Result, mine};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut app_ids: Vec<u32> = Vec::new();
    let mut show = 5usize;
    let mut out = std::path::PathBuf::from("data");
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--show" => show = args.next().and_then(|n| n.parse().ok()).unwrap_or(show),
            "--out" => out = args.next().map_or(out, std::path::PathBuf::from),
            other => {
                if let Ok(app_id) = other.parse() {
                    app_ids.push(app_id);
                }
            }
        }
    }
    if app_ids.is_empty() {
        eprintln!("usage: mine-check <app id>... [--show N] [--out data]");
        std::process::exit(2);
    }

    let mut caught: HashMap<&'static str, (usize, Vec<String>)> = HashMap::new();
    let mut claims_seen = 0usize;
    for app_id in app_ids {
        let snapshot = steamgauge_core::embed::latest_snapshot(&out, app_id)?;
        let reading: steamgauge_core::read::ReadReport =
            serde_json::from_slice(&std::fs::read(snapshot.join("reading.json"))?)?;
        let depth = reading.depth;
        steamgauge_core::capture::for_each_body(&snapshot, |_, _, text| {
            for claim in depth.claims_of(text) {
                claims_seen += 1;
                let Some(subject) = mine::hooked(&claim) else {
                    continue;
                };
                let entry = caught.entry(subject).or_insert_with(|| (0, Vec::new()));
                entry.0 += 1;
                // Keep the first few rather than a sample: the point is to read them, and the
                // first few of a corpus are as representative of the probe's vocabulary as any
                // other few would be.
                if entry.1.len() < show {
                    entry
                        .1
                        .push(claim.split_whitespace().collect::<Vec<_>>().join(" "));
                }
            }
            Ok(())
        })?;
    }

    println!("{claims_seen} claims read\n");
    for probe in mine::PROBES {
        let (count, examples) = caught
            .remove(probe.subject)
            .unwrap_or_else(|| (0, Vec::new()));
        #[expect(clippy::cast_precision_loss, reason = "millions of claims, not 2^53")]
        let rate = if claims_seen == 0 {
            0.0
        } else {
            count as f64 * 100.0 / claims_seen as f64
        };
        println!(
            "{:<16} {count:>7} claims  {rate:>6.3}% of the corpus",
            probe.subject
        );
        for example in examples {
            println!("    {}", example.chars().take(108).collect::<String>());
        }
        if count == 0 {
            println!(
                "    nothing, which is a probe list to rewrite or a subject this game never discusses"
            );
        }
        println!();
    }
    Ok(())
}
