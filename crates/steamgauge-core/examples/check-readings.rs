//! Does every reading say what its own rows hold?
//!
//! A reading is two artefacts written by one pass: `reading.json`, which holds the counts every
//! rate and report is built from, and `readings.parquet`, which holds one row per claim. Nothing
//! else checks that they agree, and a pass that counted a review twice, or lost its place inside
//! one, would leave a corpus that reads perfectly and is wrong.
//!
//! Four things have to hold, and each is a different way for the walk to have gone wrong:
//! the rows are as many as the claims counted, the rows with no subject are as many as the
//! claims declined, each review's rows sit together and are numbered from zero, and the reviews
//! in the rows plus the reviews the splitter found nothing in are the reviews counted.
//!
//!     cargo run --release -p steamgauge-core --example check-readings -- data

use std::collections::HashMap;

use steamgauge_core::read::ReadReport;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::args().nth(1).unwrap_or_else(|| "data".to_owned());
    let mut checked = 0_usize;
    let mut complained = 0_usize;
    let mut stale = 0_usize;

    let mut games: Vec<std::path::PathBuf> = std::fs::read_dir(&root)?
        .filter_map(|entry| entry.ok().map(|found| found.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("appid="))
        })
        .collect();
    games.sort();

    for game in games {
        let Some(snapshot) = newest(&game)? else {
            continue;
        };
        let sidecar = snapshot.join("reading.json");
        if !sidecar.is_file() {
            continue;
        }
        let reading: ReadReport = serde_json::from_slice(&std::fs::read(&sidecar)?)?;
        let rows = snapshot.join("readings.parquet");
        if !rows.is_file() {
            println!("{}: a reading with no rows beside it", reading.app_id);
            complained += 1;
            continue;
        }
        // A read in flight has the file open and has written nothing to it yet, which is not
        // something wrong with the corpus.
        if std::fs::metadata(&rows)?.len() == 0 {
            println!("{}: being read right now, skipped", reading.app_id);
            continue;
        }

        // A reading cut by another splitter cannot be reconciled against this one's cut: the
        // pieces are different pieces. It is stale rather than wrong, and the tool refuses it
        // wherever a claim would be quoted or scored, so this says so and moves on.
        if !reading.splitter.is_empty()
            && reading.splitter != steamgauge_core::claims::SPLITTER_VERSION
        {
            println!(
                "{}: cut by {}, and this build cuts {}: read the game again",
                reading.app_id,
                reading.splitter,
                steamgauge_core::claims::SPLITTER_VERSION
            );
            stale += 1;
            continue;
        }

        checked += 1;
        let mut said: Vec<String> = Vec::new();
        let mut counted = 0_u64;
        let mut unclassified = 0_u64;
        let mut runs = 0_u64;
        let mut out_of_order = 0_u64;
        let mut seen: HashMap<String, u32> = HashMap::new();
        let mut last: Option<String> = None;
        let mut expected = 0_usize;

        steamgauge_core::read::for_each_reading(&rows, |id, index, subject, _, _| {
            counted += 1;
            if subject.is_none() {
                unclassified += 1;
            }
            if last.as_deref() != Some(id) {
                runs += 1;
                *seen.entry(id.to_owned()).or_default() += 1;
                last = Some(id.to_owned());
                expected = 0;
            }
            if usize::from(index) != expected {
                out_of_order += 1;
            }
            expected += 1;
        })?;

        if counted != reading.claims {
            said.push(format!(
                "{counted} rows against {} claims counted",
                reading.claims
            ));
        }
        if unclassified != reading.unclassified_claims {
            said.push(format!(
                "{unclassified} rows with no subject against {} counted",
                reading.unclassified_claims
            ));
        }
        let repeated = seen.values().filter(|count| **count > 1).count();
        if repeated > 0 {
            said.push(format!("{repeated} reviews written in more than one place"));
        }
        if out_of_order > 0 {
            said.push(format!(
                "{out_of_order} claims numbered out of order inside their review"
            ));
        }
        // Counted from the capture rather than read out of the reading: an auditor that takes
        // the audited pass's word for the one number that reconciles it is not checking
        // anything. A reading made before the count existed is reconciled the same way.
        let walk = capture(&snapshot, &reading, &seen)?;
        if walk.walked != reading.reviews {
            said.push(format!(
                "the capture holds {} reviews to read where the reading counted {}",
                walk.walked, reading.reviews
            ));
        }
        if runs + walk.claimless != reading.reviews {
            said.push(format!(
                "{runs} reviews in the rows and {} the splitter finds no point in, against {} \
                 counted",
                walk.claimless, reading.reviews
            ));
        }
        // The one that is not arithmetic. A review with rows in the reading that this build
        // now finds nothing in was cut by a different splitter from the one in this binary,
        // whatever version both of them claim to be.
        if walk.both > 0 {
            said.push(format!(
                "{} reviews hold rows in the reading and no claim in this build: the splitter \
                 has moved without its version moving, e.g. {:?}",
                walk.both,
                walk.example.as_deref().unwrap_or("")
            ));
        }
        if reading.claimless_reviews > 0 && reading.claimless_reviews != walk.claimless {
            said.push(format!(
                "the reading says {} reviews hold no point where this build finds {}",
                reading.claimless_reviews, walk.claimless
            ));
        }

        if said.is_empty() {
            continue;
        }
        complained += 1;
        println!("{}: {}", reading.app_id, said.join("; "));
    }

    println!("\n{checked} readings checked, {complained} with something to say");
    if stale > 0 {
        println!("{stale} cut by an older splitter, which no arithmetic here can reconcile");
    }
    Ok(())
}

/// What a walk of the capture finds, counted the way the reading pass counts reviews: in the
/// language it was read in, and at the depth it was read at.
struct Walk {
    /// Reviews the reading pass would have visited.
    walked: u64,
    /// Of those, the ones this build's splitter finds no point in.
    claimless: u64,
    /// Of those, the ones that nonetheless have rows in the reading.
    both: u64,
    example: Option<String>,
}

fn capture(
    snapshot: &std::path::Path,
    reading: &ReadReport,
    with_rows: &HashMap<String, u32>,
) -> Result<Walk, Box<dyn std::error::Error>> {
    let mut walk = Walk {
        walked: 0,
        claimless: 0,
        both: 0,
        example: None,
    };
    steamgauge_core::capture::for_each_row(snapshot, |row, text| {
        if reading
            .language
            .as_ref()
            .is_some_and(|wanted| wanted != &row.language)
        {
            return Ok(());
        }
        walk.walked += 1;
        if !reading.depth.claims_of(text).is_empty() {
            return Ok(());
        }
        walk.claimless += 1;
        if with_rows.contains_key(&row.recommendationid) {
            walk.both += 1;
            if walk.example.is_none() {
                walk.example = Some(text.chars().take(80).collect());
            }
        }
        Ok(())
    })?;
    Ok(walk)
}

/// The newest snapshot of a game, which is the one every other command reads.
fn newest(game: &std::path::Path) -> std::io::Result<Option<std::path::PathBuf>> {
    let mut snapshots: Vec<std::path::PathBuf> = std::fs::read_dir(game)?
        .filter_map(|entry| entry.ok().map(|found| found.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("snapshot="))
        })
        .collect();
    snapshots.sort();
    Ok(snapshots.pop())
}
