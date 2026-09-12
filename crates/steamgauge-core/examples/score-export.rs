//! Scores the shipped reader over the exported training claims, in Rust, on the exact input
//! training used.
//!
//! There are two implementations of "ask the model about this claim": `training/train.py`
//! builds the window in Python and this crate builds it in Rust. They have to agree, and when
//! they do not, every figure the tool prints is about a different question from the one the
//! model was trained on. `training/frontier.py reader` is the Python side of this check;
//! `steamgauge measure-claims` scores the tool over a corpus it split itself, which adds the
//! splitter to the comparison. This one removes it: same claims, same reviews, same offsets as
//! the exported file, so a disagreement with Python is the reader and nothing else.
//!
//!     cargo run --release -p steamgauge-core --example score-export -- \
//!         training/data/claims.jsonl models/claim-reader 214490

use std::collections::HashMap;

use steamgauge_core::{
    reader::{Asked, ClaimReader},
    taxonomy::CORE_SPINE,
};

#[derive(serde::Deserialize)]
struct Row {
    text: String,
    review: Option<String>,
    review_offset: Option<usize>,
    subject: String,
    app_id: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .unwrap_or_else(|| "training/data/claims.jsonl".to_owned());
    let model = args
        .next()
        .unwrap_or_else(|| "models/claim-reader".to_owned());
    let only: Option<u32> = args.next().and_then(|id| id.parse().ok());

    let mut reader = ClaimReader::load(std::path::Path::new(&model))?;
    let text = std::fs::read_to_string(&path)?;
    let rows: Vec<Row> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    let rows: Vec<&Row> = rows
        .iter()
        .filter(|row| only.is_none_or(|id| row.app_id == id))
        .collect();
    println!("{} claims from {path}", rows.len());

    if std::env::var("SHOW_WINDOWS").is_ok() {
        for row in rows.iter().take(3) {
            let asked = Asked {
                claim: &row.text,
                review: row.review.as_deref().unwrap_or(&row.text),
                at: row.review_offset.unwrap_or(0),
            };
            let windows = reader.windows_for(std::slice::from_ref(&asked));
            println!("CLAIM {:?}", row.text);
            println!("AT {} REVIEW LEN {}", asked.at, asked.review.len());
            println!("WINDOW {:?}\n", windows[0]);
        }
    }

    let mut answered = 0_u64;
    let mut agreed = 0_u64;
    let mut confused: HashMap<&str, u64> = HashMap::new();
    for chunk in rows.chunks(128) {
        let asked: Vec<Asked<'_>> = chunk
            .iter()
            .map(|row| Asked {
                claim: &row.text,
                review: row.review.as_deref().unwrap_or(&row.text),
                at: row.review_offset.unwrap_or(0),
            })
            .collect();
        for (row, reading) in chunk.iter().zip(reader.read(&asked)?) {
            let Some(said) = reading.subject.and_then(|at| CORE_SPINE.get(at)) else {
                continue;
            };
            answered += 1;
            if said.id == row.subject {
                agreed += 1;
            } else {
                *confused.entry(said.id).or_default() += 1;
            }
        }
    }

    #[expect(clippy::cast_precision_loss, reason = "thousands of claims, not 2^53")]
    let share = |part: u64, whole: u64| {
        if whole == 0 {
            0.0
        } else {
            part as f64 / whole as f64 * 100.0
        }
    };
    println!(
        "answered {answered} of {} ({:.1}%), agreed {agreed} ({:.1}%)",
        rows.len(),
        share(answered, rows.len() as u64),
        share(agreed, answered)
    );
    Ok(())
}
