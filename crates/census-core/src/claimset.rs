//! Drawing claims to be labelled, and handing them out in batches.
//!
//! A batch is a review with its claims numbered, never a heap of loose claims. Two reasons.
//! A claim on its own is often unreadable ("it doesn't", "same here", "this one too"), and
//! the labeller needs the review around it to know what it refers to. And sending the review
//! once with its claims enumerated costs a fraction of sending the review again for every
//! claim it contains.
//!
//! Selection is by hash of the review id, so it depends only on the review: a corpus that
//! gains reviews does not renumber the ones already labelled, and the same seed draws the
//! same sample on any machine.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::Result;

/// One claim as it is handed to a labeller and recorded in the sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawnClaim {
    pub index: u16,
    /// Byte offsets into the review as captured. The published label set carries these
    /// rather than the text, so a labelled span can be recovered from Steam by anyone
    /// without this project redistributing a word anybody wrote.
    pub start: u32,
    pub end: u32,
    pub text: String,
}

/// One review, split, as it is handed out.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawnReview {
    pub id: String,
    pub app_id: u32,
    pub language: String,
    /// `stratified` is training data and `random` is held back. A pilot draws only random,
    /// because the first question is what a corpus actually contains.
    pub subset: String,
    pub claims: Vec<DrawnClaim>,
}

#[derive(Debug, Clone, Copy)]
pub struct DrawReport {
    pub reviews: usize,
    pub claims: usize,
    pub batches: usize,
}

impl DrawReport {
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "a sample is thousands of claims, not 2^53 of them"
    )]
    pub fn per_review(&self) -> f64 {
        if self.reviews == 0 {
            return 0.0;
        }
        self.claims as f64 / self.reviews as f64
    }
}

/// Draws reviews at random from a corpus and splits each into its claims.
///
/// Every claim of a drawn review is labelled, never a subset of them: a review labelled in
/// part cannot say what share of a corpus is contentless, which is the first thing the pilot
/// has to answer.
///
/// # Errors
///
/// Fails if there is no capture for the app.
pub fn draw(out_dir: &Path, app_id: u32, wanted: usize, seed: u64) -> Result<Vec<DrawnReview>> {
    let snapshot = crate::embed::latest_snapshot(out_dir, app_id)?;
    let mut best: crate::bounded::Smallest<[u8; 32], DrawnReview> =
        crate::bounded::Smallest::new(wanted);

    crate::capture::for_each_body(&snapshot, |id, language, text| {
        let spans = crate::claims::spans(text);
        if spans.is_empty() {
            return Ok(());
        }
        let claims = spans
            .into_iter()
            .enumerate()
            .map(|(index, at)| DrawnClaim {
                index: u16::try_from(index).unwrap_or(u16::MAX),
                start: u32::try_from(at.start).unwrap_or(u32::MAX),
                end: u32::try_from(at.end).unwrap_or(u32::MAX),
                text: text[at].to_owned(),
            })
            .collect();
        best.offer(
            crate::sample::rank(seed, "claims", id),
            DrawnReview {
                id: id.to_owned(),
                app_id,
                language: language.to_owned(),
                subset: "random".to_owned(),
                claims,
            },
        );
        Ok(())
    })?;

    Ok(best.take())
}

/// What a labeller is shown: one review, its claims numbered, and nothing else.
///
/// No app id, no rating, no language name, no prediction. The model reads the claim and the
/// review around it, so a label made from more than that measures what the labeller was told
/// rather than how well the text reads.
#[derive(Debug, Clone, Serialize)]
struct Handout<'a> {
    review_id: &'a str,
    review: String,
    claims: Vec<HandoutClaim<'a>>,
}

#[derive(Debug, Clone, Serialize)]
struct HandoutClaim<'a> {
    index: u16,
    text: &'a str,
}

/// Writes the drawn sample and the batches to hand out.
///
/// # Errors
///
/// Fails if the directory cannot be created or a file cannot be written.
pub fn write_set(
    dir: &Path,
    drawn: &[DrawnReview],
    reviews_per_batch: usize,
) -> Result<DrawReport> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("sample.json"), serde_json::to_vec_pretty(&drawn)?)?;

    let batches = dir.join("batches");
    std::fs::create_dir_all(&batches)?;
    // A smaller draw than last time leaves the tail of the previous one on disk, and those
    // files look exactly like work to hand out.
    if let Ok(entries) = std::fs::read_dir(&batches) {
        for stale in entries.filter_map(std::result::Result::ok) {
            let path = stale.path();
            let named = path.extension().is_some_and(|kind| kind == "json")
                && path
                    .file_stem()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("batch-"));
            if named {
                std::fs::remove_file(&path)?;
            }
        }
    }

    let mut written = 0;
    if reviews_per_batch > 0 {
        for (index, chunk) in drawn.chunks(reviews_per_batch).enumerate() {
            let items: Vec<Handout<'_>> = chunk
                .iter()
                .map(|review| Handout {
                    review_id: &review.id,
                    review: rejoined(review),
                    claims: review
                        .claims
                        .iter()
                        .map(|claim| HandoutClaim {
                            index: claim.index,
                            text: &claim.text,
                        })
                        .collect(),
                })
                .collect();
            std::fs::write(
                batches.join(format!("batch-{index:03}.json")),
                serde_json::to_vec_pretty(&items)?,
            )?;
            written += 1;
        }
    }

    Ok(DrawReport {
        reviews: drawn.len(),
        claims: drawn.iter().map(|review| review.claims.len()).sum(),
        batches: written,
    })
}

/// The review as the labeller reads it, rebuilt from its claims.
///
/// Rebuilt rather than stored a second time: what the labeller sees is exactly what was
/// split, so a claim that reads oddly is visibly the splitter's doing rather than a
/// discrepancy between two copies of the text.
fn rejoined(review: &DrawnReview) -> String {
    review
        .claims
        .iter()
        .map(|claim| claim.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

/// One returned label, as a labeller writes it.
#[derive(Debug, Clone, Deserialize)]
pub struct ReturnedClaimLabel {
    pub review_id: String,
    pub index: u16,
    pub subject: String,
    pub polarity: String,
    pub ironic: bool,
    pub confidence: String,
    pub ambiguous: bool,
    #[serde(default)]
    pub split_wrong: bool,
}

/// One label as it is stored, joined back to what was drawn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimLabel {
    pub review_id: String,
    pub index: u16,
    pub app_id: u32,
    pub language: String,
    pub subset: String,
    pub start: u32,
    pub end: u32,
    pub subject: String,
    pub polarity: String,
    pub ironic: bool,
    pub confidence: String,
    pub ambiguous: bool,
    pub split_wrong: bool,
}

/// What was wrong with a returned set, in the labeller's own terms.
#[derive(Debug, Clone, Default)]
pub struct ClaimIngest {
    pub accepted: usize,
    /// Claims that were drawn and came back with no label. A partly labelled review cannot
    /// say what share of a corpus names no aspect, so this is a failure rather than a gap.
    pub missing: Vec<String>,
    /// Labels naming a claim that was never drawn.
    pub unknown: Vec<String>,
    /// Labels whose subject, polarity or confidence is not one the sheet offers.
    pub rejected: Vec<String>,
}

impl ClaimIngest {
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.missing.is_empty() && self.unknown.is_empty() && self.rejected.is_empty()
    }
}

/// Merges returned label files into a claim reference set.
///
/// # Errors
///
/// Fails if the sample or the returned files cannot be read.
pub fn ingest(dir: &Path, from: &Path) -> Result<(Vec<ClaimLabel>, ClaimIngest)> {
    let drawn: Vec<DrawnReview> =
        serde_json::from_slice(&std::fs::read(dir.join("sample.json")).map_err(|_| {
            crate::Error::NoReferenceSet {
                path: dir.join("sample.json"),
            }
        })?)?;

    let mut wanted: std::collections::HashMap<(String, u16), (&DrawnReview, &DrawnClaim)> =
        std::collections::HashMap::new();
    for review in &drawn {
        for claim in &review.claims {
            wanted.insert((review.id.clone(), claim.index), (review, claim));
        }
    }

    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(from)?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|kind| kind == "json"))
        .collect();
    files.sort();

    let mut report = ClaimIngest::default();
    let mut labels: Vec<ClaimLabel> = Vec::new();
    let mut seen: std::collections::HashSet<(String, u16)> = std::collections::HashSet::new();

    for file in files {
        let returned: Vec<ReturnedClaimLabel> = serde_json::from_slice(&std::fs::read(&file)?)?;
        for label in returned {
            let key = (label.review_id.clone(), label.index);
            let Some((review, claim)) = wanted.get(&key) else {
                report.unknown.push(format!("{}#{}", label.review_id, label.index));
                continue;
            };
            if !crate::taxonomy::CORE_SPINE
                .iter()
                .any(|category| category.id == label.subject)
                || !crate::taxonomy::POLARITY.contains(&label.polarity.as_str())
                || !crate::taxonomy::CONFIDENCE.contains(&label.confidence.as_str())
            {
                report.rejected.push(format!(
                    "{}#{} {} / {} / {}",
                    label.review_id, label.index, label.subject, label.polarity, label.confidence
                ));
                continue;
            }
            if !seen.insert(key) {
                continue;
            }
            labels.push(ClaimLabel {
                review_id: label.review_id,
                index: label.index,
                app_id: review.app_id,
                language: review.language.clone(),
                subset: review.subset.clone(),
                start: claim.start,
                end: claim.end,
                subject: label.subject,
                polarity: label.polarity,
                ironic: label.ironic,
                confidence: label.confidence,
                ambiguous: label.ambiguous,
                split_wrong: label.split_wrong,
            });
        }
    }

    for (id, index) in wanted.keys() {
        if !seen.contains(&(id.clone(), *index)) {
            report.missing.push(format!("{id}#{index}"));
        }
    }
    report.missing.sort();
    report.accepted = labels.len();
    labels.sort_by(|left, right| {
        (left.review_id.as_str(), left.index).cmp(&(right.review_id.as_str(), right.index))
    });

    std::fs::write(dir.join("labels.json"), serde_json::to_vec_pretty(&labels)?)?;
    Ok((labels, report))
}

/// Writes every labelled claim, with its text, as JSONL for training.
///
/// The text is joined here from the drawn sample rather than re-sliced from the capture, so
/// the model is trained on exactly the characters the labeller read. The file it writes holds
/// review text and never leaves the machine: what gets published is the label set, which
/// carries ids and offsets and no text at all.
///
/// # Errors
///
/// Fails if a reference set cannot be read or the destination cannot be written.
pub fn export_training(reference_root: &Path, to: &Path) -> Result<usize> {
    use std::io::Write as _;

    if let Some(parent) = to.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = std::io::BufWriter::new(std::fs::File::create(to)?);
    let mut written = 0;

    let mut sets: Vec<std::path::PathBuf> = std::fs::read_dir(reference_root)
        .into_iter()
        .flatten()
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.join("labels.json").exists())
        .collect();
    sets.sort();

    for set in sets {
        let labels: Vec<ClaimLabel> =
            serde_json::from_slice(&std::fs::read(set.join("labels.json"))?)?;
        let drawn: Vec<DrawnReview> =
            serde_json::from_slice(&std::fs::read(set.join("sample.json"))?)?;

        let mut text: std::collections::HashMap<(&str, u16), &str> =
            std::collections::HashMap::new();
        for review in &drawn {
            for claim in &review.claims {
                text.insert((review.id.as_str(), claim.index), claim.text.as_str());
            }
        }

        for label in &labels {
            let Some(claim) = text.get(&(label.review_id.as_str(), label.index)) else {
                continue;
            };
            let row = serde_json::json!({
                "text": claim,
                "subject": label.subject,
                "polarity": label.polarity,
                "confidence": label.confidence,
                "ambiguous": label.ambiguous,
                "ironic": label.ironic,
                "split_wrong": label.split_wrong,
                "language": label.language,
                "app_id": label.app_id,
                "review_id": label.review_id,
                "claim_index": label.index,
                "subset": label.subset,
            });
            writeln!(out, "{row}")?;
            written += 1;
        }
    }
    out.flush()?;
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn review(id: &str, claims: &[&str]) -> DrawnReview {
        DrawnReview {
            id: id.to_owned(),
            app_id: 1,
            language: "english".to_owned(),
            subset: "random".to_owned(),
            claims: claims
                .iter()
                .enumerate()
                .map(|(index, text)| DrawnClaim {
                    index: u16::try_from(index).unwrap(),
                    start: 0,
                    end: 0,
                    text: (*text).to_owned(),
                })
                .collect(),
        }
    }

    #[test]
    fn a_batch_holds_whole_reviews_so_a_claim_is_never_shown_alone() {
        let dir = std::env::temp_dir().join(format!("census-claimset-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let drawn = vec![
            review("a", &["The combat is superb.", "It runs badly."]),
            review("b", &["gg"]),
        ];
        let report = write_set(&dir, &drawn, 1).unwrap();

        assert_eq!(report.reviews, 2);
        assert_eq!(report.claims, 3);
        assert_eq!(report.batches, 2);

        let first =
            std::fs::read_to_string(dir.join("batches").join("batch-000.json")).unwrap();
        assert!(first.contains("The combat is superb."));
        assert!(
            first.contains("It runs badly."),
            "a review's other claims are what make the first one readable"
        );
        assert!(
            !first.contains("app_id"),
            "a labeller told which game it is can infer what the model cannot"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_drawn_sample_records_the_offsets_rather_than_only_the_text() {
        let dir = std::env::temp_dir().join(format!("census-offsets-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut drawn = review("a", &["Great port.", "Runs at 4k60."]);
        drawn.claims[1].start = 12;
        drawn.claims[1].end = 25;
        write_set(&dir, &[drawn], 8).unwrap();

        let sample = std::fs::read_to_string(dir.join("sample.json")).unwrap();
        assert!(sample.contains("\"start\": 12") && sample.contains("\"end\": 25"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
