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
    /// Which claims the labeller is asked about, where that is not all of them. The rest are
    /// still handed over, because they are the review: a claim reading "it doesn't" cannot be
    /// labelled without the sentence before it, and a draw that asks about one claim in a
    /// review must still show the review.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asked: Option<Vec<u16>>,
}

impl DrawnReview {
    /// Whether the labeller is asked about this claim, rather than shown it for context.
    #[must_use]
    pub fn asks(&self, index: u16) -> bool {
        self.asked
            .as_ref()
            .is_none_or(|asked| asked.contains(&index))
    }

    /// How many claims of this review are being asked about.
    #[must_use]
    pub fn asked_count(&self) -> usize {
        self.asked.as_ref().map_or(self.claims.len(), Vec::len)
    }
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
pub fn draw(
    out_dir: &Path,
    app_id: u32,
    wanted: usize,
    english_share: f64,
    seed: u64,
) -> Result<Vec<DrawnReview>> {
    let snapshot = crate::embed::latest_snapshot(out_dir, app_id)?;
    // Two draws rather than one, so the mix is decided rather than inherited. A corpus is
    // whatever languages its players write in, and drawing straight from it would train the
    // model mostly on whichever one that happens to be. The reports default to English and
    // the model has to hold up in the rest, so the split is set here and recorded.
    #[expect(
        clippy::cast_precision_loss,
        reason = "a sample is hundreds of reviews, not 2^53 of them"
    )]
    #[expect(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "english_share is a share and wanted is a count"
    )]
    let english_wanted = (wanted as f64 * english_share.clamp(0.0, 1.0)).round() as usize;
    let mut english: crate::bounded::Smallest<[u8; 32], DrawnReview> =
        crate::bounded::Smallest::new(english_wanted);
    let mut rest: crate::bounded::Smallest<[u8; 32], DrawnReview> =
        crate::bounded::Smallest::new(wanted.saturating_sub(english_wanted));

    crate::capture::for_each_body(&snapshot, |id, language, text| {
        let found = crate::claims::claims_of(text);
        if found.is_empty() {
            return Ok(());
        }
        let claims = found
            .into_iter()
            .enumerate()
            .map(|(index, (at, claim))| DrawnClaim {
                index: u16::try_from(index).unwrap_or(u16::MAX),
                start: u32::try_from(at.start).unwrap_or(u32::MAX),
                end: u32::try_from(at.end).unwrap_or(u32::MAX),
                text: claim.into_owned(),
            })
            .collect();
        let drawn = DrawnReview {
            id: id.to_owned(),
            app_id,
            language: language.to_owned(),
            subset: "random".to_owned(),
            claims,
            asked: None,
        };
        let key = crate::bounded::rank(seed, "claims", id);
        if language == "english" {
            english.offer(key, drawn);
        } else {
            rest.offer(key, drawn);
        }
        Ok(())
    })?;

    let mut drawn = english.take();
    drawn.extend(rest.take());
    drawn.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(drawn)
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
                        .filter(|claim| review.asks(claim.index))
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
        claims: drawn.iter().map(DrawnReview::asked_count).sum(),
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

/// Which sheet a labeller was working from, and which cut of the claims they were shown.
///
/// Not the versions this build holds. A labeller handed a set on Monday answers Monday's
/// sheet however long they take, and a revision that lands while they work does not reach
/// back and change what they were asked. Stamping the build's version at ingest is how a
/// set comes to claim it was labelled against rules its labeller never saw.
#[derive(Debug, Clone)]
pub struct Sheet {
    pub splitter: String,
    pub taxonomy: String,
    /// The labeller. Named rather than defaulted: the first set written by a second model is
    /// the one where a default would be wrong, and it is also the one nobody would think to
    /// check.
    pub produced_by: String,
}

impl Default for Sheet {
    /// What this build would hand out now, which is right for a set drawn and labelled
    /// without a revision in between.
    fn default() -> Self {
        Self {
            splitter: crate::claims::SPLITTER_VERSION.to_owned(),
            taxonomy: crate::CORE_SPINE_VERSION.to_owned(),
            produced_by: String::new(),
        }
    }
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
    /// Which splitter cut the claim this label was written about, and which taxonomy its
    /// subject comes from. A published row is joined back by its span and read against a
    /// category sheet, and neither means anything without the version it was made under.
    #[serde(default)]
    pub splitter: String,
    #[serde(default)]
    pub taxonomy: String,
    /// Which labeller wrote it. A set labelled by two models is not one set: they disagree
    /// with each other about as often as they disagree with the truth, and pooling them
    /// without saying so turns a measurable difference into noise nobody can find again.
    #[serde(default)]
    pub produced_by: String,
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
    /// Labels whose subject the revision moved. The measure of what a rule change was worth:
    /// a rule that moves nothing was already understood, and one that moves everything was
    /// not a clarification.
    pub moved: u32,
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

/// Draws the share of an already-drawn set that a second labeller should read.
///
/// A set labelled once cannot say anything about its own reliability, so a tenth of it is
/// labelled again by a different labeller and the two are compared. The same reviews, drawn by
/// hash so that the choice depends on the review rather than on the order of the file: a set
/// redrawn after more games are added asks for a second opinion on the same reviews it already
/// has one for.
///
/// The purpose string differs from the one that drew the set, so this is not the front of the
/// original draw. Taking the first tenth of an ordering the set was already built from would
/// ask the second labeller about a systematically unrepresentative slice.
///
/// # Errors
///
/// Fails if the set has no drawn sample, or the sample cannot be read.
pub fn draw_second(dir: &Path, share: f64, seed: u64) -> Result<Vec<DrawnReview>> {
    let drawn: Vec<DrawnReview> =
        serde_json::from_slice(&std::fs::read(dir.join("sample.json")).map_err(|_| {
            crate::Error::NoReferenceSet {
                path: dir.join("sample.json"),
            }
        })?)?;

    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a reference set is a few hundred reviews"
    )]
    let wanted = ((drawn.len() as f64) * share.clamp(0.0, 1.0)).round() as usize;

    let mut ranked: Vec<([u8; 32], DrawnReview)> = drawn
        .into_iter()
        .map(|review| {
            (
                crate::bounded::rank(seed, "second-opinion", &review.id),
                review,
            )
        })
        .collect();
    ranked.sort_by_key(|(key, _)| *key);
    Ok(ranked
        .into_iter()
        .take(wanted)
        .map(|(_, review)| review)
        .collect())
}

/// Draws the claims of a labelled set that a revision of the sheet puts back in question.
///
/// A taxonomy revision does not invalidate a set: the subjects that survive it mean what they
/// meant, and re-asking about all of them costs what the set cost. What it does is move
/// boundaries, and only claims near a moved one can change. This finds them by what they say,
/// because a claim about a subject the sheet has just learned to name almost always says its
/// name: eight games' worth of "modding" sat under `content` and `updates` without a single
/// one of them failing to use the word.
///
/// Matched by the same word cut that counts the terms a report shows, so "mod" finds "mod"
/// and "mods" and not "modern", and a caller need not know how a word is bounded in a script
/// that writes without spaces.
///
/// Narrowed further by which subjects a claim is currently filed under, where the caller
/// names any: a rule moves a boundary between two rows, and a claim on neither side of it
/// cannot cross. "Worth" appears in claims about eleven subjects and the rule about what a
/// thing is worth paying touches two of them.
///
/// Returns the claims that matched, as a set the labeller reads exactly like a fresh one.
///
/// # Errors
///
/// Fails if the set has no drawn sample or labels, or they cannot be read.
pub fn draw_revisit(dir: &Path, words: &[String], subjects: &[String]) -> Result<Vec<DrawnReview>> {
    let drawn: Vec<DrawnReview> =
        serde_json::from_slice(&std::fs::read(dir.join("sample.json")).map_err(|_| {
            crate::Error::NoReferenceSet {
                path: dir.join("sample.json"),
            }
        })?)?;
    let labels: Vec<ClaimLabel> =
        serde_json::from_slice(&std::fs::read(dir.join("labels.json")).map_err(|_| {
            crate::Error::NoReferenceSet {
                path: dir.join("labels.json"),
            }
        })?)?;
    let labelled: std::collections::HashSet<(&str, u16)> = labels
        .iter()
        .filter(|label| subjects.is_empty() || subjects.contains(&label.subject))
        .map(|label| (label.review_id.as_str(), label.index))
        .collect();

    Ok(drawn
        .iter()
        .filter_map(|review| {
            // The whole review goes to the labeller, as it did the first time, but only the
            // claims in question are asked about: a claim shown without the rest of its
            // review is a claim nobody can label, and a claim nobody asked about is one
            // whose existing label stands.
            let asked: Vec<u16> = review
                .claims
                .iter()
                .filter(|claim| labelled.contains(&(review.id.as_str(), claim.index)))
                .filter(|claim| {
                    words
                        .iter()
                        .any(|word| crate::said::mentions(&claim.text, word))
                })
                .map(|claim| claim.index)
                .collect();
            (!asked.is_empty()).then(|| DrawnReview {
                asked: Some(asked),
                ..review.clone()
            })
        })
        .collect())
}

/// Sets beside a game's random draw that add claims to train on rather than answers to
/// compare. Every one of them is labelled as its own `subset`, and no prevalence figure
/// counts a row from any of them.
pub const TEACHING_SETS: &[&str] = &["declined"];

/// Draws the claims the reader would not answer, as a set to teach it on.
///
/// Every set so far is a random draw, which is what makes prevalence measurable and is the
/// right default: a set drawn for being hard cannot say what share of a corpus mentions
/// price. But once a reader exists, the claims it abstains on are worth several times a random
/// claim to label, because a random draw spends most of its budget confirming answers the
/// model already gets right.
///
/// So this is a teaching draw and it is marked as one. Every review it produces carries
/// `subset: "declined"`, which keeps it out of every prevalence figure and out of the
/// validation and frozen games entirely: a model measured on the claims it was known to find
/// hard would report an accuracy nobody can interpret.
///
/// Drawn uniformly from the declined claims rather than from the least confident of them. The
/// bottom of a confidence ordering is mostly text with nothing in it, and a set of that
/// teaches the model to say `offtopic` rather than to read anything.
///
/// A review already in the game's reference set is never drawn, so no review is labelled twice
/// under two different draws.
///
/// # Errors
///
/// Fails if there is no capture, no reading, or the reading was cut by another splitter.
pub fn draw_declined(
    out_dir: &Path,
    app_id: u32,
    dir: &Path,
    wanted: usize,
    seed: u64,
) -> Result<Vec<DrawnReview>> {
    let snapshot = crate::embed::latest_snapshot(out_dir, app_id)?;
    let reading: crate::read::ReadReport =
        serde_json::from_slice(&std::fs::read(snapshot.join("reading.json")).map_err(|_| {
            crate::Error::NoClassifications {
                path: snapshot.join("reading.json"),
            }
        })?)?;
    reading.cut_as_this_build()?;

    let drawn_already: Vec<DrawnReview> = std::fs::read(dir.join("sample.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default();
    let already: std::collections::HashSet<&str> = drawn_already
        .iter()
        .map(|review| review.id.as_str())
        .collect();

    let mut chosen: crate::bounded::Smallest<[u8; 32], (String, u16)> =
        crate::bounded::Smallest::new(wanted);
    crate::read::for_each_reading(
        &snapshot.join("readings.parquet"),
        |id, index, subject, _, _| {
            if subject.is_some() || already.contains(id) {
                return;
            }
            let key = crate::bounded::rank(seed, "declined", &format!("{id}\u{0}{index}"));
            chosen.offer(key, (id.to_owned(), index));
        },
    )?;

    let mut picks: std::collections::HashMap<String, Vec<u16>> = std::collections::HashMap::new();
    for (id, index) in chosen.take() {
        picks.entry(id).or_default().push(index);
    }

    let depth = reading.depth;
    let mut drawn = Vec::new();
    crate::capture::for_each_body(&snapshot, |id, language, text| {
        let Some(asked) = picks.get(id) else {
            return Ok(());
        };
        let spans = depth.spans_of(text);
        let claims: Vec<DrawnClaim> = depth
            .claims_of(text)
            .into_iter()
            .enumerate()
            .zip(spans)
            .map(|((index, claim), span)| DrawnClaim {
                index: u16::try_from(index).unwrap_or(u16::MAX),
                start: u32::try_from(span.start).unwrap_or(u32::MAX),
                end: u32::try_from(span.end).unwrap_or(u32::MAX),
                text: claim.into_owned(),
            })
            .collect();
        // The corpus can have moved on since it was read: a review edited between the two is
        // cut into different claims, and the index the reader declined then names a different
        // sentence now.
        let mut asked: Vec<u16> = asked
            .iter()
            .copied()
            .filter(|index| usize::from(*index) < claims.len())
            .collect();
        asked.sort_unstable();
        if asked.is_empty() {
            return Ok(());
        }
        drawn.push(DrawnReview {
            id: id.to_owned(),
            app_id,
            language: language.to_owned(),
            subset: "declined".to_owned(),
            claims,
            asked: Some(asked),
        });
        Ok(())
    })?;

    drawn.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(drawn)
}

/// Replaces the labels of the revisited claims, leaving every other label alone.
///
/// The returned labels are the answers under the sheet as it now reads, so they carry the
/// current taxonomy while the rest of the set carries the one it was labelled under. A set
/// where every row claimed the current version would be a set that had quietly relabelled
/// itself.
///
/// # Errors
///
/// Fails if the set or the returned files cannot be read or written.
pub fn ingest_revisit(dir: &Path, from: &Path, by: &str) -> Result<ClaimIngest> {
    let mut labels: Vec<ClaimLabel> =
        serde_json::from_slice(&std::fs::read(dir.join("labels.json")).map_err(|_| {
            crate::Error::NoReferenceSet {
                path: dir.join("labels.json"),
            }
        })?)?;
    let mut held: std::collections::HashMap<(String, u16), &mut ClaimLabel> = labels
        .iter_mut()
        .map(|label| ((label.review_id.clone(), label.index), label))
        .collect();

    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(from)?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|kind| kind == "json"))
        .collect();
    files.sort();

    let mut report = ClaimIngest::default();
    let mut changed = 0;
    for file in files {
        let returned: Vec<ReturnedClaimLabel> = serde_json::from_slice(&std::fs::read(&file)?)?;
        for answer in returned {
            let Some(label) = held.get_mut(&(answer.review_id.clone(), answer.index)) else {
                report
                    .unknown
                    .push(format!("{}#{}", answer.review_id, answer.index));
                continue;
            };
            if !crate::taxonomy::CORE_SPINE
                .iter()
                .any(|category| category.id == answer.subject)
                || !crate::taxonomy::POLARITY.contains(&answer.polarity.as_str())
                || !crate::taxonomy::CONFIDENCE.contains(&answer.confidence.as_str())
            {
                report.rejected.push(format!(
                    "{}#{} {} / {} / {}",
                    answer.review_id,
                    answer.index,
                    answer.subject,
                    answer.polarity,
                    answer.confidence
                ));
                continue;
            }
            changed += u32::from(label.subject != answer.subject);
            label.subject = answer.subject;
            label.polarity = answer.polarity;
            label.ironic = answer.ironic;
            label.confidence = answer.confidence;
            label.ambiguous = answer.ambiguous;
            label.split_wrong = answer.split_wrong;
            crate::CORE_SPINE_VERSION.clone_into(&mut label.taxonomy);
            // A revisited label is a new answer from whoever gave it, not a correction of the
            // first labeller's, so it carries the second labeller's name.
            by.clone_into(&mut label.produced_by);
            report.accepted += 1;
        }
    }

    std::fs::write(dir.join("labels.json"), serde_json::to_vec_pretty(&labels)?)?;
    report.moved = changed;
    Ok(report)
}

/// Where the claim reference sets live.
#[must_use]
pub fn reference_root() -> std::path::PathBuf {
    std::path::PathBuf::from("reference").join("claims")
}

/// Where one game's claim reference set lives.
#[must_use]
pub fn default_reference_dir(app_id: u32) -> std::path::PathBuf {
    reference_root().join(app_id.to_string())
}

/// Merges returned label files into a claim reference set.
///
/// # Errors
///
/// Fails if the sample or the returned files cannot be read.
pub fn ingest(dir: &Path, from: &Path, sheet: &Sheet) -> Result<(Vec<ClaimLabel>, ClaimIngest)> {
    let drawn: Vec<DrawnReview> =
        serde_json::from_slice(&std::fs::read(dir.join("sample.json")).map_err(|_| {
            crate::Error::NoReferenceSet {
                path: dir.join("sample.json"),
            }
        })?)?;

    // Only the claims the draw asks about. The rest of a review is in the handout because a
    // claim cannot be read without it, and a label on one of those answers a question nobody
    // put: it would land in the set as though it had been drawn.
    let mut wanted: std::collections::HashMap<(String, u16), (&DrawnReview, &DrawnClaim)> =
        std::collections::HashMap::new();
    for review in &drawn {
        for claim in review
            .claims
            .iter()
            .filter(|claim| review.asks(claim.index))
        {
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
                report
                    .unknown
                    .push(format!("{}#{}", label.review_id, label.index));
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
                splitter: sheet.splitter.clone(),
                taxonomy: sheet.taxonomy.clone(),
                produced_by: sheet.produced_by.clone(),
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

    // A game's directory holds its random draw, and beside it the teaching draws named here.
    // Named rather than "every subdirectory holding labels": `second/` holds a second
    // labeller's answers to claims the set already has, and sweeping those in would train the
    // model on the same claim twice, on both answers wherever the two labellers disagreed.
    let mut sets: Vec<std::path::PathBuf> = std::fs::read_dir(reference_root)
        .into_iter()
        .flatten()
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .flat_map(|game| {
            let teaching: Vec<std::path::PathBuf> =
                TEACHING_SETS.iter().map(|name| game.join(name)).collect();
            std::iter::once(game).chain(teaching)
        })
        .filter(|path| path.join("labels.json").exists() && path.join("sample.json").exists())
        .collect();
    sets.sort();

    for set in sets {
        let labels: Vec<ClaimLabel> =
            serde_json::from_slice(&std::fs::read(set.join("labels.json"))?)?;
        let drawn: Vec<DrawnReview> =
            serde_json::from_slice(&std::fs::read(set.join("sample.json"))?)?;

        // The claim, and where it starts in the review around it. Searching for the text
        // instead would find the first copy of "Great game." in a review that says it twice,
        // and centre the window on the wrong half of the review.
        let mut text: std::collections::HashMap<(&str, u16), (&str, usize)> =
            std::collections::HashMap::new();
        // The review around each claim, rebuilt exactly as the labeller was shown it. A
        // labeller reads "it doesn't" with the sentence before it; a model given the claim
        // alone is being asked a question nobody could answer, and the gap between what the
        // two saw is measurable error attributed to the model.
        let mut around: std::collections::HashMap<&str, String> = std::collections::HashMap::new();
        for review in &drawn {
            let mut at = 0;
            for claim in &review.claims {
                text.insert((review.id.as_str(), claim.index), (claim.text.as_str(), at));
                at += claim.text.len() + 1;
            }
            around.insert(review.id.as_str(), rejoined(review));
        }

        for label in &labels {
            let Some(&(claim, at)) = text.get(&(label.review_id.as_str(), label.index)) else {
                continue;
            };
            let row = serde_json::json!({
                "text": claim,
                "review": around.get(label.review_id.as_str()),
                "review_offset": at,
                "subject": label.subject,
                "polarity": label.polarity,
                "confidence": label.confidence,
                "ambiguous": label.ambiguous,
                "ironic": label.ironic,
                "split_wrong": label.split_wrong,
                "produced_by": label.produced_by,
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
            asked: None,
        }
    }

    #[test]
    fn a_batch_holds_whole_reviews_so_a_claim_is_never_shown_alone() {
        let dir = std::env::temp_dir().join(format!("steamgauge-claimset-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let drawn = vec![
            review("a", &["The combat is superb.", "It runs badly."]),
            review("b", &["gg"]),
        ];
        let report = write_set(&dir, &drawn, 1).unwrap();

        assert_eq!(report.reviews, 2);
        assert_eq!(report.claims, 3);
        assert_eq!(report.batches, 2);

        let first = std::fs::read_to_string(dir.join("batches").join("batch-000.json")).unwrap();
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
    fn a_draw_that_asks_about_one_claim_still_hands_over_the_whole_review() {
        let dir = std::env::temp_dir().join(format!("steamgauge-asked-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut drawn = review("a", &["Bought it for the mod support.", "It doesn't."]);
        drawn.asked = Some(vec![1]);
        let report = write_set(&dir, &[drawn], 8).unwrap();

        assert_eq!(report.claims, 1, "one claim was asked about, not two");
        let batch = std::fs::read_to_string(dir.join("batches").join("batch-000.json")).unwrap();
        assert!(
            batch.contains("Bought it for the mod support. It doesn't."),
            "the claim nobody is asked about is the only thing that makes the other readable"
        );
        let handed: serde_json::Value = serde_json::from_str(&batch).unwrap();
        let claims = handed[0]["claims"].as_array().unwrap();
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0]["index"], 1, "the index is the one in the review");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_label_on_a_claim_nobody_asked_about_does_not_join_the_set() {
        let dir = std::env::temp_dir().join(format!("steamgauge-unasked-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut drawn = review("a", &["Bought it for the mod support.", "It doesn't."]);
        drawn.asked = Some(vec![1]);
        write_set(&dir, &[drawn], 8).unwrap();

        let returned = dir.join("returned");
        std::fs::create_dir_all(&returned).unwrap();
        std::fs::write(
            returned.join("batch-000.json"),
            serde_json::to_vec(&serde_json::json!([
                {"review_id": "a", "index": 1, "subject": "mods", "polarity": "complaint",
                 "ironic": false, "confidence": "high", "ambiguous": false, "split_wrong": false},
                {"review_id": "a", "index": 0, "subject": "mods", "polarity": "praise",
                 "ironic": false, "confidence": "high", "ambiguous": false, "split_wrong": false},
            ]))
            .unwrap(),
        )
        .unwrap();

        let sheet = Sheet {
            produced_by: "a test".to_owned(),
            ..Default::default()
        };
        let (labels, report) = ingest(&dir, &returned, &sheet).unwrap();

        assert_eq!(labels.len(), 1, "only the claim that was asked about");
        assert_eq!(labels[0].index, 1);
        assert_eq!(report.unknown, vec!["a#0".to_owned()]);
        assert!(
            report.missing.is_empty(),
            "a claim shown for context is not a claim that came back unlabelled"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_drawn_sample_records_the_offsets_rather_than_only_the_text() {
        let dir = std::env::temp_dir().join(format!("steamgauge-offsets-{}", std::process::id()));
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
