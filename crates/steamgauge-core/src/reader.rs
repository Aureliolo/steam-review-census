//! Reading one claim with the trained model.
//!
//! The model says which subject a claim is about, whether it is praise or a complaint, and
//! how sure it is. Below the threshold recorded beside it, it says nothing, and a claim that
//! gets no subject is counted as unclassified rather than filed under whatever scored highest.
//!
//! That last part is the whole reason this module exists. What came before compared a review
//! to twenty-four category prototypes and kept the nearest, which has no way to express "this
//! is about nothing": a review reading "gfg" came back as graphics and art. Abstention is not
//! a nicety here, it is the difference between a mention rate and a rate of nearest matches.

use std::{path::Path, sync::Arc};

use ndarray::Array2;
use ort::{session::Session, value::Tensor};
use serde::Deserialize;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::{Error, Result, taxonomy::CORE_SPINE_VERSION};

/// What a claim does about its subject.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Polarity {
    Praise,
    Complaint,
    #[default]
    Neutral,
}

impl Polarity {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Praise => "praise",
            Self::Complaint => "complaint",
            Self::Neutral => "neutral",
        }
    }

    const fn from_index(index: usize) -> Self {
        match index {
            0 => Self::Praise,
            1 => Self::Complaint,
            _ => Self::Neutral,
        }
    }
}

/// What the model said about one claim.
///
/// The default is "no subject, no confidence", which is what an unfilled slot honestly means
/// and what a claim the model never reached should count as.
#[derive(Debug, Clone, Copy, Default)]
pub struct Reading {
    /// Position in [`crate::taxonomy::CORE_SPINE`], or `None` when nothing cleared the
    /// threshold. `None` is an answer and is counted as one.
    pub subject: Option<usize>,
    /// How sure the model was of its best guess, whether or not it cleared the threshold.
    pub confidence: f32,
    pub polarity: Polarity,
}

/// What was trained, recorded beside the graph so a reader cannot be used blind.
#[derive(Debug, Clone, Deserialize)]
pub struct Provenance {
    pub spine_version: String,
    pub subjects: Vec<String>,
    /// Below this the model says nothing. Chosen on validation claims, never on the held-out
    /// ones, because a threshold tuned against the test set makes the test set an opinion.
    pub threshold: f32,
    /// One line per subject, in `subjects` order, where the export drew them.
    ///
    /// A single threshold keeps the promise on average and breaks it subject by subject,
    /// because the reader is not equally reliable across twenty-six of them. Measured
    /// out-of-fold over twenty-eight games, one line answers `compatibility` predictions at
    /// 0.64 and `vr` at 0.17 while the overall figure sits on 0.75, and a report quoting a
    /// `compatibility` count is quoting that 0.64. A line per subject holds the floor for each
    /// subject's own predictions instead.
    ///
    /// `None` where a subject has no threshold that keeps the promise: that subject is
    /// declined outright, which is the honest answer and not the same as a low one. Absent
    /// entirely on a reader exported before this existed, and then `threshold` governs.
    #[serde(default)]
    pub thresholds: Option<Vec<Option<f32>>>,
    pub max_tokens: usize,
    /// Whether the model was trained on the claim with its review around it. A model trained
    /// one way and read the other is answering a question in a form it has never seen, and
    /// nothing about the output would look wrong, so the graph carries the answer.
    #[serde(default)]
    pub context: bool,
    /// Whether the claim is marked where it sits inside the window, as well as being given as
    /// the first sequence. Travels with the graph for the same reason `context` does: a model
    /// trained to look for the marks and read without them is answering a question in a form
    /// it has never seen, and every answer still looks plausible.
    #[serde(default)]
    pub mark: bool,
    /// Whether the pair is written the way `multilingual-e5-*` was pre-trained on it, as
    /// "query: " and "passage: ". Travels with the graph for the same reason the other two do.
    #[serde(default)]
    pub prefix: bool,
    #[serde(default)]
    pub trained_from: String,
    #[serde(default)]
    pub data_fingerprint: String,
    /// The training run this reader came out of. Every candidate of one generation shares a
    /// backbone and a label set, so this is the only thing that tells two of them apart.
    #[serde(default)]
    pub run_id: String,
    /// The share of claims this model declined on games it never saw. A corpus declined at
    /// far above this is a corpus about something the taxonomy lacks, and the only way a
    /// reader of one game's report can know that is if the reader carries the comparison.
    #[serde(default)]
    pub usual_declined: Option<f32>,
    /// What this model did on the games it never saw. Carried with the graph so that a report
    /// of a game nobody has labelled can still say what its rates are worth: without it the
    /// only honest thing such a report could say is nothing at all.
    #[serde(default)]
    pub frozen: Option<Frozen>,
}

impl Provenance {
    /// The line this subject has to clear, by the model's own class index.
    ///
    /// Infinity where a subject has a line of `None`, so a subject nothing can make reliable
    /// declines every claim rather than answering them at whatever threshold happens to be
    /// lying around.
    #[must_use]
    pub fn line_for(&self, class: usize) -> f32 {
        match self.thresholds.as_ref().and_then(|lines| lines.get(class)) {
            Some(Some(line)) => *line,
            Some(None) => f32::INFINITY,
            None => self.threshold,
        }
    }
}

/// A model's own measurement, on games that chose nothing about it.
#[derive(Debug, Clone, Copy, Deserialize, serde::Serialize)]
pub struct Frozen {
    pub games: u32,
    pub claims: u32,
    /// The share of claims it answered rather than declining.
    pub coverage: f64,
    /// How often it was right about the ones it answered.
    pub accuracy: f64,
    pub macro_f1: f64,
}

/// Where the claim reader lives when nobody has said.
///
/// Beside the encoder in the platform cache, for the same reason: an application opened from
/// a menu has no working directory, and a model found only relative to a checkout is a model
/// only a developer can use. A directory in the working tree wins when there is one, which is
/// how a freshly trained model is tried before it is published.
#[must_use]
pub fn default_dir() -> std::path::PathBuf {
    let local = std::path::PathBuf::from("models").join("claim-reader");
    if local.join("model.onnx").is_file() {
        return local;
    }
    crate::model::default_cache_dir().join("claim-reader")
}

/// The published model, pinned file by file.
///
/// Empty hashes mean nothing has been published yet, and [`ensure`] refuses rather than
/// fetching something unverified: a model file that does not match a pin would change every
/// number the tool reports without anything appearing to go wrong, and "no pin" is not a
/// weaker version of that guarantee, it is its absence.
pub const PUBLISHED: Published = Published {
    repository: "",
    files: [
        crate::model::Asset {
            remote: "model.onnx",
            local: "model.onnx",
            sha256: "",
        },
        crate::model::Asset {
            remote: "tokenizer.json",
            local: "tokenizer.json",
            sha256: "",
        },
        crate::model::Asset {
            remote: "reader.json",
            local: "reader.json",
            sha256: "",
        },
    ],
};

/// A claim reader as published: which repository, and which three files at which hashes.
#[derive(Debug, Clone, Copy)]
pub struct Published {
    pub repository: &'static str,
    files: [crate::model::Asset; 3],
}

impl Published {
    /// Whether anything is pinned at all.
    #[must_use]
    pub fn is_pinned(&self) -> bool {
        !self.repository.is_empty() && self.files.iter().all(|file| !file.sha256.is_empty())
    }
}

/// Fetches the published reader into `dir` unless the copy there already matches its pins.
///
/// # Errors
///
/// Fails if no reader has been published, if a downloaded file does not match its pin, or on
/// transport and filesystem failures.
pub async fn ensure(
    dir: &Path,
    mut on_progress: impl FnMut(crate::model::DownloadProgress),
) -> Result<()> {
    if !PUBLISHED.is_pinned() {
        return Err(Error::NoAnchors {
            path: dir.to_path_buf(),
        });
    }
    std::fs::create_dir_all(dir)?;
    let http = crate::model::client()?;
    for file in PUBLISHED.files {
        crate::model::ensure_asset(&http, PUBLISHED.repository, file, dir, &mut on_progress)
            .await?;
    }
    Ok(())
}

/// The trained model, loaded and ready to read claims.
pub struct ClaimReader {
    session: Session,
    encoder: Arc<Encoder>,
    provenance: Arc<Provenance>,
    order: Vec<usize>,
    device: &'static str,
}

/// Turning claims into the numbers the graph takes: the half of a reading that runs on the
/// processor.
///
/// Apart from the session so that one batch can be prepared while the card works on the batch
/// before it. Tokenising is most of what a reading spends its processor on, and a card that
/// waits for it is a card at half duty.
pub struct Encoder {
    tokenizer: Tokenizer,
    /// The same tokenizer with nothing cut off, for reading a whole review's offsets.
    whole: Tokenizer,
    provenance: Arc<Provenance>,
}

/// A batch tokenised and laid out, waiting for the card.
pub struct Prepared {
    rows: usize,
    cols: usize,
    ids: Vec<i64>,
    mask: Vec<i64>,
}

/// Written by hand for the same reason the reader's is: the tokenizers hold megabytes each.
impl std::fmt::Debug for Encoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Encoder")
            .field("context", &self.provenance.context)
            .field("max_tokens", &self.provenance.max_tokens)
            .finish_non_exhaustive()
    }
}

/// The shape is the part worth seeing; the numbers are a batch of token ids.
impl std::fmt::Debug for Prepared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Prepared")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .finish_non_exhaustive()
    }
}

/// Written by hand because the session and the tokenizer hold megabytes each, and a debug
/// line that prints a model is not a debug line anyone reads.
impl std::fmt::Debug for ClaimReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClaimReader")
            .field("device", &self.device)
            .field("subjects", &self.provenance.subjects.len())
            .field("threshold", &self.provenance.threshold)
            .finish_non_exhaustive()
    }
}

impl ClaimReader {
    /// Loads a reader from a directory holding `model.onnx`, `tokenizer.json` and
    /// `reader.json`.
    ///
    /// # Errors
    ///
    /// Fails if a file is missing, if the graph cannot be loaded on any backend, or if the
    /// model was trained against a different taxonomy from this build. The last of those is
    /// refused rather than worked around: a model that learned twenty categories cannot be
    /// asked about twenty-four, and letting it try would move every number silently.
    pub fn load(dir: &Path) -> Result<Self> {
        let provenance: Provenance =
            serde_json::from_slice(&std::fs::read(dir.join("reader.json")).map_err(|_| {
                Error::NoAnchors {
                    path: dir.join("reader.json"),
                }
            })?)?;

        if provenance.spine_version != CORE_SPINE_VERSION {
            return Err(Error::StaleAnchors {
                field: "taxonomy",
                expected: CORE_SPINE_VERSION.to_owned(),
                actual: provenance.spine_version,
            });
        }

        // The model's classes are in whatever order the training data produced. Mapping them
        // onto taxonomy order here means nothing downstream has to know that they differ.
        let order = provenance
            .subjects
            .iter()
            .map(|id| {
                crate::taxonomy::CORE_SPINE
                    .iter()
                    .position(|category| category.id == id)
                    .ok_or_else(|| Error::StaleAnchors {
                        field: "category",
                        expected: "a category this build has".to_owned(),
                        actual: id.clone(),
                    })
            })
            .collect::<Result<Vec<usize>>>()?;

        let (session, device) = crate::model::session_at(&dir.join("model.onnx"))?;
        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        // Finding where a claim sits in its review means tokenising the whole review, and a
        // tokenizer set to truncate hands back the first `max_tokens` of it. Every offset past
        // that goes missing, the claim is not found among them, and the window falls back to
        // the opening of the review: exactly the head truncation the centring exists to avoid,
        // on exactly the long reviews where it matters. So the offsets come from a tokenizer
        // that cuts nothing, and the pair is encoded by one that cuts to the budget.
        // A tokenizer saved by a trainer carries that trainer's padding and truncation inside
        // the file, so a fresh one is already set to pad to the budget and cut at it. Reading
        // a review's offsets through that gives 128 entries whatever the review says, the tail
        // of them padding at (0, 0), and the claim is then looked for among offsets that do
        // not describe the text. Everything downstream still returns a string.
        let mut whole = tokenizer.clone();
        whole.with_padding(None);
        whole
            .with_truncation(None)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..PaddingParams::default()
        }));
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: provenance.max_tokens,
                ..TruncationParams::default()
            }))
            .map_err(|e| Error::Tokenizer(e.to_string()))?;

        // Proved once, here, rather than trusted. A tokenizer that pads or truncates hands back
        // offsets that do not describe the text, the window is then cut from somewhere the
        // claim is not, and every answer still looks like an answer. This has cost the project
        // a whole measurement once already, in two languages.
        // Short, because padding only shows itself on text shorter than the budget, and long
        // enough to be several tokens whatever the vocabulary.
        let probe = "the quick brown fox jumps over the lazy dog";
        let covered = whole
            .encode(probe, false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let offsets = covered.get_offsets();
        if offsets.last().is_none_or(|&(_, end)| end < probe.len() - 1) {
            return Err(Error::Tokenizer(format!(
                "the tokenizer beside this model does not report where its tokens are: {} bytes \
                 of text came back as {} offsets ending at {:?}. A tokenizer file saved by a \
                 trainer carries that trainer's padding and truncation, and reading a review's \
                 offsets through it gives padding rather than the review.",
                probe.len(),
                offsets.len(),
                offsets.last(),
            )));
        }

        let provenance = Arc::new(provenance);
        Ok(Self {
            session,
            encoder: Arc::new(Encoder {
                tokenizer,
                whole,
                provenance: Arc::clone(&provenance),
            }),
            provenance,
            order,
            device,
        })
    }

    #[must_use]
    pub const fn device(&self) -> &'static str {
        self.device
    }

    #[must_use]
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    /// The half of this reader that runs on the processor, to be used while the card is busy.
    #[must_use]
    pub fn encoder(&self) -> Arc<Encoder> {
        Arc::clone(&self.encoder)
    }

    /// Reads a batch of claims.
    ///
    /// # Errors
    ///
    /// Fails if tokenisation or the forward pass fails.
    pub fn read(&mut self, asked: &[Asked<'_>]) -> Result<Vec<Reading>> {
        if asked.is_empty() {
            return Ok(Vec::new());
        }
        let prepared = self.encoder.prepare(asked)?;
        self.run(prepared)
    }

    /// Runs a batch somebody else tokenised.
    ///
    /// # Errors
    ///
    /// Fails if the forward pass does.
    pub fn run(&mut self, prepared: Prepared) -> Result<Vec<Reading>> {
        let Prepared {
            rows,
            cols,
            ids,
            mask,
        } = prepared;
        if rows == 0 {
            return Ok(Vec::new());
        }

        let outputs = self.session.run(ort::inputs![
            "input_ids" => Tensor::from_array(Array2::from_shape_vec((rows, cols), ids)?)?,
            "attention_mask" => Tensor::from_array(Array2::from_shape_vec((rows, cols), mask)?)?,
        ])?;

        let subject = outputs["subject_logits"]
            .try_extract_array::<f32>()?
            .into_dimensionality::<ndarray::Ix2>()?;
        let polarity = outputs["polarity_logits"]
            .try_extract_array::<f32>()?
            .into_dimensionality::<ndarray::Ix2>()?;

        Ok((0..rows)
            .map(|row| {
                let (best, confidence) = softmax_best(subject.row(row).as_slice().unwrap_or(&[]));
                let (polar, _) = softmax_best(polarity.row(row).as_slice().unwrap_or(&[]));
                Reading {
                    subject: (confidence >= self.provenance.line_for(best))
                        .then(|| self.order.get(best).copied().unwrap_or(best)),
                    confidence,
                    polarity: Polarity::from_index(polar),
                }
            })
            .collect())
    }

    /// The windows this reader would build, for comparing against the trainer's.
    #[must_use]
    pub fn windows_for(&self, asked: &[Asked<'_>]) -> Vec<String> {
        self.encoder.windows_for(asked)
    }
}

impl Encoder {
    /// Tokenises a batch and lays it out as the graph takes it.
    ///
    /// # Errors
    ///
    /// Fails if tokenisation does.
    pub fn prepare(&self, asked: &[Asked<'_>]) -> Result<Prepared> {
        let encodings = if self.provenance.context {
            let windows = self.windows(asked);
            let pairs: Vec<(String, String)> = asked
                .iter()
                .zip(windows)
                .map(|(one, window)| (self.claim(one.claim), self.window_text(&window)))
                .collect();
            self.tokenizer.encode_batch(pairs, true)
        } else {
            let alone: Vec<String> = asked.iter().map(|one| self.claim(one.claim)).collect();
            self.tokenizer.encode_batch(alone, true)
        }
        .map_err(|e| Error::Tokenizer(e.to_string()))?;

        Ok(Prepared {
            rows: encodings.len(),
            cols: encodings.first().map_or(0, |e| e.get_ids().len()),
            ids: encodings
                .iter()
                .flat_map(|e| e.get_ids().iter().map(|&id| i64::from(id)))
                .collect(),
            mask: encodings
                .iter()
                .flat_map(|e| e.get_attention_mask().iter().map(|&m| i64::from(m)))
                .collect(),
        })
    }

    /// The windows this encoder would build, for comparing against the trainer's.
    #[must_use]
    pub fn windows_for(&self, asked: &[Asked<'_>]) -> Vec<String> {
        self.windows(asked)
    }

    fn claim(&self, text: &str) -> String {
        as_query(self.provenance.prefix, text)
    }

    fn window_text(&self, window: &str) -> String {
        as_passage(self.provenance.prefix, window)
    }

    /// Every claim's window, tokenising each review once however many claims it holds.
    ///
    /// Finding where a claim sits means knowing where the review's tokens fall, and that is
    /// the same answer for every claim of one review. Asking per claim tokenised a review of
    /// ten claims ten times, and over a corpus that was most of what the reader spent its
    /// processor on while the card waited.
    fn windows(&self, asked: &[Asked<'_>]) -> Vec<String> {
        let owners = same_review(asked);
        let mut offsets: Vec<Vec<(usize, usize)>> = vec![Vec::new(); asked.len()];
        for (at, &owner) in owners.iter().enumerate() {
            if owner == at {
                offsets[at] = self
                    .whole
                    .encode(asked[at].review, false)
                    .map(|encoded| encoded.get_offsets().to_vec())
                    .unwrap_or_default();
            }
        }
        asked
            .iter()
            .enumerate()
            .map(|(at, one)| self.window(one, &offsets[owners[at]]))
            .collect()
    }

    /// The part of the review the token budget can afford, centred on the claim.
    ///
    /// Truncating a pair from the end spends the whole budget on the opening of the review,
    /// so a claim at the foot of a long one would be read beside paragraphs it is nowhere
    /// near. The budget is the same either way; where it is spent is not, and it is worth
    /// the most immediately around the claim.
    fn window(&self, asked: &Asked<'_>, offsets: &[(usize, usize)]) -> String {
        if offsets.is_empty() {
            return asked.review.to_owned();
        }
        let (opens, closes) = centred(
            offsets,
            asked.at,
            asked.claim.len(),
            self.provenance.max_tokens,
        );
        let kept = asked.review.get(opens..closes).unwrap_or(asked.review);
        if !self.provenance.mark {
            return kept.to_owned();
        }
        let ends = asked.at + asked.claim.len();
        let (Some(before), Some(claim), Some(after)) = (
            asked.review.get(opens..asked.at),
            asked.review.get(asked.at..ends),
            asked.review.get(ends..closes),
        ) else {
            return kept.to_owned();
        };
        format!("{before}{MARK} {claim} {MARK}{after}")
    }
}

/// For each claim, where the first claim of the same review sits in the batch.
///
/// Two live string slices are one review exactly when they start at the same byte and run the
/// same length, and every claim of a review is handed the same slice of the same text.
fn same_review(asked: &[Asked<'_>]) -> Vec<usize> {
    let mut first: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    asked
        .iter()
        .enumerate()
        .map(|(at, one)| {
            *first
                .entry((one.review.as_ptr() as usize, one.review.len()))
                .or_insert(at)
        })
        .collect()
}

/// The claim as the first sequence, in the form the model was trained on.
///
/// `multilingual-e5-*` was pre-trained on pairs written as "query: " and "passage: ", and a
/// model fine-tuned that way is answering a differently shaped question without them. Matches
/// `Claims.pair` in `training/train.py`, which is the only other place a pair is written.
fn as_query(prefix: bool, text: &str) -> String {
    if prefix {
        format!("query: {text}")
    } else {
        text.to_owned()
    }
}

fn as_passage(prefix: bool, window: &str) -> String {
    if prefix {
        format!("passage: {window}")
    } else {
        window.to_owned()
    }
}

/// What marks the claim inside its window, matching `training/train.py`.
///
/// Two characters no reviewer writes and the tokenizer already knows: a token added to the
/// vocabulary starts from noise, and this one has to mean something after four thousand
/// training claims rather than four hundred thousand.
const MARK: &str = "**";

/// The bytes of a review to keep, given where its tokens fall and where the claim sits.
fn centred(offsets: &[(usize, usize)], at: usize, length: usize, budget: usize) -> (usize, usize) {
    let ends = at + length;
    let first = offsets.iter().position(|&(_, end)| end > at).unwrap_or(0);
    let last = offsets
        .iter()
        .position(|&(start, _)| start >= ends)
        .unwrap_or(offsets.len());
    // Four special tokens on a pair, and the claim is spent twice: once as the first
    // sequence, and again where it sits inside the window.
    let spare = budget.saturating_sub(2 * last.saturating_sub(first) + 4) / 2;
    (
        offsets[first.saturating_sub(spare)].0,
        offsets[(last + spare).clamp(1, offsets.len()) - 1].1,
    )
}

/// A claim and the review around it, as the labeller saw the pair.
#[derive(Debug, Clone, Copy)]
pub struct Asked<'a> {
    pub claim: &'a str,
    /// The claims of the review joined back together: what the labeller read, and so what
    /// the model was trained against. The capture's own text carries the markup and list
    /// bullets the splitter removed, and asking the model about that instead would put the
    /// question in a form it has never seen. Ignored by a model that reads claims alone.
    pub review: &'a str,
    /// Where `claim` starts in `review`.
    pub at: usize,
}

/// The best class and its probability, from logits.
///
/// The maximum is subtracted before exponentiating for the usual reason: a logit of 90 is
/// entirely possible and `exp(90)` is not.
fn softmax_best(logits: &[f32]) -> (usize, f32) {
    let mut best = 0;
    let mut highest = f32::NEG_INFINITY;
    for (index, value) in logits.iter().enumerate() {
        if *value > highest {
            highest = *value;
            best = index;
        }
    }
    if logits.is_empty() {
        return (0, 0.0);
    }
    let total: f32 = logits.iter().map(|value| (value - highest).exp()).sum();
    (best, if total > 0.0 { 1.0 / total } else { 0.0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provenance(lines: Option<Vec<Option<f32>>>) -> Provenance {
        let mut written = serde_json::json!({
            "spine_version": crate::CORE_SPINE_VERSION,
            "subjects": ["verdict", "vr", "licensing"],
            "threshold": 0.69,
            "max_tokens": 128,
        });
        if let Some(lines) = lines {
            written["thresholds"] = serde_json::to_value(lines).expect("lines are values");
        }
        serde_json::from_value(written).expect("a provenance")
    }

    #[test]
    fn a_reader_exported_before_the_lines_existed_keeps_its_one_line() {
        let found = provenance(None);
        assert!((found.line_for(0) - 0.69).abs() < f32::EPSILON);
        assert!((found.line_for(2) - 0.69).abs() < f32::EPSILON);
    }

    #[test]
    fn a_subject_no_threshold_can_make_reliable_answers_nothing() {
        // Not a high line but no line: the export writes `null` where no threshold over the
        // validation games keeps the promise for that subject, and a claim of that subject has
        // to be declined rather than answered at whatever number was lying nearest.
        let found = provenance(Some(vec![Some(0.55), Some(0.94), None]));
        assert!((found.line_for(0) - 0.55).abs() < f32::EPSILON);
        assert!((found.line_for(1) - 0.94).abs() < f32::EPSILON);
        assert!(found.line_for(2).is_infinite());
        assert!(1.0_f32 < found.line_for(2), "certainty must not clear it");
    }

    #[test]
    fn a_class_the_lines_do_not_reach_falls_back_rather_than_answering_everything() {
        let found = provenance(Some(vec![Some(0.55)]));
        assert!((found.line_for(2) - 0.69).abs() < f32::EPSILON);
    }

    #[test]
    fn offsets_that_do_not_describe_the_review_give_a_window_of_nothing() {
        // Not behaviour to rely on: a warning about where the offsets have to come from. A
        // tokenizer file saved by a trainer carries that trainer's padding and truncation, so
        // a review read through it comes back as exactly `max_tokens` entries whatever it says,
        // the tail of them padding at (0, 0). The claim is then looked for among offsets that
        // describe nothing, and this hands back an empty window or the whole review: both are
        // still strings, neither is what the model was trained on, and every answer still looks
        // plausible. The reader keeps a second tokenizer with padding and truncation off for
        // exactly this reason.
        let real: Vec<(usize, usize)> = (0..10).map(|token| (token * 5, token * 5 + 4)).collect();
        let padded: Vec<(usize, usize)> = real
            .iter()
            .copied()
            .chain(std::iter::repeat_n((0, 0), 118))
            .collect();

        assert_eq!(
            centred(&padded, 0, 20, 128),
            (0, 0),
            "padding at the end closes the window onto nothing"
        );
        assert_eq!(
            centred(&real, 0, 20, 128),
            (0, 49),
            "the same claim, through offsets that are the review's"
        );
        // A claim past the offsets given is read from the start of the review instead, which is
        // the shape the same mistake takes when the tokenizer truncates rather than pads.
        assert_eq!(centred(&real, 500, 10, 128), (0, 49));
    }

    #[test]
    fn the_claims_of_one_review_share_the_tokens_of_that_review_and_no_other() {
        let first: std::sync::Arc<str> = std::sync::Arc::from("It runs badly. Worth the money.");
        let second: std::sync::Arc<str> = std::sync::Arc::from("It runs badly. Worth the money.");
        let asked = vec![
            Asked {
                claim: "It runs badly.",
                review: &first,
                at: 0,
            },
            Asked {
                claim: "Worth the money.",
                review: &second,
                at: 0,
            },
            Asked {
                claim: "Worth the money.",
                review: &first,
                at: 15,
            },
        ];

        // The second review reads the same and is a different review: a copy of a text is
        // tokenised again rather than sharing an answer with something it only resembles.
        assert_eq!(same_review(&asked), vec![0, 1, 0]);
    }

    #[test]
    fn the_best_class_comes_back_with_a_probability_rather_than_a_logit() {
        let (best, confidence) = softmax_best(&[0.0, 4.0, 1.0]);
        assert_eq!(best, 1);
        assert!(
            (confidence - 0.936).abs() < 0.01,
            "expected roughly 0.94, got {confidence}"
        );
    }

    #[test]
    fn a_flat_answer_is_reported_as_unsure_rather_than_as_a_choice() {
        let (_, confidence) = softmax_best(&[1.0; 24]);
        assert!(
            (confidence - 1.0 / 24.0).abs() < 1e-5,
            "twenty-four equal logits is one twenty-fourth, not a decision"
        );
    }

    #[test]
    fn enormous_logits_do_not_overflow_into_nothing() {
        let (best, confidence) = softmax_best(&[90.0, 10.0]);
        assert_eq!(best, 0);
        assert!(confidence.is_finite() && confidence > 0.99);
    }

    #[test]
    fn a_pin_with_any_hash_missing_is_no_pin_at_all() {
        let mut half = PUBLISHED;
        half.repository = "someone/claim-reader";
        half.files[0].sha256 = "0".repeat(64).leak();
        half.files[1].sha256 = "0".repeat(64).leak();
        assert!(
            !half.is_pinned(),
            "two of three files pinned would fetch the third unverified"
        );
        half.files[2].sha256 = "0".repeat(64).leak();
        assert!(half.is_pinned());
    }

    /// One token a word, which is close enough to make the arithmetic readable.
    fn words(review: &str) -> Vec<(usize, usize)> {
        let mut offsets = Vec::new();
        let mut at = 0;
        for word in review.split(' ') {
            offsets.push((at, at + word.len()));
            at += word.len() + 1;
        }
        offsets
    }

    /// The trainer writes the pair one way and the reader has to write it the same way. This
    /// has already cost the project a night once, through a tokenizer rather than a prefix.
    #[test]
    fn a_pair_is_written_the_way_the_model_was_trained_on_it() {
        assert_eq!(as_query(false, "it crashes"), "it crashes");
        assert_eq!(as_passage(false, "the review"), "the review");
        assert_eq!(as_query(true, "it crashes"), "query: it crashes");
        assert_eq!(as_passage(true, "the review"), "passage: the review");
    }

    #[test]
    fn the_window_is_centred_on_the_claim_not_the_start_of_the_review() {
        let review = (0..40)
            .map(|n| format!("w{n}"))
            .collect::<Vec<_>>()
            .join(" ");
        let offsets = words(&review);
        let (at, length) = (offsets[30].0, offsets[30].1 - offsets[30].0);

        let (opens, closes) = centred(&offsets, at, length, 12);
        let kept = &review[opens..closes];
        assert!(
            kept.contains("w30"),
            "the claim itself must be in its own window"
        );
        assert!(
            kept.contains("w28") && kept.contains("w32"),
            "a budget of twelve tokens should reach either side, got {kept:?}"
        );
        assert!(
            !kept.contains("w0 ") && !kept.contains("w39"),
            "the window must not run to the ends of the review, got {kept:?}"
        );
    }

    #[test]
    fn the_mark_goes_round_the_claim_and_nothing_else() {
        // Written out rather than computed, because the one thing this has to match is a
        // string built by `training/train.py`, and a test that computes it the same way the
        // code does would agree with a mistake.
        let review = "before it. the claim itself. after it.";
        let at = review.find("the claim").unwrap();
        let claim = "the claim itself.";
        let marked = format!(
            "{}{MARK} {claim} {MARK}{}",
            &review[..at],
            &review[at + claim.len()..]
        );
        assert_eq!(marked, "before it. ** the claim itself. ** after it.");
    }

    #[test]
    fn a_budget_that_fits_the_whole_review_keeps_all_of_it() {
        let review = "one two three four five";
        let offsets = words(review);
        let (opens, closes) = centred(&offsets, offsets[2].0, 5, 512);
        assert_eq!(&review[opens..closes], review);
    }

    #[test]
    fn a_claim_longer_than_the_budget_still_yields_a_window() {
        // The spare is zero here and the arithmetic must not run off either end.
        let review = "one two three four five";
        let offsets = words(review);
        let (opens, closes) = centred(&offsets, 0, review.len(), 4);
        assert!(opens < closes && closes <= review.len());
    }

    #[tokio::test]
    async fn nothing_is_fetched_until_something_is_published() {
        // The empty pin must refuse rather than reach for the network. A refusal that names
        // the directory is what the CLI turns into "train one, or pass --model".
        if PUBLISHED.is_pinned() {
            return;
        }
        let dir = std::env::temp_dir().join(format!("steamgauge-unpinned-{}", std::process::id()));
        let refused = ensure(&dir, |_| {}).await;
        assert!(matches!(refused, Err(Error::NoAnchors { .. })));
        assert!(!dir.exists(), "a refused fetch must leave nothing behind");
    }
}
