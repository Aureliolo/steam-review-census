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

use std::path::Path;

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
    pub max_tokens: usize,
    #[serde(default)]
    pub trained_from: String,
    #[serde(default)]
    pub data_fingerprint: String,
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

/// The trained model, loaded and ready to read claims.
pub struct ClaimReader {
    session: Session,
    tokenizer: Tokenizer,
    provenance: Provenance,
    order: Vec<usize>,
    device: &'static str,
}

impl std::fmt::Debug for ClaimReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClaimReader")
            .field("device", &self.device)
            .field("subjects", &self.provenance.subjects.len())
            .field("threshold", &self.provenance.threshold)
            .finish()
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

        Ok(Self {
            session,
            tokenizer,
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
    pub const fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    /// Reads a batch of claims.
    ///
    /// # Errors
    ///
    /// Fails if tokenisation or the forward pass fails.
    pub fn read(&mut self, claims: &[String]) -> Result<Vec<Reading>> {
        if claims.is_empty() {
            return Ok(Vec::new());
        }
        let encodings = self
            .tokenizer
            .encode_batch(claims.to_vec(), true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;

        let rows = encodings.len();
        let cols = encodings.first().map_or(0, |e| e.get_ids().len());
        let ids: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_ids().iter().map(|&id| i64::from(id)))
            .collect();
        let mask: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_attention_mask().iter().map(|&m| i64::from(m)))
            .collect();

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
                    subject: (confidence >= self.provenance.threshold)
                        .then(|| self.order.get(best).copied().unwrap_or(best)),
                    confidence,
                    polarity: Polarity::from_index(polar),
                }
            })
            .collect())
    }
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
}
