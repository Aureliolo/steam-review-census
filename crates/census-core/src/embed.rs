//! Turning captured review text into vectors, locally.
//!
//! Vectors are stored once per *distinct* review text, not once per review, so repeated
//! reviews cost nothing to embed twice. Rows join back to the capture on `sha256(review)`.

use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use arrow::{
    array::{Array, ArrayRef, FixedSizeListBuilder, Float32Builder, StringBuilder, UInt32Builder},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use ndarray::Array2;
use ort::{session::Session, value::Tensor};
use parquet::{
    arrow::{ArrowWriter, arrow_reader::ParquetRecordBatchReaderBuilder},
    basic::{Compression, ZstdLevel},
    file::properties::WriterProperties,
};
use sha2::{Digest, Sha256};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::{
    Error, Result,
    model::{self, E5_PREFIX, EMBEDDING_DIM, MAX_TOKENS, MODEL_ID},
};

pub const DEFAULT_BATCH_SIZE: usize = 64;

pub struct Embedder {
    session: Session,
    tokenizer: Tokenizer,
    precision: model::Precision,
    device_name: &'static str,
    output_name: String,
    wants_token_type_ids: bool,
}

impl std::fmt::Debug for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedder")
            .field("model", &MODEL_ID)
            .field("device", &self.device_name)
            .finish_non_exhaustive()
    }
}

impl Embedder {
    /// Which backend the model actually ended up on.
    #[must_use]
    pub fn device(&self) -> &'static str {
        self.device_name
    }

    /// Which build of the graph is loaded. Part of a corpus's provenance.
    #[must_use]
    pub fn precision(&self) -> model::Precision {
        self.precision
    }

    /// # Errors
    ///
    /// Fails if the model files are missing, corrupt, or cannot be loaded on any backend.
    pub fn load(cache_dir: &Path, precision: model::Precision) -> Result<Self> {
        let (session, device_name) = model::session(cache_dir, precision)?;
        let mut tokenizer = Tokenizer::from_file(model::tokenizer_path(cache_dir))
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..PaddingParams::default()
        }));
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_TOKENS,
                ..TruncationParams::default()
            }))
            .map_err(|e| Error::Tokenizer(e.to_string()))?;

        // Read the graph's own signature rather than assuming one: exports of the same
        // model differ in whether they take token_type_ids and in what they name outputs.
        let wants_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");
        let output_name = session
            .outputs()
            .first()
            .ok_or(Error::MalformedPayload {
                field: "onnx output",
            })?
            .name()
            .to_owned();

        Ok(Self {
            session,
            tokenizer,
            precision,
            device_name,
            output_name,
            wants_token_type_ids,
        })
    }

    /// Embeds a batch, mean-pooled over real tokens and L2-normalised.
    ///
    /// Under [`Precision::Int8`] a review's vector depends on which reviews were embedded
    /// alongside it: dynamic quantisation derives its activation scales per tensor, and a
    /// tensor spans the batch, so the extremes of a batch set the rounding for everything in
    /// it. Batches of 64 against batches of one gave a mean cosine of 0.997 and put 12.5% of
    /// reviews in a different category. It is not padding, which grouping by exact token
    /// length removes without closing the gap, and it is not the execution provider, which
    /// reproduces it on CPU. The float builds do not have the problem at all, which is why
    /// one of them is the default.
    ///
    /// Even under int8 the aggregate survives: mention rates across a corpus moved at most
    /// 1.6 percentage points and 0.31 on average, because per-review disagreements largely
    /// cancel. Batch size is recorded alongside the vectors either way, so two corpora can
    /// be compared knowingly.
    ///
    /// # Errors
    ///
    /// Fails if tokenisation or the forward pass fails.
    pub fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let prefixed: Vec<String> = texts.iter().map(|t| format!("{E5_PREFIX}{t}")).collect();
        let encodings = self
            .tokenizer
            .encode_batch(prefixed, true)
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

        let ids_array = Array2::from_shape_vec((rows, cols), ids)?;
        let mask_array = Array2::from_shape_vec((rows, cols), mask.clone())?;

        let outputs = if self.wants_token_type_ids {
            let types = Array2::<i64>::zeros((rows, cols));
            self.session.run(ort::inputs![
                "input_ids" => Tensor::from_array(ids_array)?,
                "attention_mask" => Tensor::from_array(mask_array)?,
                "token_type_ids" => Tensor::from_array(types)?,
            ])?
        } else {
            self.session.run(ort::inputs![
                "input_ids" => Tensor::from_array(ids_array)?,
                "attention_mask" => Tensor::from_array(mask_array)?,
            ])?
        };

        let hidden = outputs[self.output_name.as_str()]
            .try_extract_array::<f32>()?
            .into_dimensionality::<ndarray::Ix3>()?;
        Ok(pool(&hidden, &mask, rows, cols))
    }
}

/// Mean-pools over real tokens, then L2-normalises.
///
/// Padding tokens must not contribute, or a short review batched with a long one would get
/// a vector that depends on its batch neighbours rather than on what it says.
fn pool(
    hidden: &ndarray::ArrayView3<'_, f32>,
    mask: &[i64],
    rows: usize,
    cols: usize,
) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut acc = vec![0.0_f32; EMBEDDING_DIM];
        let mut kept = 0.0_f32;
        for col in 0..cols {
            if mask[row * cols + col] == 0 {
                continue;
            }
            kept += 1.0;
            for (dim, value) in acc.iter_mut().enumerate() {
                *value += hidden[[row, col, dim]];
            }
        }
        let divisor = if kept > 0.0 { kept } else { 1.0 };
        for value in &mut acc {
            *value /= divisor;
        }
        let norm = acc.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut acc {
                *value /= norm;
            }
        }
        out.push(acc);
    }
    out
}

#[derive(Debug, Clone, Copy)]
pub struct EmbedProgress {
    pub embedded: u64,
    pub unique_texts: u64,
}

#[derive(Debug, Clone)]
pub struct EmbedReport {
    pub app_id: u32,
    pub reviews: u64,
    pub unique_texts: u64,
    pub dim: usize,
    pub device: &'static str,
    pub elapsed: Duration,
    pub path: PathBuf,
    pub bytes: u64,
}

impl EmbedReport {
    /// Share of review texts that were already seen, and so cost nothing to embed.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "review counts are far below 2^53"
    )]
    pub fn dedupe_rate(&self) -> Option<f64> {
        (self.reviews > 0).then(|| 1.0 - (self.unique_texts as f64 / self.reviews as f64))
    }

    /// Distinct texts embedded per second.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "review counts are far below 2^53"
    )]
    pub fn texts_per_second(&self) -> f64 {
        let seconds = self.elapsed.as_secs_f64();
        if seconds <= 0.0 {
            return 0.0;
        }
        self.unique_texts as f64 / seconds
    }
}

/// Embeds the most recent capture for an app.
///
/// # Errors
///
/// Fails if no capture exists, or if reading, embedding or writing fails.
pub fn embed_corpus(
    embedder: &mut Embedder,
    out_dir: &Path,
    app_id: u32,
    batch_size: usize,
    mut on_progress: impl FnMut(EmbedProgress),
) -> Result<EmbedReport> {
    let started = Instant::now();
    let snapshot = latest_snapshot(out_dir, app_id)?;
    let (mut counts, reviews) = text_counts(&snapshot)?;
    let unique_texts = counts.len() as u64;

    let path = snapshot.join("embeddings.parquet");
    // Written aside and renamed at the end. An interrupted run used to leave an empty
    // embeddings.parquet behind, which reads as a finished artefact and fails confusingly
    // everywhere downstream; a partial file under its own name cannot be mistaken for one.
    let partial = snapshot.join("embeddings.parquet.partial");
    let schema = embedding_schema();
    // A row group is buffered whole before it reaches the disk, and the default holds a
    // million rows. At 1.5 KB a vector that is the entire file in memory for any corpus
    // smaller than that, which is most of them, and it is why embedding a million-review
    // game was killed for memory even after the text itself stopped being held.
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .set_max_row_group_row_count(Some(VECTORS_PER_ROW_GROUP))
        .build();
    let mut writer = ArrowWriter::try_new(
        std::fs::File::create(&partial)?,
        Arc::clone(&schema),
        Some(props),
    )?;

    let mut window: Vec<(String, u32)> = Vec::with_capacity(LENGTH_WINDOW);
    let mut written: u64 = 0;

    for_each_review_text(&snapshot, |text| {
        // Removing rather than looking up means the second and later copies of a repeated
        // review find nothing and are skipped, and the map shrinks as the corpus is walked.
        let Some(seen) = counts.remove(&sha256_bytes(text)) else {
            return Ok(());
        };
        window.push((text.to_owned(), seen));
        if window.len() >= LENGTH_WINDOW {
            written = written.saturating_add(drain_window(
                embedder,
                app_id,
                batch_size,
                &schema,
                &mut writer,
                &mut window,
            )?);
            on_progress(EmbedProgress {
                embedded: written,
                unique_texts,
            });
        }
        Ok(())
    })?;
    drain_window(
        embedder,
        app_id,
        batch_size,
        &schema,
        &mut writer,
        &mut window,
    )?;
    writer.close()?;
    std::fs::rename(&partial, &path)?;

    // Batch size belongs with the vectors, not in a shell history. The int8 model is not
    // batch-invariant, so two corpora embedded at different batch sizes are not strictly
    // comparable, and a reader has no other way to find out.
    std::fs::write(
        snapshot.join("embeddings.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "model": MODEL_ID,
            "precision": embedder.precision().as_str(),
            "dimensions": EMBEDDING_DIM,
            "batch_size": batch_size,
            "device": embedder.device(),
            "reviews": reviews,
            "unique_texts": unique_texts,
            "note": "Vectors from the int8 build depend on batch size: dynamic quantisation \
                     takes its activation scales per tensor, and a tensor spans the batch. \
                     The float builds do not. Compare corpora built the same way.",
        }))?,
    )?;

    Ok(EmbedReport {
        app_id,
        reviews,
        unique_texts,
        dim: EMBEDDING_DIM,
        device: embedder.device(),
        elapsed: started.elapsed(),
        bytes: std::fs::metadata(&path)
            .map(|m| m.len())
            .unwrap_or_default(),
        path,
    })
}

/// Vectors per Parquet row group, which is what the writer buffers before flushing.
///
/// At 384 floats a row that is roughly 100 MB held at a time, which is a sensible row group
/// for readers and a bounded amount of memory for writers.
const VECTORS_PER_ROW_GROUP: usize = 65_536;

/// Distinct texts buffered before they are sorted by length and embedded.
///
/// Every sequence in a batch is padded to the longest member, so batching a ten-character
/// review with a three-thousand-character one processes both as though they were three
/// thousand characters. Sorting the whole corpus by length removes that waste and costs a
/// corpus-sized allocation; sorting a window of this size removes nearly all of it, because
/// a window drawn in corpus order has the corpus's own spread of lengths and cutting it into
/// batches still puts similar lengths together.
const LENGTH_WINDOW: usize = 16_384;

/// Embeds one window, similar lengths together, and empties it.
///
/// Sorting by length keeps padding down, which is worth doing for speed. It is deliberately
/// not taken further than that: batching reviews of *identical* token length, so that no
/// batch pads at all, was measured at 1.6 times the cost and moved the vectors no closer to
/// what the model produces one review at a time. Padding is not what makes them differ.
fn drain_window(
    embedder: &mut Embedder,
    app_id: u32,
    batch_size: usize,
    schema: &Arc<Schema>,
    writer: &mut ArrowWriter<std::fs::File>,
    window: &mut Vec<(String, u32)>,
) -> Result<u64> {
    window.sort_unstable_by_key(|(text, _)| text.len());
    let mut written = 0_u64;
    for chunk in window.chunks(batch_size.max(1)) {
        let texts: Vec<String> = chunk.iter().map(|(text, _)| text.clone()).collect();
        let vectors = embedder.embed(&texts)?;
        writer.write(&batch_to_record(app_id, chunk, &vectors, schema)?)?;
        written = written.saturating_add(chunk.len() as u64);
    }
    window.clear();
    Ok(written)
}

fn embedding_schema() -> Arc<Schema> {
    let dim = i32::try_from(EMBEDDING_DIM).unwrap_or(0);
    Arc::new(Schema::new(vec![
        Field::new("text_sha256", DataType::Utf8, false),
        Field::new("appid", DataType::UInt32, false),
        Field::new("n_reviews", DataType::UInt32, false),
        Field::new("model", DataType::Utf8, false),
        // The inner field is nullable to match what FixedSizeListBuilder produces; the
        // values themselves are never null, since a vector is written or the row is not.
        Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
            false,
        ),
    ]))
}

fn batch_to_record(
    app_id: u32,
    chunk: &[(String, u32)],
    vectors: &[Vec<f32>],
    schema: &Arc<Schema>,
) -> Result<RecordBatch> {
    let dim = i32::try_from(EMBEDDING_DIM).unwrap_or(0);
    let mut hashes = StringBuilder::new();
    let mut appids = UInt32Builder::new();
    let mut counts = UInt32Builder::new();
    let mut models = StringBuilder::new();
    let mut embeddings = FixedSizeListBuilder::new(Float32Builder::new(), dim);

    for ((text, n), vector) in chunk.iter().zip(vectors) {
        hashes.append_value(sha256_hex(text));
        appids.append_value(app_id);
        counts.append_value(*n);
        models.append_value(MODEL_ID);
        embeddings.values().append_slice(vector);
        embeddings.append(true);
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(hashes.finish()),
        Arc::new(appids.finish()),
        Arc::new(counts.finish()),
        Arc::new(models.finish()),
        Arc::new(embeddings.finish()),
    ];
    Ok(RecordBatch::try_new(Arc::clone(schema), columns)?)
}

/// Distinct review texts and how many reviews carry each, plus the total review count.
/// Visits the text of every review in a snapshot, one at a time.
fn for_each_review_text(snapshot: &Path, mut visit: impl FnMut(&str) -> Result<()>) -> Result<()> {
    let mut shards: Vec<PathBuf> = std::fs::read_dir(snapshot)?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("shard-") && n.ends_with(".parquet"))
        })
        .collect();
    shards.sort();

    for shard in shards {
        let reader = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(&shard)?)?
            .with_batch_size(8192)
            .build()?;
        for batch in reader {
            let batch = batch?;
            let Some(column) = batch.column_by_name("review") else {
                continue;
            };
            let texts = column
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or(Error::MalformedPayload { field: "review" })?;
            for i in 0..texts.len() {
                if texts.is_null(i) {
                    continue;
                }
                visit(texts.value(i))?;
            }
        }
    }
    Ok(())
}

/// How many reviews share each distinct text, keyed by the hash rather than by the text.
///
/// Holding the text costs an allocation the size of the corpus. A million reviews averaging
/// a few hundred characters is well over a gigabyte, and embedding Cyberpunk 2077 was killed
/// by the operating system for exactly that. A hash is thirty-two bytes whatever the review
/// says, and the text is read back from the capture when it is actually needed.
fn text_counts(snapshot: &Path) -> Result<(HashMap<[u8; 32], u32>, u64)> {
    let mut counts: HashMap<[u8; 32], u32> = HashMap::new();
    let mut reviews = 0_u64;

    for_each_review_text(snapshot, |text| {
        reviews += 1;
        *counts.entry(sha256_bytes(text)).or_insert(0) += 1;
        Ok(())
    })?;

    if counts.is_empty() {
        return Err(Error::NoCapture {
            path: snapshot.to_path_buf(),
        });
    }
    Ok((counts, reviews))
}

/// Streams every stored vector, one at a time.
///
/// A corpus of a million reviews holds roughly 1.8 GB of vectors, so anything that reads
/// them all into a map costs more memory than the rest of the tool put together. Everything
/// that needs the vectors either reduces them to something small or joins them against
/// something small, and both are streaming operations.
///
/// # Errors
///
/// Fails if the embeddings are missing or malformed.
pub(crate) fn for_each_vector(
    snapshot: &Path,
    mut visit: impl FnMut(&str, &[f32]) -> Result<()>,
) -> Result<()> {
    use arrow::array::{FixedSizeListArray, Float32Array, StringArray};

    let path = snapshot.join("embeddings.parquet");
    let file = std::fs::File::open(&path).map_err(|_| Error::NoEmbeddings { path })?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(4096)
        .build()?;

    for batch in reader {
        let batch = batch?;
        let hashes = batch
            .column_by_name("text_sha256")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or(Error::MalformedPayload {
                field: "text_sha256",
            })?;
        let vectors = batch
            .column_by_name("embedding")
            .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
            .ok_or(Error::MalformedPayload { field: "embedding" })?;
        for row in 0..batch.num_rows() {
            let values = vectors.value(row);
            let floats = values
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or(Error::MalformedPayload { field: "embedding" })?;
            visit(hashes.value(row), floats.values())?;
        }
    }
    Ok(())
}

/// The mean of every distinct review vector in a corpus.
///
/// This is all that calibration ever needed from a corpus. The mean of an anchor's
/// similarity to every review is the anchor's similarity to the mean review, because a dot
/// product is linear in its second argument, so a million vectors reduce to 384 floats
/// computed once instead of being walked again for every anchor, fold and parameter tried.
///
/// Distinct texts rather than reviews, so that a copypasta posted five hundred times counts
/// as one thing the corpus says rather than five hundred.
///
/// # Errors
///
/// Fails if the embeddings are missing or malformed.
pub fn corpus_centroid(out_dir: &Path, app_id: u32) -> Result<Vec<f32>> {
    let snapshot = latest_snapshot(out_dir, app_id)?;
    let mut total = vec![0.0_f64; crate::model::EMBEDDING_DIM];
    let mut seen = 0_u64;

    for_each_vector(&snapshot, |_, vector| {
        for (slot, value) in total.iter_mut().zip(vector) {
            *slot += f64::from(*value);
        }
        seen += 1;
        Ok(())
    })?;

    if seen == 0 {
        return Err(Error::NoEmbeddings {
            path: snapshot.join("embeddings.parquet"),
        });
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a mean of unit vectors is far inside f32 range"
    )]
    #[expect(
        clippy::cast_precision_loss,
        reason = "corpora are millions of reviews, not quadrillions"
    )]
    Ok(total
        .into_iter()
        .map(|sum| (sum / seen as f64) as f32)
        .collect())
}

/// The stored vectors for a named set of reviews, and nothing else.
///
/// Reference labels name reviews by id while vectors are keyed by the hash of the text, so
/// this walks the capture for the wanted ids first and then streams the vectors, keeping
/// only the few hundred that were asked for.
///
/// # Errors
///
/// Fails if the capture or the embeddings cannot be read.
pub fn vectors_for<S: std::hash::BuildHasher>(
    out_dir: &Path,
    app_id: u32,
    ids: &HashSet<String, S>,
) -> Result<HashMap<String, Vec<f32>>> {
    let snapshot = latest_snapshot(out_dir, app_id)?;
    let texts = crate::capture::texts_for(&snapshot, ids)?;

    let mut wanted: HashMap<String, Vec<String>> = HashMap::new();
    for (id, text) in texts {
        wanted.entry(sha256_hex(&text)).or_default().push(id);
    }

    let mut found = HashMap::new();
    for_each_vector(&snapshot, |hash, vector| {
        if let Some(ids) = wanted.get(hash) {
            for id in ids {
                found.insert(id.clone(), vector.to_vec());
            }
        }
        Ok(())
    })?;
    Ok(found)
}

/// The newest snapshot directory for an app.
pub(crate) fn latest_snapshot(out_dir: &Path, app_id: u32) -> Result<PathBuf> {
    let app_dir = out_dir.join(format!("appid={app_id}"));
    let mut best: Option<(i64, PathBuf)> = None;

    for entry in std::fs::read_dir(&app_dir).map_err(|_| Error::NoCapture {
        path: app_dir.clone(),
    })? {
        let path = entry?.path();
        let Some(stamp) = path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_prefix("snapshot="))
            .and_then(|n| n.parse::<i64>().ok())
        else {
            continue;
        };
        // An interrupted crawl leaves a snapshot directory holding nothing, and its stamp is
        // newer than the completed crawl it was meant to replace. Taking it on age alone
        // would hide a finished corpus behind an empty one, so a snapshot has to contain at
        // least one shard before it can shadow anything.
        if !holds_a_shard(&path) {
            continue;
        }
        if best.as_ref().is_none_or(|(seen, _)| stamp > *seen) {
            best = Some((stamp, path));
        }
    }
    best.map(|(_, path)| path)
        .ok_or(Error::NoCapture { path: app_dir })
}

/// Whether a snapshot holds a shard with anything in it.
///
/// The file has to be non-empty, not merely present. A crawl creates each shard's file
/// before it fetches the first page, so an interrupted one leaves a directory of zero-byte
/// shards, and a check on the name alone would take that for a corpus.
fn holds_a_shard(snapshot: &Path) -> bool {
    std::fs::read_dir(snapshot).is_ok_and(|entries| {
        entries.filter_map(std::result::Result::ok).any(|entry| {
            let named_like_a_shard = entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with("shard-") && name.ends_with(".parquet"));
            named_like_a_shard && entry.metadata().is_ok_and(|meta| meta.len() > 0)
        })
    })
}

/// The raw digest, for keeping in memory. Thirty-two bytes against sixty-four for the hex.
pub(crate) fn sha256_bytes(text: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hasher.finalize().into()
}

/// The digest as it is written to disk and joined on.
pub(crate) fn sha256_hex(text: &str) -> String {
    let mut out = String::with_capacity(64);
    for byte in sha256_bytes(text) {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_join_key_is_plain_sha256_of_the_review_text() {
        // These are what `sha256(review)` returns in DuckDB, so a join against the capture
        // lines up without the caller having to know anything about how rows were keyed.
        assert_eq!(
            sha256_hex("good game"),
            "e192095a02c29325df05003235dba8978751279a7405eee58488604cc5068c43"
        );
        assert_eq!(
            sha256_hex(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn pooling_ignores_padding_and_returns_unit_vectors() {
        // Two positions, the second masked out. The result must equal the first position
        // alone, normalised, not the average of both.
        let mut data = vec![0.0_f32; 2 * EMBEDDING_DIM];
        data[0] = 3.0;
        data[1] = 4.0;
        data[EMBEDDING_DIM] = 100.0;
        let hidden = ndarray::Array3::from_shape_vec((1, 2, EMBEDDING_DIM), data).unwrap();
        let pooled = pool(&hidden.view(), &[1, 0], 1, 2);

        assert!((pooled[0][0] - 0.6).abs() < 1e-6, "{:?}", &pooled[0][..2]);
        assert!((pooled[0][1] - 0.8).abs() < 1e-6);
        let norm = pooled[0].iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");
    }

    #[test]
    fn dedupe_rate_reports_the_share_that_cost_nothing() {
        let report = |reviews, unique| EmbedReport {
            app_id: 1,
            reviews,
            unique_texts: unique,
            dim: EMBEDDING_DIM,
            device: "cpu",
            elapsed: Duration::from_secs(1),
            path: PathBuf::new(),
            bytes: 0,
        };
        assert_eq!(report(100, 100).dedupe_rate(), Some(0.0));
        assert_eq!(report(100, 60).dedupe_rate(), Some(0.4));
        assert_eq!(report(0, 0).dedupe_rate(), None);
        assert!((report(100, 60).texts_per_second() - 60.0).abs() < 1e-9);
    }

    #[test]
    fn a_missing_capture_is_named_rather_than_silently_empty() {
        let err = latest_snapshot(Path::new("definitely-not-a-corpus-dir"), 1).unwrap_err();
        assert!(matches!(err, Error::NoCapture { .. }));
    }

    #[test]
    fn an_interrupted_crawl_does_not_hide_the_finished_one_behind_it() {
        // Restarting a crawl creates its snapshot directory before it fetches anything, so
        // an interrupted restart leaves an empty directory stamped later than the corpus it
        // was replacing. Age alone would pick the empty one and every later command would
        // report a corpus that is not there.
        let root = std::env::temp_dir().join("census-snapshot-precedence");
        let app = root.join("appid=1");
        let complete = app.join("snapshot=100");
        let abandoned = app.join("snapshot=200");
        std::fs::create_dir_all(&complete).unwrap();
        std::fs::create_dir_all(&abandoned).unwrap();
        std::fs::write(complete.join("shard-0000.parquet"), b"not really parquet").unwrap();

        assert_eq!(latest_snapshot(&root, 1).unwrap(), complete);

        // A crawl opens each shard's file before it fetches anything, so an interrupted one
        // leaves empty shards behind. Those are not a corpus either.
        std::fs::write(abandoned.join("shard-0000.parquet"), b"").unwrap();
        assert_eq!(latest_snapshot(&root, 1).unwrap(), complete);

        // Once the restart writes a shard with something in it, it does take precedence.
        std::fs::write(abandoned.join("shard-0000.parquet"), b"not really parquet").unwrap();
        assert_eq!(latest_snapshot(&root, 1).unwrap(), abandoned);

        std::fs::remove_dir_all(&root).ok();
    }
}
