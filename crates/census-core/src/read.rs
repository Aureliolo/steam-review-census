//! Reading a whole corpus, one claim at a time.
//!
//! Two passes over the capture. The first finds every distinct claim and asks the model about
//! it; the second walks the reviews again and adds up what the answers mean. Distinct rather
//! than every claim because "Great game." is one claim written a thousand times, and reading
//! it a thousand times is a thousand times the electricity for the same answer.
//!
//! What comes out is deliberately three different shapes of number, and the difference
//! matters more than any of them:
//!
//! - **Mention rate**: the share of reviews raising a subject. A review counts once however
//!   many claims it makes about it, so nobody's verbosity moves it. This is the headline.
//! - **Polarity**: per review, per subject, as praised, criticised or mixed. A review that
//!   loves the art and hates the framerate says two things and is recorded as saying two.
//! - **Claim share**: the share of all claims. Verbosity-weighted, useful for reading,
//!   never a headline, and labelled wherever it appears.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use arrow::{
    array::{ArrayRef, Float32Builder, StringBuilder, UInt16Builder},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use parquet::{
    arrow::ArrowWriter,
    basic::{Compression, ZstdLevel},
    file::properties::WriterProperties,
};
use serde::Serialize;

use crate::{
    Result,
    reader::{ClaimReader, Polarity, Reading},
    taxonomy::CORE_SPINE,
};

/// Claims per forward pass. Claims are short, so this is larger than the review-level default.
pub const DEFAULT_READ_BATCH: usize = 128;

#[derive(Debug, Clone)]
pub struct ReadOptions {
    pub out_dir: PathBuf,
    pub top_helpful: usize,
    pub batch_size: usize,
    /// Which language to count. The capture is always the whole census; this decides what is
    /// counted from it, so the choice can change without re-downloading anything and every
    /// figure can say which reviews it is about.
    pub language: Option<String>,
}

impl Default for ReadOptions {
    fn default() -> Self {
        Self {
            out_dir: PathBuf::from("data"),
            top_helpful: crate::classify::DEFAULT_TOP_HELPFUL,
            batch_size: DEFAULT_READ_BATCH,
            language: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReadProgress {
    pub done: u64,
    pub total: u64,
    pub reading_claims: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct Tally {
    mention_reviews: u64,
    primary_reviews: u64,
    claims: u64,
    praised: u64,
    criticised: u64,
    mixed: u64,
    top_mention_reviews: u64,
    positive_mentions: u64,
}

/// One subject, counted over a corpus.
#[derive(Debug, Clone, Serialize)]
pub struct SubjectCount {
    pub id: &'static str,
    pub label: &'static str,
    /// Reviews raising this subject at least once. The headline denominator.
    pub mention_reviews: u64,
    /// Reviews whose main subject this is, which are exhaustive across subjects.
    pub primary_reviews: u64,
    /// Claims about this subject. Verbosity-weighted; never a headline.
    pub claims: u64,
    pub praised: u64,
    pub criticised: u64,
    pub mixed: u64,
    pub top_mention_reviews: u64,
    /// Of the reviews raising this, how many still recommended the game.
    pub positive_mentions: u64,
}

/// What reading a corpus found.
#[derive(Debug, Clone, Serialize)]
pub struct ReadReport {
    pub app_id: u32,
    /// Reviews counted, which is every review in the capture unless a language was named.
    pub reviews: u64,
    /// Reviews in the capture, whatever the language.
    pub corpus_reviews: u64,
    pub language: Option<String>,
    pub claims: u64,
    /// Claims the model would not put a subject on. Reported rather than filed under
    /// whatever scored highest, which is the whole point of the rebuild.
    pub unclassified_claims: u64,
    /// Reviews where no claim got a subject at all.
    pub silent_reviews: u64,
    pub positive: u64,
    pub top_helpful: u64,
    pub model: String,
    pub spine_version: String,
    pub threshold: f32,
    pub device: String,
    pub subjects: Vec<SubjectCount>,
    pub languages: Vec<(String, u64)>,
    #[serde(skip)]
    pub elapsed: Duration,
}

impl ReadReport {
    /// Share of claims the model declined to put a subject on.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "claim counts are far below 2^53"
    )]
    pub fn unclassified_share(&self) -> Option<f64> {
        (self.claims > 0).then(|| self.unclassified_claims as f64 / self.claims as f64)
    }

    /// Writes the counts beside the readings they were computed from.
    ///
    /// # Errors
    ///
    /// Fails if the snapshot cannot be written to.
    pub fn save(&self, path: &Path) -> Result<()> {
        std::fs::write(path, serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }
}

fn reading_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("recommendationid", DataType::Utf8, false),
        Field::new("claim_index", DataType::UInt16, false),
        // Null where the model declined, which is a recorded answer rather than a gap.
        Field::new("subject", DataType::Utf8, true),
        Field::new("confidence", DataType::Float32, false),
        Field::new("polarity", DataType::Utf8, false),
    ]))
}

/// Reads every claim in the most recent capture.
///
/// # Errors
///
/// Fails if the capture is missing, or if reading, running the model or writing fails.
pub fn read_corpus(
    model: &mut ClaimReader,
    app_id: u32,
    options: &ReadOptions,
    mut on_progress: impl FnMut(ReadProgress),
) -> Result<ReadReport> {
    let started = Instant::now();
    let snapshot = crate::embed::latest_snapshot(&options.out_dir, app_id)?;

    let answers = read_distinct_claims(model, &snapshot, options.batch_size, &mut on_progress)?;
    let counted = count_reviews(app_id, &snapshot, options, &answers, &mut on_progress)?;

    Ok(ReadReport {
        elapsed: started.elapsed(),
        device: model.device().to_owned(),
        threshold: model.provenance().threshold,
        model: model.provenance().trained_from.clone(),
        spine_version: model.provenance().spine_version.clone(),
        ..counted
    })
}

/// Asks the model about every distinct claim in the corpus.
fn read_distinct_claims(
    model: &mut ClaimReader,
    snapshot: &Path,
    batch_size: usize,
    on_progress: &mut impl FnMut(ReadProgress),
) -> Result<HashMap<[u8; 32], Reading>> {
    let mut answers: HashMap<[u8; 32], Reading> = HashMap::new();
    let mut pending: Vec<String> = Vec::with_capacity(batch_size);
    let mut keys: Vec<[u8; 32]> = Vec::with_capacity(batch_size);

    crate::capture::for_each_body(snapshot, |_, _, text| {
        for claim in crate::claims::split(text) {
            let key = crate::embed::sha256_bytes(&claim);
            if answers.contains_key(&key) || keys.contains(&key) {
                continue;
            }
            keys.push(key);
            pending.push(claim.into_owned());
            if pending.len() >= batch_size {
                for (key, reading) in keys.drain(..).zip(model.read(&pending)?) {
                    answers.insert(key, reading);
                }
                pending.clear();
                on_progress(ReadProgress {
                    done: answers.len() as u64,
                    total: 0,
                    reading_claims: true,
                });
            }
        }
        Ok(())
    })?;

    if !pending.is_empty() {
        for (key, reading) in keys.drain(..).zip(model.read(&pending)?) {
            answers.insert(key, reading);
        }
    }
    Ok(answers)
}

/// What one review turned out to be about.
struct Verdict {
    subjects: Vec<usize>,
    primary: Option<usize>,
    praise: Vec<bool>,
    complaint: Vec<bool>,
    claims: usize,
    unclassified: usize,
}

fn judge(text: &str, answers: &HashMap<[u8; 32], Reading>) -> Verdict {
    let mut praise = vec![false; CORE_SPINE.len()];
    let mut complaint = vec![false; CORE_SPINE.len()];
    let mut seen = vec![false; CORE_SPINE.len()];
    let mut primary = None;
    let mut best = f32::NEG_INFINITY;
    let mut claims = 0;
    let mut unclassified = 0;

    for claim in crate::claims::split(text) {
        claims += 1;
        let Some(reading) = answers.get(&crate::embed::sha256_bytes(&claim)) else {
            unclassified += 1;
            continue;
        };
        let Some(subject) = reading.subject else {
            unclassified += 1;
            continue;
        };
        seen[subject] = true;
        match reading.polarity {
            Polarity::Praise => praise[subject] = true,
            Polarity::Complaint => complaint[subject] = true,
            Polarity::Neutral => {}
        }
        // The review's main subject is whichever claim the model was surest about. A review
        // is most about the thing it says most clearly, not the thing it says first.
        if reading.confidence > best {
            best = reading.confidence;
            primary = Some(subject);
        }
    }

    Verdict {
        subjects: (0..CORE_SPINE.len()).filter(|index| seen[*index]).collect(),
        primary,
        praise,
        complaint,
        claims,
        unclassified,
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "one pass over a corpus, doing every tally it needs in the one walk it can afford"
)]
fn count_reviews(
    app_id: u32,
    snapshot: &Path,
    options: &ReadOptions,
    answers: &HashMap<[u8; 32], Reading>,
    on_progress: &mut impl FnMut(ReadProgress),
) -> Result<ReadReport> {
    let path = snapshot.join("readings.parquet");
    let schema = reading_schema();
    let mut writer = ArrowWriter::try_new(
        std::fs::File::create(&path)?,
        Arc::clone(&schema),
        Some(
            WriterProperties::builder()
                .set_compression(Compression::ZSTD(ZstdLevel::default()))
                .build(),
        ),
    )?;

    let mut tallies = vec![Tally::default(); CORE_SPINE.len()];
    let mut languages: HashMap<String, u64> = HashMap::new();
    let mut top: crate::bounded::Smallest<std::cmp::Reverse<u64>, Vec<usize>> =
        crate::bounded::Smallest::new(options.top_helpful);
    let mut rows = ReadingRows::default();

    let mut reviews = 0_u64;
    let mut corpus_reviews = 0_u64;
    let mut claims = 0_u64;
    let mut unclassified = 0_u64;
    let mut silent = 0_u64;
    let mut positive = 0_u64;

    crate::classify::for_each_review(snapshot, |row, text| {
        corpus_reviews += 1;
        *languages.entry(row.language.clone()).or_default() += 1;
        if options
            .language
            .as_ref()
            .is_some_and(|wanted| wanted != &row.language)
        {
            return Ok(());
        }
        reviews += 1;
        if row.voted_up {
            positive += 1;
        }

        let verdict = judge(text, answers);
        claims += verdict.claims as u64;
        unclassified += verdict.unclassified as u64;
        if verdict.subjects.is_empty() {
            silent += 1;
        }
        if let Some(primary) = verdict.primary {
            tallies[primary].primary_reviews += 1;
        }
        for &subject in &verdict.subjects {
            let tally = &mut tallies[subject];
            tally.mention_reviews += 1;
            if row.voted_up {
                tally.positive_mentions += 1;
            }
            match (verdict.praise[subject], verdict.complaint[subject]) {
                (true, true) => tally.mixed += 1,
                (true, false) => tally.praised += 1,
                (false, true) => tally.criticised += 1,
                (false, false) => {}
            }
        }
        for (index, claim) in crate::claims::split(text).into_iter().enumerate() {
            let reading = answers.get(&crate::embed::sha256_bytes(&claim));
            if let Some(reading) = reading {
                if let Some(subject) = reading.subject {
                    tallies[subject].claims += 1;
                }
            }
            rows.push(&row.recommendationid, index, reading);
        }

        // Helpfulness ranks descending, and the bounded keeper takes the smallest key.
        top.offer(
            std::cmp::Reverse(row.helpfulness.to_bits()),
            verdict.subjects,
        );

        if rows.len() >= 16_384 {
            writer.write(&rows.take(&schema)?)?;
            on_progress(ReadProgress {
                done: reviews,
                total: 0,
                reading_claims: false,
            });
        }
        Ok(())
    })?;

    if rows.len() > 0 {
        writer.write(&rows.take(&schema)?)?;
    }
    writer.close()?;

    let top_reviews = top.take();
    for subjects in &top_reviews {
        for &subject in subjects {
            tallies[subject].top_mention_reviews += 1;
        }
    }

    let mut ranked: Vec<(String, u64)> = languages.into_iter().collect();
    ranked.sort_by_key(|(name, count)| (std::cmp::Reverse(*count), name.clone()));

    Ok(ReadReport {
        app_id,
        reviews,
        corpus_reviews,
        language: options.language.clone(),
        claims,
        unclassified_claims: unclassified,
        silent_reviews: silent,
        positive,
        top_helpful: top_reviews.len() as u64,
        model: String::new(),
        spine_version: String::new(),
        threshold: 0.0,
        device: String::new(),
        subjects: CORE_SPINE
            .iter()
            .zip(&tallies)
            .map(|(category, tally)| SubjectCount {
                id: category.id,
                label: category.label,
                mention_reviews: tally.mention_reviews,
                primary_reviews: tally.primary_reviews,
                claims: tally.claims,
                praised: tally.praised,
                criticised: tally.criticised,
                mixed: tally.mixed,
                top_mention_reviews: tally.top_mention_reviews,
                positive_mentions: tally.positive_mentions,
            })
            .collect(),
        languages: ranked,
        elapsed: Duration::default(),
    })
}

#[derive(Default)]
struct ReadingRows {
    ids: Vec<String>,
    indexes: Vec<u16>,
    subjects: Vec<Option<&'static str>>,
    confidences: Vec<f32>,
    polarities: Vec<&'static str>,
}

impl ReadingRows {
    fn len(&self) -> usize {
        self.ids.len()
    }

    fn push(&mut self, id: &str, index: usize, reading: Option<&Reading>) {
        self.ids.push(id.to_owned());
        self.indexes.push(u16::try_from(index).unwrap_or(u16::MAX));
        self.subjects.push(
            reading
                .and_then(|reading| reading.subject)
                .and_then(|subject| CORE_SPINE.get(subject))
                .map(|category| category.id),
        );
        self.confidences
            .push(reading.map_or(0.0, |reading| reading.confidence));
        self.polarities.push(
            reading.map_or(Polarity::Neutral, |reading| reading.polarity)
                .as_str(),
        );
    }

    fn take(&mut self, schema: &Arc<Schema>) -> Result<RecordBatch> {
        let mut ids = StringBuilder::new();
        let mut indexes = UInt16Builder::new();
        let mut subjects = StringBuilder::new();
        let mut confidences = Float32Builder::new();
        let mut polarities = StringBuilder::new();

        for row in 0..self.len() {
            ids.append_value(&self.ids[row]);
            indexes.append_value(self.indexes[row]);
            subjects.append_option(self.subjects[row]);
            confidences.append_value(self.confidences[row]);
            polarities.append_value(self.polarities[row]);
        }
        self.ids.clear();
        self.indexes.clear();
        self.subjects.clear();
        self.confidences.clear();
        self.polarities.clear();

        let columns: Vec<ArrayRef> = vec![
            Arc::new(ids.finish()),
            Arc::new(indexes.finish()),
            Arc::new(subjects.finish()),
            Arc::new(confidences.finish()),
            Arc::new(polarities.finish()),
        ];
        Ok(RecordBatch::try_new(Arc::clone(schema), columns)?)
    }
}
