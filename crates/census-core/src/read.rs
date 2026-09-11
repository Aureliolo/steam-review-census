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
use serde::{Deserialize, Serialize};

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
    pub depth: Depth,
}

/// How closely each review is read.
///
/// Neither setting drops a review. What changes is how finely one is taken apart before the
/// model reads it, and therefore what a count is a count of.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Depth {
    /// A review becomes the separate points it makes, and each point is read on its own.
    /// This is what makes a mention rate honest: a review that praises the art and damns the
    /// story counts once for each, rather than once for whichever the model noticed.
    #[default]
    Deep,
    /// The whole review is one point. A third of the work, and it systematically understates
    /// anyone who wrote more than a sentence, because the model answers about the review as a
    /// whole and a review about six things is about none of them clearly enough.
    Shallow,
}

impl Depth {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Deep => "deep",
            Self::Shallow => "shallow",
        }
    }

    /// The points a review makes, at this depth.
    ///
    /// Every place the reading pass takes a review apart goes through here, so the pass and
    /// the readings it writes cannot disagree about what a claim index names.
    #[must_use]
    pub fn claims_of(self, text: &str) -> Vec<std::borrow::Cow<'_, str>> {
        match self {
            Self::Deep => crate::claims::split(text),
            Self::Shallow => {
                let whole = text.trim();
                if whole.is_empty() {
                    Vec::new()
                } else {
                    vec![std::borrow::Cow::Borrowed(whole)]
                }
            }
        }
    }
}

impl Default for ReadOptions {
    fn default() -> Self {
        Self {
            out_dir: PathBuf::from("data"),
            top_helpful: crate::capture::DEFAULT_TOP_HELPFUL,
            batch_size: DEFAULT_READ_BATCH,
            language: None,
            depth: Depth::Deep,
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectCount {
    pub id: String,
    pub label: String,
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

impl SubjectCount {
    /// Share of the reviews raising this subject that recommended the game.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "review counts are far below 2^53"
    )]
    pub fn positive_share(&self) -> Option<f64> {
        (self.mention_reviews > 0)
            .then(|| self.positive_mentions as f64 / self.mention_reviews as f64)
    }

    /// Reviews that both praise and criticise this subject, as a share of those raising it.
    ///
    /// The figure the old report had no way to produce. A subject praised by half and damned
    /// by the other half, and one that every reviewer has mixed feelings about, are different
    /// findings that a single positive share renders identically.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "review counts are far below 2^53"
    )]
    pub fn mixed_share(&self) -> Option<f64> {
        (self.mention_reviews > 0).then(|| self.mixed as f64 / self.mention_reviews as f64)
    }
}

/// One month of a corpus.
///
/// A subject raised steadily for two years and one raised furiously in a single week read
/// identically in a total, and they are not the same fact about anything.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Month {
    /// `2024-02`, which sorts as it reads.
    pub label: String,
    pub reviews: u64,
    pub positive: u64,
    /// Reviews raising each subject, in taxonomy order.
    pub subjects: Vec<u64>,
}

impl Month {
    /// Reviews a month needs before a rate of it is worth drawing. Below this a single review
    /// moves the figure by tens of points, and one such month would set the scale for every
    /// month that has something to say.
    pub const ENOUGH_FOR_A_RATE: u64 = 30;

    /// Share of this month's reviews that recommended the game.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "review counts are far below 2^53"
    )]
    pub fn positive_share(&self) -> Option<f64> {
        (self.reviews > 0).then(|| self.positive as f64 / self.reviews as f64)
    }

    /// The positive share, only where the month is big enough to carry one.
    #[must_use]
    pub fn positive_share_if_enough(&self) -> Option<f64> {
        (self.reviews >= Self::ENOUGH_FOR_A_RATE)
            .then(|| self.positive_share())
            .flatten()
    }

    /// Share of this month's reviews raising a subject, by its taxonomy position.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "review counts are far below 2^53"
    )]
    pub fn rate(&self, slot: usize) -> Option<f64> {
        let raised = *self.subjects.get(slot)?;
        (self.reviews > 0).then(|| raised as f64 / self.reviews as f64)
    }
}

/// What reading a corpus found.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadReport {
    pub app_id: u32,
    /// Reviews counted, which is every review in the capture unless a language was named.
    pub reviews: u64,
    /// Reviews in the capture, whatever the language.
    pub corpus_reviews: u64,
    pub language: Option<String>,
    /// How each review was taken apart. Anything that quotes a claim by its index has to
    /// take the review apart the same way, so this is recorded rather than assumed.
    #[serde(default)]
    pub depth: Depth,
    pub claims: u64,
    /// Claims the model would not put a subject on. Reported rather than filed under
    /// whatever scored highest, which is the whole point of the rebuild.
    pub unclassified_claims: u64,
    /// Reviews where no claim got a subject at all.
    pub silent_reviews: u64,
    pub positive: u64,
    pub top_helpful: u64,
    pub model: String,
    /// Which labels the model was trained from. Two readers with the same backbone and the
    /// same threshold trained on different label sets give different numbers, and a reading
    /// that recorded only the backbone could not say which it was.
    #[serde(default)]
    pub trained_on: String,
    /// What this model usually declines, on games it never saw, so this corpus's share can be
    /// read against something.
    #[serde(default)]
    pub usual_declined: Option<f32>,
    pub spine_version: String,
    pub threshold: f32,
    pub device: String,
    /// When the capture this describes was last changed: its crawl, or the last sweep that
    /// brought it up to date. A capture swept since is one these counts no longer describe.
    #[serde(default)]
    pub captured_unix: i64,
    pub subjects: Vec<SubjectCount>,
    /// The terms that separate each subject's praise from its complaints, in taxonomy order.
    #[serde(default)]
    pub said: Vec<crate::said::SaidAbout>,
    pub languages: Vec<(String, u64)>,
    /// What was said month by month, oldest first.
    pub months: Vec<Month>,
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

    /// How much more of this corpus the model declined than it usually does, as a ratio.
    ///
    /// Above about 1.2 the corpus is talking about something the model was not trained to
    /// find, and that is a finding about the game and the taxonomy rather than a detail of
    /// the run. `None` when the model carries no usual figure to compare against.
    #[must_use]
    pub fn declined_against_usual(&self) -> Option<f64> {
        let usual = f64::from(self.usual_declined?);
        let here = self.unclassified_share()?;
        (usual > 0.0).then(|| here / usual)
    }
}

/// Above this ratio to the usual decline rate, a corpus is about something the taxonomy
/// lacks rather than merely hard to read.
pub const UNUSUALLY_DECLINED: f64 = 1.2;

impl ReadReport {
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

    let answers = read_distinct_claims(model, &snapshot, options, &mut on_progress)?;
    let counted = count_reviews(app_id, &snapshot, options, &answers, &mut on_progress)?;
    let captured = crate::report::crawl_facts(&options.out_dir, app_id)?;

    Ok(ReadReport {
        elapsed: started.elapsed(),
        device: model.device().to_owned(),
        threshold: model.provenance().threshold,
        model: model.provenance().trained_from.clone(),
        trained_on: model.provenance().data_fingerprint.clone(),
        usual_declined: model.provenance().usual_declined,
        spine_version: model.provenance().spine_version.clone(),
        captured_unix: captured.changed_unix(),
        ..counted
    })
}

/// Asks the model about every distinct claim in the corpus.
fn read_distinct_claims(
    model: &mut ClaimReader,
    snapshot: &Path,
    options: &ReadOptions,
    on_progress: &mut impl FnMut(ReadProgress),
) -> Result<HashMap<[u8; 32], Reading>> {
    let mut answers: HashMap<[u8; 32], Reading> = HashMap::new();
    let mut window: Vec<([u8; 32], String)> = Vec::with_capacity(LENGTH_WINDOW);

    crate::capture::for_each_body(snapshot, |_, language, text| {
        // Reading a claim nothing will count is a forward pass for nothing, and on a corpus
        // where the named language is a third of the reviews it is most of the work.
        if options
            .language
            .as_ref()
            .is_some_and(|wanted| wanted != language)
        {
            return Ok(());
        }
        for claim in options.depth.claims_of(text) {
            let key = crate::embed::sha256_bytes(&claim);
            if answers.contains_key(&key) {
                continue;
            }
            // Reserved immediately, so a claim repeated later in the same window is not
            // queued twice. The reading is filled in when the window drains.
            answers.insert(key, Reading::default());
            window.push((key, claim.into_owned()));
            if window.len() >= LENGTH_WINDOW {
                drain(model, options.batch_size, &mut window, &mut answers)?;
                on_progress(ReadProgress {
                    done: answers.len() as u64,
                    total: 0,
                    reading_claims: true,
                });
            }
        }
        Ok(())
    })?;

    drain(model, options.batch_size, &mut window, &mut answers)?;
    Ok(answers)
}

/// Claims held back before a run of batches, so they can be sorted by length first.
///
/// Every batch pads to its longest member, so a batch holding one long claim and a hundred
/// two-word ones costs as much as a hundred long ones. Sorting a window before cutting it
/// into batches puts claims of a size together, and on a corpus of mostly short claims that
/// is most of the arithmetic. The embedding pass has done this since a million-review game
/// took a day; this pass was missing it.
const LENGTH_WINDOW: usize = 16_384;

fn drain(
    model: &mut ClaimReader,
    batch_size: usize,
    window: &mut Vec<([u8; 32], String)>,
    answers: &mut HashMap<[u8; 32], Reading>,
) -> Result<()> {
    if window.is_empty() {
        return Ok(());
    }
    window.sort_unstable_by_key(|(_, claim)| claim.len());

    for chunk in window.chunks(batch_size.max(1)) {
        let texts: Vec<String> = chunk.iter().map(|(_, claim)| claim.clone()).collect();
        for ((key, _), reading) in chunk.iter().zip(model.read(&texts)?) {
            answers.insert(*key, reading);
        }
    }
    window.clear();
    Ok(())
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

fn judge(text: &str, depth: Depth, answers: &HashMap<[u8; 32], Reading>) -> Verdict {
    let mut praise = vec![false; CORE_SPINE.len()];
    let mut complaint = vec![false; CORE_SPINE.len()];
    let mut seen = vec![false; CORE_SPINE.len()];
    let mut primary = None;
    let mut best = f32::NEG_INFINITY;
    let mut claims = 0;
    let mut unclassified = 0;

    for claim in depth.claims_of(text) {
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
    let mut calendar: HashMap<String, Month> = HashMap::new();
    let mut top: crate::bounded::Smallest<std::cmp::Reverse<u64>, Vec<usize>> =
        crate::bounded::Smallest::new(options.top_helpful);
    let mut rows = ReadingRows::default();
    let mut said = crate::said::Said::new(CORE_SPINE.len());

    let mut reviews = 0_u64;
    let mut corpus_reviews = 0_u64;
    let mut claims = 0_u64;
    let mut unclassified = 0_u64;
    let mut silent = 0_u64;
    let mut positive = 0_u64;

    crate::capture::for_each_row(snapshot, |row, text| {
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

        let verdict = judge(text, options.depth, answers);
        claims += verdict.claims as u64;
        unclassified += verdict.unclassified as u64;
        if verdict.subjects.is_empty() {
            silent += 1;
        }
        if let Some(primary) = verdict.primary {
            tallies[primary].primary_reviews += 1;
        }
        let month = calendar
            .entry(crate::time::year_month(row.created))
            .or_insert_with(|| Month {
                label: crate::time::year_month(row.created),
                reviews: 0,
                positive: 0,
                subjects: vec![0; CORE_SPINE.len()],
            });
        month.reviews += 1;
        if row.voted_up {
            month.positive += 1;
        }
        for &subject in &verdict.subjects {
            month.subjects[subject] += 1;
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
        for (index, claim) in options.depth.claims_of(text).into_iter().enumerate() {
            let reading = answers.get(&crate::embed::sha256_bytes(&claim));
            if let Some(reading) = reading
                && let Some(subject) = reading.subject
            {
                tallies[subject].claims += 1;
                said.note(subject, reading.polarity, &claim);
            }
            rows.push(&row.recommendationid, index, reading);
        }
        said.next_review();

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
    let mut months: Vec<Month> = calendar.into_values().collect();
    months.sort_by(|left, right| left.label.cmp(&right.label));

    Ok(ReadReport {
        app_id,
        reviews,
        corpus_reviews,
        language: options.language.clone(),
        depth: options.depth,
        claims,
        unclassified_claims: unclassified,
        silent_reviews: silent,
        positive,
        top_helpful: top_reviews.len() as u64,
        model: String::new(),
        trained_on: String::new(),
        usual_declined: None,
        spine_version: String::new(),
        threshold: 0.0,
        device: String::new(),
        captured_unix: 0,
        subjects: CORE_SPINE
            .iter()
            .zip(&tallies)
            .map(|(category, tally)| SubjectCount {
                id: category.id.to_owned(),
                label: category.label.to_owned(),
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
        said: said.finish(
            &CORE_SPINE
                .iter()
                .map(|c| (c.id, c.label))
                .collect::<Vec<_>>(),
        ),
        languages: ranked,
        months,
        elapsed: Duration::default(),
    })
}

/// Streams every stored reading, one claim at a time.
///
/// # Errors
///
/// Fails if the file is missing or was written by an older build.
pub fn for_each_reading(
    path: &Path,
    mut visit: impl FnMut(&str, u16, Option<&str>, f32, &str),
) -> Result<()> {
    use arrow::array::{Array, Float32Array, StringArray, UInt16Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = std::fs::File::open(path).map_err(|_| crate::Error::NoClassifications {
        path: path.to_path_buf(),
    })?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(8192)
        .build()?;

    for batch in reader {
        let batch = batch?;
        let column = |name: &'static str| -> Result<&dyn Array> {
            batch
                .column_by_name(name)
                .map(AsRef::as_ref)
                .ok_or(crate::Error::MalformedPayload { field: name })
        };
        let ids = column("recommendationid")?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(crate::Error::MalformedPayload {
                field: "recommendationid",
            })?;
        let indexes = column("claim_index")?
            .as_any()
            .downcast_ref::<UInt16Array>()
            .ok_or(crate::Error::MalformedPayload {
                field: "claim_index",
            })?;
        let subjects = column("subject")?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(crate::Error::MalformedPayload { field: "subject" })?;
        let confidences = column("confidence")?
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or(crate::Error::MalformedPayload {
                field: "confidence",
            })?;
        let polarities = column("polarity")?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(crate::Error::MalformedPayload { field: "polarity" })?;

        for row in 0..batch.num_rows() {
            visit(
                ids.value(row),
                indexes.value(row),
                (!subjects.is_null(row)).then(|| subjects.value(row)),
                confidences.value(row),
                polarities.value(row),
            );
        }
    }
    Ok(())
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
            reading
                .map_or(Polarity::Neutral, |reading| reading.polarity)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shallow_reads_a_review_as_one_point_and_deep_as_several() {
        let text = "The art is stunning. The story is a mess. Runs fine on a 3070.";
        assert_eq!(Depth::Shallow.claims_of(text).len(), 1);
        assert_eq!(Depth::Deep.claims_of(text).len(), 3);
    }

    #[test]
    fn neither_depth_invents_a_point_from_nothing() {
        assert!(Depth::Shallow.claims_of("   \n  ").is_empty());
        assert!(Depth::Deep.claims_of("   \n  ").is_empty());
    }

    #[test]
    fn a_reading_written_before_depth_existed_reads_back_as_deep() {
        // Every reading on disk before this field was deep, and a missing field must say so
        // rather than fail, or every existing corpus would need re-reading to open.
        let stored = serde_json::json!({
            "app_id": 1, "reviews": 1, "corpus_reviews": 1, "language": null, "claims": 1,
            "unclassified_claims": 0, "silent_reviews": 0, "positive": 1, "top_helpful": 1,
            "model": "m", "spine_version": "core-5", "threshold": 0.5, "device": "cpu",
            "subjects": [], "languages": [], "months": []
        });
        let found: ReadReport = serde_json::from_value(stored).expect("an older reading opens");
        assert_eq!(found.depth, Depth::Deep);
        assert!(found.trained_on.is_empty());
    }

    #[test]
    fn a_corpus_declined_far_above_usual_is_a_corpus_about_something_missing() {
        let mut found = ReadReport {
            app_id: 1,
            reviews: 100,
            corpus_reviews: 100,
            language: None,
            depth: Depth::Deep,
            claims: 1_000,
            unclassified_claims: 900,
            silent_reviews: 0,
            positive: 50,
            top_helpful: 10,
            model: String::new(),
            trained_on: String::new(),
            usual_declined: Some(0.73),
            spine_version: String::new(),
            threshold: 0.5,
            device: String::new(),
            captured_unix: 0,
            subjects: Vec::new(),
            said: Vec::new(),
            languages: Vec::new(),
            months: Vec::new(),
            elapsed: Duration::ZERO,
        };
        let ratio = found
            .declined_against_usual()
            .expect("a usual figure is carried");
        assert!(
            ratio >= UNUSUALLY_DECLINED,
            "90% against a usual 73% is {ratio}"
        );

        found.unclassified_claims = 700;
        assert!(found.declined_against_usual().unwrap() < UNUSUALLY_DECLINED);

        // A reader exported before the figure existed cannot make the comparison, and says
        // nothing rather than comparing against zero.
        found.usual_declined = None;
        assert_eq!(found.declined_against_usual(), None);
    }

    #[test]
    fn the_depth_a_reading_was_made_at_travels_with_it() {
        let stored = serde_json::json!({
            "app_id": 1, "reviews": 1, "corpus_reviews": 1, "language": null, "claims": 1,
            "depth": "shallow",
            "unclassified_claims": 0, "silent_reviews": 0, "positive": 1, "top_helpful": 1,
            "model": "m", "spine_version": "core-5", "threshold": 0.5, "device": "cpu",
            "subjects": [], "languages": [], "months": []
        });
        let found: ReadReport = serde_json::from_value(stored).expect("a shallow reading opens");
        assert_eq!(found.depth, Depth::Shallow);
    }
}
