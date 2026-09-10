//! Assigning categories to reviews, and the numbers that fall out of it.
//!
//! Every review is compared against one vector per category and keeps the categories it
//! sits nearest. Where those vectors come from is [`crate::anchors`]: either the written
//! category descriptions, which need no labels and no setup, or a set fitted from labelled
//! reviews, which is measurably better and needs a reference set to exist first.
//!
//! Nothing here corrects for classifier error. What that error is, on the corpora where it
//! has been measured, is what `census evaluate` reports.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use arrow::{
    array::{Array, ArrayRef, Float32Builder, ListBuilder, StringBuilder, UInt32Builder},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use parquet::{
    arrow::{ArrowWriter, arrow_reader::ParquetRecordBatchReaderBuilder},
    basic::{Compression, ZstdLevel},
    file::properties::WriterProperties,
};

use crate::{
    Error, Result,
    anchors::Anchors,
    taxonomy::{CORE_SPINE, CORE_SPINE_VERSION},
};

/// How close to the best-matching category another must score to count as mentioned.
///
/// A spread-based rule (say, everything more than one standard deviation above a review's
/// own mean) sounds adaptive but is not: with a fixed number of categories it selects
/// roughly a fixed number of them for every review, whatever the review says. Measured on a
/// real corpus it gave 2.7 categories to nearly every review and a single category to only
/// 5% of them, which invents the secondary topics it claims to detect.
///
/// Distance from the top does not have that failure. A review about one thing leaves a wide
/// gap to the runner-up and keeps one category; a review about two things scores both
/// closely and keeps both.
///
/// Swept against a 2,909-review corpus, the share of reviews keeping a single category ran
/// 45.8% at 0.01, 59.5% at 0.02 and 96.1% at 0.04. The widest setting abolishes secondary
/// topics altogether, so this sits at the end that still detects them.
///
/// This is the fallback for description anchors, fitted to the shape of one small corpus
/// rather than to measured agreement. A fitted anchor set carries its own margin, chosen by
/// cross-validation against labels, and [`Anchors::mention_margin`] should be preferred
/// whenever one exists.
pub const DEFAULT_MENTION_MARGIN: f32 = 0.01;

/// Reviews taken as "the top of the pile" when measuring helpfulness bias. Steam's own
/// default view shows a page of this order, which is what most people actually read.
pub const DEFAULT_TOP_HELPFUL: usize = 50;

/// Beyond this many near-tied categories, a review is treated as being about its best match
/// alone rather than about everything it happens to sit near.
const MAX_MENTIONS: usize = 4;

/// Rows per Parquet row group, which is what the writer buffers before flushing.
const ROWS_PER_ROW_GROUP: usize = 262_144;

#[derive(Debug, Clone)]
pub struct ClassifyOptions {
    pub out_dir: PathBuf,
    pub mention_margin: f32,
    pub top_helpful: usize,
}

impl Default for ClassifyOptions {
    fn default() -> Self {
        Self {
            out_dir: PathBuf::from("data"),
            mention_margin: DEFAULT_MENTION_MARGIN,
            top_helpful: DEFAULT_TOP_HELPFUL,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ClassifyProgress {
    pub classified: u64,
    pub total: u64,
}

#[derive(Debug, Clone)]
pub struct CategoryStats {
    pub id: &'static str,
    pub label: &'static str,
    /// Reviews whose single main subject is this category. These sum to the review count.
    pub primary_count: u64,
    /// Reviews that say anything about this category. These sum to more than the review
    /// count, because a review can mention several things.
    pub mention_count: u64,
    /// Mentions among the most-helpful reviews only.
    pub top_mention_count: u64,
    /// Of the reviews mentioning this category, how many recommended the game.
    ///
    /// A topic people raise while recommending a game is a different thing from one they
    /// raise while refusing to, and a mention rate alone cannot tell the two apart.
    pub positive_mentions: u64,
}

impl CategoryStats {
    /// Share of the reviews mentioning this category that still recommended the game.
    #[must_use]
    pub fn positive_share(&self) -> Option<f64> {
        (self.mention_count > 0).then(|| ratio(self.positive_mentions, self.mention_count))
    }

    /// Share of all reviews that mention this category. The headline figure.
    #[must_use]
    pub fn mention_rate(&self, reviews: u64) -> f64 {
        ratio(self.mention_count, reviews)
    }

    /// Share of all reviews whose main subject is this category.
    #[must_use]
    pub fn primary_share(&self, reviews: u64) -> f64 {
        ratio(self.primary_count, reviews)
    }

    /// Share of the most-helpful reviews that mention this category.
    #[must_use]
    pub fn top_mention_rate(&self, top_n: u64) -> f64 {
        ratio(self.top_mention_count, top_n)
    }

    /// How much reading only the top of the pile overstates this category.
    ///
    /// This is the project's whole thesis expressed as a number: above 1.0 means the
    /// most-helpful reviews talk about this more than players in general do.
    #[must_use]
    pub fn bias_factor(&self, reviews: u64, top_n: u64) -> Option<f64> {
        let overall = self.mention_rate(reviews);
        (overall > 0.0).then(|| self.top_mention_rate(top_n) / overall)
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "review counts are far below 2^53"
)]
fn ratio(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        return 0.0;
    }
    part as f64 / whole as f64
}

#[derive(Debug, Clone)]
pub struct ClassifyReport {
    pub app_id: u32,
    pub reviews: u64,
    /// Classified reviews that recommended the game. The line every category's own share is
    /// read against, counted over exactly the reviews the categories were counted over.
    pub positive: u64,
    pub unmatched: u64,
    pub top_helpful: u64,
    pub categories: Vec<CategoryStats>,
    pub spine_version: &'static str,
    /// The encoder behind the anchors, which is also the encoder behind the vectors: the
    /// two are checked against each other before a single review is placed.
    pub model: String,
    /// Apps whose labels fitted the anchors. Empty when the written descriptions were used.
    pub anchors_fitted_from: Vec<u32>,
    /// How close to the best match a category had to score to count as mentioned.
    pub mention_margin: f32,
    /// Reviews per language, most common first. Empty strings are grouped as unknown.
    pub languages: Vec<(String, u64)>,
    /// What was said month by month, oldest first.
    pub months: Vec<Month>,
    /// The most-helpful reviews and what they were about, so a reader can be shown the top
    /// of the pile rather than only told how far it differs from everyone else.
    pub top_reviews: Vec<TopReview>,
    pub elapsed: Duration,
    pub path: PathBuf,
}

impl ClassifyReport {
    /// Writes the counts beside the assignments they were computed from.
    ///
    /// These numbers are the point of the tool and used to exist only in whatever terminal
    /// happened to be open. Storing them makes a run inspectable afterwards, lets `census
    /// report` render without redoing the pass, and records which anchors and which encoder
    /// produced them next to the counts rather than in a shell history.
    ///
    /// # Errors
    ///
    /// Fails if the file cannot be written.
    pub fn save(&self, path: &Path) -> Result<()> {
        let categories: Vec<serde_json::Value> = self
            .categories
            .iter()
            .map(|c| {
                serde_json::json!({
                    "id": c.id,
                    "label": c.label,
                    "primary_count": c.primary_count,
                    "mention_count": c.mention_count,
                    "top_mention_count": c.top_mention_count,
                    "positive_mentions": c.positive_mentions,
                })
            })
            .collect();
        let top: Vec<serde_json::Value> = self
            .top_reviews
            .iter()
            .map(|r| {
                serde_json::json!({
                    "id": r.id,
                    "helpfulness": r.helpfulness,
                    "votes_up": r.votes_up,
                    "primary": CORE_SPINE[r.primary].id,
                    "mentions": category_ids(r.mentions),
                })
            })
            .collect();
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "app_id": self.app_id,
                "reviews": self.reviews,
                "positive": self.positive,
                "unmatched": self.unmatched,
                "top_helpful": self.top_helpful,
                "mention_margin": self.mention_margin,
                "spine_version": self.spine_version,
                "model": self.model,
                "anchors_fitted_from": self.anchors_fitted_from,
                "categories": categories,
                "languages": self.languages,
                "months": self.months.iter().map(|month| serde_json::json!({
                    "label": month.label,
                    "reviews": month.reviews,
                    "positive": month.positive,
                    "categories": month.categories,
                })).collect::<Vec<_>>(),
                "top_reviews": top,
            }))?,
        )?;
        Ok(())
    }
}

/// Everything counted while walking the corpus once.
///
/// Kept together because they are: one pass over every review feeds all of them, and a
/// counter that lives somewhere else is a counter that can be forgotten in a new branch.
struct Tallies {
    categories: Vec<CategoryStats>,
    /// What language a review is written in is not decoration. Steam's own default shows a
    /// reader only their own language, so a corpus that keeps every language is measuring
    /// something the store's own page cannot.
    languages: HashMap<String, u64>,
    /// A rate is one number for a corpus that took years to accumulate. A game review-bombed
    /// in one month and quiet since reads identically to one grumbled about steadily, and
    /// the two are not the same fact about anything.
    calendar: HashMap<String, MonthTally>,
    classified: u64,
    positive: u64,
}

impl Tallies {
    fn new() -> Self {
        Self {
            categories: empty_stats(),
            languages: HashMap::new(),
            calendar: HashMap::new(),
            classified: 0,
            positive: 0,
        }
    }

    fn record(&mut self, review: &ReviewRow, primary: usize, mentions: u32) {
        self.classified += 1;
        if review.voted_up {
            self.positive += 1;
        }
        self.categories[primary].primary_count += 1;
        for (index, stat) in self.categories.iter_mut().enumerate() {
            if mentions & (1 << index) != 0 {
                stat.mention_count += 1;
                if review.voted_up {
                    stat.positive_mentions += 1;
                }
            }
        }
        *self.languages.entry(review.language.clone()).or_default() += 1;

        let month = self
            .calendar
            .entry(crate::time::year_month(review.created))
            .or_insert_with(|| MonthTally {
                reviews: 0,
                positive: 0,
                categories: vec![0; CORE_SPINE.len()],
            });
        month.reviews += 1;
        if review.voted_up {
            month.positive += 1;
        }
        for (index, count) in month.categories.iter_mut().enumerate() {
            if mentions & (1 << index) != 0 {
                *count += 1;
            }
        }
    }
}

/// One month of a corpus.
#[derive(Debug, Clone)]
pub struct Month {
    /// `2024-02`, which sorts as it reads.
    pub label: String,
    pub reviews: u64,
    pub positive: u64,
    /// Mentions per category, in taxonomy order.
    pub categories: Vec<u64>,
}

struct MonthTally {
    reviews: u64,
    positive: u64,
    categories: Vec<u64>,
}

/// Months oldest first, with the gaps left as gaps: a month nobody reviewed in is a fact
/// about the game, and inventing a zero row for it would say the same thing less clearly.
fn by_month(tallies: HashMap<String, MonthTally>) -> Vec<Month> {
    let mut months: Vec<Month> = tallies
        .into_iter()
        .map(|(label, tally)| Month {
            label,
            reviews: tally.reviews,
            positive: tally.positive,
            categories: tally.categories,
        })
        .collect();
    months.sort_by(|left, right| left.label.cmp(&right.label));
    months
}

/// Languages by how many reviews are written in them, most first.
fn ranked(counts: HashMap<String, u64>) -> Vec<(String, u64)> {
    let mut ranked: Vec<(String, u64)> = counts
        .into_iter()
        .map(|(language, count)| {
            let named = if language.is_empty() {
                "unknown".to_owned()
            } else {
                language
            };
            (named, count)
        })
        .collect();
    ranked.sort_by(|(left_name, left), (right_name, right)| {
        right.cmp(left).then_with(|| left_name.cmp(right_name))
    });
    ranked
}

/// The categories a mention bitmask names, in taxonomy order.
fn category_ids(mentions: u32) -> Vec<&'static str> {
    CORE_SPINE
        .iter()
        .enumerate()
        .filter(|(index, _)| mentions & (1 << index) != 0)
        .map(|(_, category)| category.id)
        .collect()
}

/// Assigns core-spine categories to every review in the most recent capture.
///
/// # Errors
///
/// Fails if the capture or its embeddings are missing, or if reading or writing fails.
pub fn classify_corpus(
    anchors: &Anchors,
    app_id: u32,
    options: &ClassifyOptions,
    mut on_progress: impl FnMut(ClassifyProgress),
) -> Result<ClassifyReport> {
    let started = Instant::now();
    let snapshot = crate::embed::latest_snapshot(&options.out_dir, app_id)?;
    refuse_a_foreign_encoder(anchors, &options.out_dir, app_id)?;
    let (by_hash, reviews) = group_reviews_by_text(&snapshot)?;

    let mut tallies = Tallies::new();
    let provenance = anchor_provenance(anchors);
    let path = snapshot.join("classifications.parquet");
    let schema = classification_schema();
    let mut writer = ArrowWriter::try_new(
        std::fs::File::create(&path)?,
        Arc::clone(&schema),
        Some(
            WriterProperties::builder()
                .set_compression(Compression::ZSTD(ZstdLevel::default()))
                .set_max_row_group_row_count(Some(ROWS_PER_ROW_GROUP))
                .build(),
        ),
    )?;

    let mut top = TopOfThePile::new(options.top_helpful);
    let mut pending = Batch::default();

    crate::embed::for_each_vector(&snapshot, |hash, vector| {
        let Some(group) = by_hash.get(hash) else {
            return Ok(());
        };
        let sims = anchors.similarities(vector);
        let (primary, mentions) = assign(&sims, options.mention_margin);

        for review in group {
            tallies.record(review, primary, mentions);
            top.offer(TopReview {
                id: review.recommendationid.clone(),
                helpfulness: review.helpfulness,
                votes_up: review.votes_up,
                primary,
                mentions,
            });
            pending.push(app_id, review, primary, sims[primary], mentions);
        }

        if pending.len() >= 8192 {
            writer.write(&pending.take(&schema, &provenance)?)?;
            on_progress(ClassifyProgress {
                classified: tallies.classified,
                total: reviews,
            });
        }
        Ok(())
    })?;
    if pending.len() > 0 {
        writer.write(&pending.take(&schema, &provenance)?)?;
    }
    writer.close()?;

    let top_reviews = top.take();
    for review in &top_reviews {
        for (index, stat) in tallies.categories.iter_mut().enumerate() {
            if review.mentions & (1 << index) != 0 {
                stat.top_mention_count += 1;
            }
        }
    }
    let classified = tallies.classified;

    let report = ClassifyReport {
        app_id,
        reviews,
        positive: tallies.positive,
        unmatched: reviews.saturating_sub(classified),
        top_helpful: classified.min(u64::try_from(options.top_helpful).unwrap_or(u64::MAX)),
        mention_margin: options.mention_margin,
        categories: tallies.categories,
        languages: ranked(tallies.languages),
        months: by_month(tallies.calendar),
        top_reviews,
        spine_version: CORE_SPINE_VERSION,
        model: anchors.model.clone(),
        anchors_fitted_from: anchors.fitted_from.clone(),
        elapsed: started.elapsed(),
        path,
    };
    report.save(&snapshot.join("classification.json"))?;
    Ok(report)
}

/// One zeroed counter per category, in taxonomy order.
fn empty_stats() -> Vec<CategoryStats> {
    CORE_SPINE
        .iter()
        .map(|c| CategoryStats {
            id: c.id,
            label: c.label,
            primary_count: 0,
            mention_count: 0,
            top_mention_count: 0,
            positive_mentions: 0,
        })
        .collect()
}

/// Refuses anchors built by an encoder other than the one behind the corpus.
///
/// Two encoders put the same review in different places, and nothing about a cosine between
/// vectors from different spaces looks wrong. It would simply be meaningless.
fn refuse_a_foreign_encoder(anchors: &Anchors, out_dir: &Path, app_id: u32) -> Result<()> {
    let corpus_encoder = crate::embed::corpus_encoder(out_dir, app_id)?;
    if corpus_encoder == anchors.model {
        return Ok(());
    }
    Err(Error::StaleAnchors {
        field: "embedding model",
        expected: corpus_encoder,
        actual: anchors.model.clone(),
    })
}

/// Reads the capture into memory, grouped by the hash of the review text.
///
/// The join runs review-side-in-memory and vector-side-streamed. A vector is a couple of
/// kilobytes and a review a few dozen bytes, so holding the reviews costs a fraction of what
/// holding the vectors would on a million-review corpus. Grouping by text also means each
/// distinct review is compared against the anchors once however many people posted it.
fn group_reviews_by_text(snapshot: &Path) -> Result<(HashMap<String, Vec<ReviewRow>>, u64)> {
    let mut by_hash: HashMap<String, Vec<ReviewRow>> = HashMap::new();
    let mut reviews = 0_u64;
    for_each_review(snapshot, |row, _| {
        reviews += 1;
        by_hash.entry(row.text_hash.clone()).or_default().push(row);
        Ok(())
    })?;
    Ok((by_hash, reviews))
}

/// Keeps the most-helpful reviews seen so far, and no more than that.
///
/// What a reader skimming the first page actually sees is a few dozen reviews, so sorting a
/// million of them to find fifty wastes the memory the streaming join was built to save.
/// This holds a small buffer and prunes it whenever it grows past twice the limit, which
/// costs an occasional sort of a hundred items instead of one sort of the corpus.
struct TopOfThePile {
    limit: usize,
    kept: Vec<TopReview>,
}

/// One of the most-helpful reviews, kept so a reader can be shown what the top of the pile
/// actually says rather than only how far it differs from the corpus.
#[derive(Debug, Clone)]
pub struct TopReview {
    pub id: String,
    /// Steam's own helpfulness score, which is what orders the page people read.
    pub helpfulness: f64,
    pub votes_up: u32,
    /// Index into [`CORE_SPINE`].
    pub primary: usize,
    /// One bit per category, primary included.
    pub mentions: u32,
}

impl TopOfThePile {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            kept: Vec::with_capacity(limit.saturating_mul(2).min(4096)),
        }
    }

    fn offer(&mut self, review: TopReview) {
        if self.limit == 0 {
            return;
        }
        self.kept.push(review);
        if self.kept.len() >= self.limit.saturating_mul(2) {
            self.prune();
        }
    }

    fn prune(&mut self) {
        self.kept
            .sort_unstable_by(|a, b| b.helpfulness.total_cmp(&a.helpfulness));
        self.kept.truncate(self.limit);
    }

    fn take(mut self) -> Vec<TopReview> {
        self.prune();
        self.kept
    }
}

/// Picks the main category and any other the review genuinely also covers.
///
/// A category counts only if it scores within `margin` of the best match, so how many a
/// review gets depends on how close its scores are rather than on how many categories exist.
pub(crate) fn assign(sims: &[f32], margin: f32) -> (usize, u32) {
    let (primary, best) = sims
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or((0, 0.0), |(index, score)| (index, *score));

    // A category that can only stand alone says no aspect was named, which nothing else can
    // be true alongside. The taxonomy says so and the labellers apply it without being asked:
    // "says nothing about the game" is the only label on every review that carries it.
    if crate::taxonomy::CORE_SPINE
        .get(primary)
        .is_some_and(|c| c.alone)
    {
        return (primary, 1_u32 << primary);
    }

    let cutoff = best - margin;
    // The primary category is always a mention: a review is about whatever it is most about.
    let mut mentions = 1_u32 << primary;
    for (index, sim) in sims.iter().enumerate() {
        let alone = crate::taxonomy::CORE_SPINE
            .get(index)
            .is_some_and(|c| c.alone);
        if *sim >= cutoff && !alone {
            mentions |= 1 << index;
        }
    }

    // A review that scores near-identically against many categories is not about all of
    // them; it is about none of them clearly. Claiming every one would be inventing topics,
    // so diffuse evidence falls back to the single best match.
    if mentions.count_ones() as usize > MAX_MENTIONS {
        return (primary, 1_u32 << primary);
    }
    (primary, mentions)
}

pub(crate) struct ReviewRow {
    pub(crate) recommendationid: String,
    pub(crate) text_hash: String,
    pub(crate) helpfulness: f64,
    pub(crate) votes_up: u32,
    pub(crate) voted_up: bool,
    pub(crate) language: String,
    pub(crate) created: i64,
}

#[derive(Default)]
struct Batch {
    ids: Vec<String>,
    appids: Vec<u32>,
    primaries: Vec<&'static str>,
    scores: Vec<f32>,
    mentions: Vec<u32>,
}

/// How the anchors used for a run are named in the output, so a later reader can tell a
/// zero-shot classification from a fitted one without being told which it is looking at.
fn anchor_provenance(anchors: &Anchors) -> String {
    if anchors.fitted_from.is_empty() {
        return "descriptions".to_owned();
    }
    let apps: Vec<String> = anchors
        .fitted_from
        .iter()
        .map(ToString::to_string)
        .collect();
    format!("fitted:{}", apps.join("+"))
}

impl Batch {
    fn len(&self) -> usize {
        self.ids.len()
    }

    fn push(&mut self, app_id: u32, review: &ReviewRow, primary: usize, score: f32, mentions: u32) {
        self.ids.push(review.recommendationid.clone());
        self.appids.push(app_id);
        self.primaries.push(CORE_SPINE[primary].id);
        self.scores.push(score);
        self.mentions.push(mentions);
    }

    fn take(&mut self, schema: &Arc<Schema>, anchors: &str) -> Result<RecordBatch> {
        let mut ids = StringBuilder::new();
        let mut appids = UInt32Builder::new();
        let mut primaries = StringBuilder::new();
        let mut scores = Float32Builder::new();
        let mut mentions = ListBuilder::new(StringBuilder::new());
        let mut spine = StringBuilder::new();
        let mut source = StringBuilder::new();

        for index in 0..self.len() {
            ids.append_value(&self.ids[index]);
            appids.append_value(self.appids[index]);
            primaries.append_value(self.primaries[index]);
            scores.append_value(self.scores[index]);
            for (bit, category) in CORE_SPINE.iter().enumerate() {
                if self.mentions[index] & (1 << bit) != 0 {
                    mentions.values().append_value(category.id);
                }
            }
            mentions.append(true);
            spine.append_value(CORE_SPINE_VERSION);
            source.append_value(anchors);
        }
        self.ids.clear();
        self.appids.clear();
        self.primaries.clear();
        self.scores.clear();
        self.mentions.clear();

        let columns: Vec<ArrayRef> = vec![
            Arc::new(ids.finish()),
            Arc::new(appids.finish()),
            Arc::new(primaries.finish()),
            Arc::new(scores.finish()),
            Arc::new(mentions.finish()),
            Arc::new(spine.finish()),
            Arc::new(source.finish()),
        ];
        Ok(RecordBatch::try_new(Arc::clone(schema), columns)?)
    }
}

fn classification_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("recommendationid", DataType::Utf8, false),
        Field::new("appid", DataType::UInt32, false),
        Field::new("primary_category", DataType::Utf8, false),
        Field::new("primary_score", DataType::Float32, false),
        Field::new(
            "mentions",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            false,
        ),
        Field::new("spine_version", DataType::Utf8, false),
        Field::new("anchors", DataType::Utf8, false),
    ]))
}

fn column<'a, T: 'static>(batch: &'a RecordBatch, name: &'static str) -> Result<&'a T> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<T>())
        .ok_or(Error::MalformedPayload { field: name })
}

/// Visits every review in a snapshot, one at a time.
///
/// A visitor rather than an iterator because the iterator this replaced built the whole
/// corpus in a `Vec` before yielding its first item, which reads as streaming and is not.
pub(crate) fn for_each_review(
    snapshot: &Path,
    mut visit: impl FnMut(ReviewRow, &str) -> Result<()>,
) -> Result<()> {
    use arrow::array::{BooleanArray, Float64Array, Int64Array, StringArray, UInt32Array};

    let mut shards: Vec<PathBuf> = std::fs::read_dir(snapshot)?
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
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
            let ids = column::<StringArray>(&batch, "recommendationid")?;
            let texts = column::<StringArray>(&batch, "review")?;
            let helpful = column::<Float64Array>(&batch, "weighted_vote_score")?;
            let votes = column::<UInt32Array>(&batch, "votes_up")?;
            let languages = column::<StringArray>(&batch, "language")?;
            let recommended = column::<BooleanArray>(&batch, "voted_up")?;
            let created = column::<Int64Array>(&batch, "timestamp_created")?;
            for row in 0..batch.num_rows() {
                if texts.is_null(row) || texts.value(row).trim().is_empty() {
                    continue;
                }
                let body = texts.value(row);
                visit(
                    ReviewRow {
                        recommendationid: ids.value(row).to_owned(),
                        text_hash: crate::embed::sha256_hex(body),
                        helpfulness: if helpful.is_null(row) {
                            0.0
                        } else {
                            helpful.value(row)
                        },
                        votes_up: if votes.is_null(row) {
                            0
                        } else {
                            votes.value(row)
                        },
                        voted_up: !recommended.is_null(row) && recommended.value(row),
                        language: if languages.is_null(row) {
                            String::new()
                        } else {
                            languages.value(row).to_owned()
                        },
                        created: if created.is_null(row) {
                            0
                        } else {
                            created.value(row)
                        },
                    },
                    body,
                )?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_review_with_no_text_is_not_classified_at_all() {
        // An empty review carries no evidence about any category, but it still embeds to
        // something, and that something lands on whichever anchor happens to sit nearest.
        // Across six corpora it put 8,673 blank reviews into a single category, every one of
        // them, which is a category's rate turned into an artefact of the corpus's silence.
        let dir = std::env::temp_dir().join("census-blank-review-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shard-0000.parquet");

        let mut writer = crate::capture::CaptureWriter::create(&path, 1).unwrap();
        let reviews = [
            serde_json::json!({"recommendationid": "1", "review": "the frame rate is awful"}),
            serde_json::json!({"recommendationid": "2", "review": ""}),
            serde_json::json!({"recommendationid": "3", "review": "   \n  "}),
            serde_json::json!({"recommendationid": "4", "review": "great game"}),
        ];
        writer.write(&reviews.iter().collect::<Vec<_>>()).unwrap();
        writer.close().unwrap();

        let mut seen = Vec::new();
        for_each_review(&dir, |row, _| {
            seen.push(row.recommendationid);
            Ok(())
        })
        .unwrap();

        assert_eq!(seen, vec!["1".to_owned(), "4".to_owned()]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_review_about_one_thing_gets_one_category() {
        let mut sims = vec![0.70; CORE_SPINE.len()];
        sims[3] = 0.95;
        let (primary, mentions) = assign(&sims, 0.02);
        assert_eq!(primary, 3);
        assert_eq!(mentions.count_ones(), 1, "a secondary topic was invented");
    }

    #[test]
    fn a_review_covering_two_things_records_both() {
        let mut sims = vec![0.70; CORE_SPINE.len()];
        sims[2] = 0.95;
        sims[7] = 0.94;
        let (primary, mentions) = assign(&sims, 0.02);
        assert_eq!(primary, 2);
        assert!(mentions & (1 << 7) != 0, "secondary topic was dropped");
        assert_eq!(mentions.count_ones(), 2);
    }

    #[test]
    fn a_near_miss_outside_the_margin_is_not_promoted() {
        let mut sims = vec![0.70; CORE_SPINE.len()];
        sims[2] = 0.95;
        sims[7] = 0.92;
        let (_, mentions) = assign(&sims, 0.02);
        assert_eq!(mentions.count_ones(), 1);
    }

    /// A review that says nothing about the game cannot also be about something, and a
    /// verdict with no reason names no aspect. Both are claims about absence.
    #[test]
    fn a_category_that_stands_alone_never_shares_a_review() {
        let slot = |id: &str| CORE_SPINE.iter().position(|c| c.id == id).expect(id);
        for exclusive in ["offtopic", "verdict"] {
            assert!(
                CORE_SPINE[slot(exclusive)].alone,
                "{exclusive} is meant to stand alone"
            );

            // Winning outright: everything else is dropped however close it came.
            let mut sims = vec![0.90; CORE_SPINE.len()];
            sims[slot(exclusive)] = 0.95;
            let (primary, mentions) = assign(&sims, 0.10);
            assert_eq!(primary, slot(exclusive));
            assert_eq!(
                mentions,
                1 << slot(exclusive),
                "{exclusive} took a second subject with it"
            );

            // Coming a close second to a real subject: it is not a second subject either.
            let mut sims = vec![0.10; CORE_SPINE.len()];
            sims[slot("bugs")] = 0.95;
            sims[slot(exclusive)] = 0.94;
            let (primary, mentions) = assign(&sims, 0.10);
            assert_eq!(primary, slot("bugs"));
            assert_eq!(
                mentions & (1 << slot(exclusive)),
                0,
                "{exclusive} was counted alongside a category it contradicts"
            );
        }
    }

    #[test]
    fn diffuse_evidence_falls_back_to_the_single_best_match() {
        // Equally similar to everything means clearly about nothing, so claiming all
        // sixteen categories would be inventing topics rather than detecting them.
        let sims = vec![0.80; CORE_SPINE.len()];
        let (_, mentions) = assign(&sims, 0.02);
        assert_eq!(mentions.count_ones(), 1);
    }

    #[test]
    fn bias_factor_expresses_how_much_the_top_of_the_pile_overstates() {
        let stat = CategoryStats {
            id: "bugs",
            label: "Bugs and crashes",
            primary_count: 100,
            mention_count: 247,
            top_mention_count: 32,
            positive_mentions: 61,
        };
        // 64% of the top 50 against 24.7% of all thousand: the README's own example.
        assert!((stat.mention_rate(1000) - 0.247).abs() < 1e-9);
        assert!((stat.top_mention_rate(50) - 0.64).abs() < 1e-9);
        let factor = stat.bias_factor(1000, 50).unwrap();
        assert!((factor - 2.591).abs() < 1e-3, "factor was {factor}");
        // Bugs are raised as often by people recommending the game as refusing to, which a
        // mention rate on its own cannot say either way.
        let positive = stat.positive_share().unwrap();
        assert!(
            (positive - 0.247).abs() < 1e-3,
            "positive share was {positive}"
        );
    }

    #[test]
    fn bias_factor_is_unknown_rather_than_infinite_when_nothing_mentions_it() {
        let stat = CategoryStats {
            id: "audio",
            label: "Audio and music",
            primary_count: 0,
            mention_count: 0,
            top_mention_count: 0,
            positive_mentions: 0,
        };
        assert_eq!(stat.bias_factor(1000, 50), None);
    }

    #[test]
    fn primary_shares_are_exhaustive_while_mention_rates_need_not_be() {
        // Two categories, every review assigned exactly one primary, some mentioning both.
        let stats = [
            CategoryStats {
                id: "a",
                label: "A",
                primary_count: 60,
                mention_count: 80,
                top_mention_count: 0,
                positive_mentions: 0,
            },
            CategoryStats {
                id: "b",
                label: "B",
                primary_count: 40,
                mention_count: 55,
                top_mention_count: 0,
                positive_mentions: 0,
            },
        ];
        let primary: f64 = stats.iter().map(|s| s.primary_share(100)).sum();
        let mentions: f64 = stats.iter().map(|s| s.mention_rate(100)).sum();
        assert!((primary - 1.0).abs() < 1e-9, "primary shares must sum to 1");
        assert!(mentions > 1.0, "mention rates are expected to exceed 1");
    }
}
