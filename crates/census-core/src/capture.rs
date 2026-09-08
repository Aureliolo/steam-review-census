//! Immutable Parquet capture of what Valve served.
//!
//! Every row carries `raw_json`, the review object exactly as it arrived, alongside the
//! typed columns. That redundancy is deliberate: a corpus is expensive to rebuild and, for
//! reviews deleted since the crawl, impossible. Re-parsing must always be an option, and a
//! fixed set of typed columns would silently discard fields Valve adds later.

use std::{fs::File, path::Path, sync::Arc};

use arrow::{
    array::{ArrayRef, BooleanBuilder, Float64Builder, Int64Builder, StringBuilder, UInt32Builder},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use parquet::{
    arrow::ArrowWriter,
    basic::{Compression, ZstdLevel},
    file::properties::WriterProperties,
};
use serde_json::Value;

use crate::{Error, Result};

/// Reviews per Parquet row group, which is what the writer buffers before flushing.
const REVIEWS_PER_ROW_GROUP: usize = 65_536;

#[must_use]
pub fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("recommendationid", DataType::Utf8, false),
        Field::new("appid", DataType::UInt32, false),
        Field::new("author_steamid", DataType::Utf8, true),
        Field::new("author_num_games_owned", DataType::UInt32, true),
        Field::new("author_num_reviews", DataType::UInt32, true),
        Field::new("author_playtime_forever", DataType::UInt32, true),
        Field::new("author_playtime_at_review", DataType::UInt32, true),
        Field::new("author_playtime_last_two_weeks", DataType::UInt32, true),
        Field::new("author_last_played", DataType::Int64, true),
        Field::new("language", DataType::Utf8, true),
        Field::new("review", DataType::Utf8, true),
        Field::new("timestamp_created", DataType::Int64, true),
        Field::new("timestamp_updated", DataType::Int64, true),
        Field::new("voted_up", DataType::Boolean, true),
        Field::new("votes_up", DataType::UInt32, true),
        Field::new("votes_funny", DataType::UInt32, true),
        Field::new("weighted_vote_score", DataType::Float64, true),
        Field::new("comment_count", DataType::UInt32, true),
        Field::new("steam_purchase", DataType::Boolean, true),
        Field::new("received_for_free", DataType::Boolean, true),
        Field::new("written_during_early_access", DataType::Boolean, true),
        Field::new("refunded", DataType::Boolean, true),
        Field::new("primarily_steam_deck", DataType::Boolean, true),
        Field::new("raw_json", DataType::Utf8, false),
    ]))
}

/// Writes reviews to a single Parquet file.
#[derive(Debug)]
pub struct CaptureWriter {
    writer: ArrowWriter<File>,
    schema: Arc<Schema>,
    app_id: u32,
    rows: u64,
}

impl CaptureWriter {
    /// # Errors
    ///
    /// Fails if the parent directory cannot be created, the file cannot be opened, or the
    /// Parquet writer rejects the schema.
    pub fn create(path: &Path, app_id: u32) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let schema = schema();
        // Buffered whole before it reaches the disk, so this bounds what a shard holding
        // tens of thousands of reviews costs in memory.
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::default()))
            .set_max_row_group_row_count(Some(REVIEWS_PER_ROW_GROUP))
            .build();
        let writer = ArrowWriter::try_new(File::create(path)?, Arc::clone(&schema), Some(props))?;
        Ok(Self {
            writer,
            schema,
            app_id,
            rows: 0,
        })
    }

    /// # Errors
    ///
    /// Fails if a review has no `recommendationid`, or if Arrow or Parquet reject the batch.
    pub fn write(&mut self, reviews: &[&Value]) -> Result<()> {
        if reviews.is_empty() {
            return Ok(());
        }
        let mut builders = RowBuilders::with_capacity(reviews.len());
        for review in reviews {
            builders.push(self.app_id, review)?;
        }
        self.writer.write(&builders.finish(&self.schema)?)?;
        self.rows += reviews.len() as u64;
        Ok(())
    }

    #[must_use]
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// # Errors
    ///
    /// Fails if the Parquet footer cannot be written.
    pub fn close(self) -> Result<u64> {
        self.writer.close()?;
        Ok(self.rows)
    }
}

struct RowBuilders {
    recommendationid: StringBuilder,
    appid: UInt32Builder,
    author_steamid: StringBuilder,
    num_games_owned: UInt32Builder,
    num_reviews: UInt32Builder,
    playtime_forever: UInt32Builder,
    playtime_at_review: UInt32Builder,
    playtime_last_two_weeks: UInt32Builder,
    last_played: Int64Builder,
    language: StringBuilder,
    review: StringBuilder,
    created: Int64Builder,
    updated: Int64Builder,
    voted_up: BooleanBuilder,
    votes_up: UInt32Builder,
    votes_funny: UInt32Builder,
    weighted_vote_score: Float64Builder,
    comment_count: UInt32Builder,
    steam_purchase: BooleanBuilder,
    received_for_free: BooleanBuilder,
    early_access: BooleanBuilder,
    refunded: BooleanBuilder,
    steam_deck: BooleanBuilder,
    raw_json: StringBuilder,
}

impl RowBuilders {
    fn with_capacity(n: usize) -> Self {
        Self {
            recommendationid: StringBuilder::with_capacity(n, n * 12),
            appid: UInt32Builder::with_capacity(n),
            author_steamid: StringBuilder::with_capacity(n, n * 18),
            num_games_owned: UInt32Builder::with_capacity(n),
            num_reviews: UInt32Builder::with_capacity(n),
            playtime_forever: UInt32Builder::with_capacity(n),
            playtime_at_review: UInt32Builder::with_capacity(n),
            playtime_last_two_weeks: UInt32Builder::with_capacity(n),
            last_played: Int64Builder::with_capacity(n),
            language: StringBuilder::with_capacity(n, n * 8),
            review: StringBuilder::with_capacity(n, n * 200),
            created: Int64Builder::with_capacity(n),
            updated: Int64Builder::with_capacity(n),
            voted_up: BooleanBuilder::with_capacity(n),
            votes_up: UInt32Builder::with_capacity(n),
            votes_funny: UInt32Builder::with_capacity(n),
            weighted_vote_score: Float64Builder::with_capacity(n),
            comment_count: UInt32Builder::with_capacity(n),
            steam_purchase: BooleanBuilder::with_capacity(n),
            received_for_free: BooleanBuilder::with_capacity(n),
            early_access: BooleanBuilder::with_capacity(n),
            refunded: BooleanBuilder::with_capacity(n),
            steam_deck: BooleanBuilder::with_capacity(n),
            raw_json: StringBuilder::with_capacity(n, n * 700),
        }
    }

    fn push(&mut self, app_id: u32, r: &Value) -> Result<()> {
        let id = text(r.get("recommendationid")).ok_or(Error::MalformedPayload {
            field: "recommendationid",
        })?;
        self.recommendationid.append_value(id);
        self.appid.append_value(app_id);

        let author = r.get("author");
        self.author_steamid
            .append_option(text(author.and_then(|a| a.get("steamid"))));
        self.num_games_owned
            .append_option(number(author.and_then(|a| a.get("num_games_owned"))));
        self.num_reviews
            .append_option(number(author.and_then(|a| a.get("num_reviews"))));
        self.playtime_forever
            .append_option(number(author.and_then(|a| a.get("playtime_forever"))));
        self.playtime_at_review
            .append_option(number(author.and_then(|a| a.get("playtime_at_review"))));
        self.playtime_last_two_weeks.append_option(number(
            author.and_then(|a| a.get("playtime_last_two_weeks")),
        ));
        self.last_played.append_option(
            author
                .and_then(|a| a.get("last_played"))
                .and_then(Value::as_i64),
        );

        self.language.append_option(text(r.get("language")));
        self.review.append_option(text(r.get("review")));
        self.created
            .append_option(r.get("timestamp_created").and_then(Value::as_i64));
        self.updated
            .append_option(r.get("timestamp_updated").and_then(Value::as_i64));
        self.voted_up
            .append_option(r.get("voted_up").and_then(Value::as_bool));
        self.votes_up.append_option(number(r.get("votes_up")));
        self.votes_funny.append_option(number(r.get("votes_funny")));
        self.weighted_vote_score
            .append_option(decimal(r.get("weighted_vote_score")));
        self.comment_count
            .append_option(number(r.get("comment_count")));
        self.steam_purchase
            .append_option(r.get("steam_purchase").and_then(Value::as_bool));
        self.received_for_free
            .append_option(r.get("received_for_free").and_then(Value::as_bool));
        self.early_access.append_option(
            r.get("written_during_early_access")
                .and_then(Value::as_bool),
        );
        self.refunded
            .append_option(r.get("refunded").and_then(Value::as_bool));
        self.steam_deck
            .append_option(r.get("primarily_steam_deck").and_then(Value::as_bool));

        self.raw_json.append_value(r.to_string());
        Ok(())
    }

    fn finish(mut self, schema: &Arc<Schema>) -> Result<RecordBatch> {
        let columns: Vec<ArrayRef> = vec![
            Arc::new(self.recommendationid.finish()),
            Arc::new(self.appid.finish()),
            Arc::new(self.author_steamid.finish()),
            Arc::new(self.num_games_owned.finish()),
            Arc::new(self.num_reviews.finish()),
            Arc::new(self.playtime_forever.finish()),
            Arc::new(self.playtime_at_review.finish()),
            Arc::new(self.playtime_last_two_weeks.finish()),
            Arc::new(self.last_played.finish()),
            Arc::new(self.language.finish()),
            Arc::new(self.review.finish()),
            Arc::new(self.created.finish()),
            Arc::new(self.updated.finish()),
            Arc::new(self.voted_up.finish()),
            Arc::new(self.votes_up.finish()),
            Arc::new(self.votes_funny.finish()),
            Arc::new(self.weighted_vote_score.finish()),
            Arc::new(self.comment_count.finish()),
            Arc::new(self.steam_purchase.finish()),
            Arc::new(self.received_for_free.finish()),
            Arc::new(self.early_access.finish()),
            Arc::new(self.refunded.finish()),
            Arc::new(self.steam_deck.finish()),
            Arc::new(self.raw_json.finish()),
        ];
        Ok(RecordBatch::try_new(Arc::clone(schema), columns)?)
    }
}

/// Reads back the text of specific reviews, by id.
///
/// Takes the ids it wants rather than returning the corpus, because a caller that needs a
/// few hundred reviews out of a million should not pay for the other million.
///
/// # Errors
///
/// Fails if a shard cannot be read.
pub fn texts_for<S: std::hash::BuildHasher>(
    snapshot: &Path,
    ids: &std::collections::HashSet<String, S>,
) -> Result<std::collections::HashMap<String, String>> {
    use arrow::array::{Array, StringArray};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let mut shards: Vec<std::path::PathBuf> = std::fs::read_dir(snapshot)?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("shard-") && name.ends_with(".parquet"))
        })
        .collect();
    shards.sort();

    let mut found = std::collections::HashMap::new();
    for shard in shards {
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&shard)?)?
            .with_batch_size(8192)
            .build()?;
        for batch in reader {
            let batch = batch?;
            let column = |name: &'static str| -> Result<&StringArray> {
                batch
                    .column_by_name(name)
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                    .ok_or(Error::MalformedPayload { field: name })
            };
            let review_ids = column("recommendationid")?;
            let bodies = column("review")?;
            for row in 0..batch.num_rows() {
                if bodies.is_null(row) {
                    continue;
                }
                let id = review_ids.value(row);
                if ids.contains(id) {
                    found.insert(id.to_owned(), bodies.value(row).to_owned());
                }
            }
        }
    }
    Ok(found)
}

fn text(v: Option<&Value>) -> Option<&str> {
    v?.as_str()
}

fn number(v: Option<&Value>) -> Option<u32> {
    u32::try_from(v?.as_u64()?).ok()
}

/// `weighted_vote_score` arrives as a JSON string on populated reviews and as a number when
/// it is zero, so neither `as_f64` nor `as_str` alone is enough.
fn decimal(v: Option<&Value>) -> Option<f64> {
    match v? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn weighted_vote_score_accepts_both_shapes() {
        assert_eq!(decimal(Some(&json!("0.5238095"))), Some(0.523_809_5));
        assert_eq!(decimal(Some(&json!(0))), Some(0.0));
        assert_eq!(decimal(Some(&json!(null))), None);
        assert_eq!(decimal(None), None);
    }

    #[test]
    fn oversized_counts_do_not_abort_a_row() {
        assert_eq!(number(Some(&json!(u64::from(u32::MAX) + 1))), None);
        assert_eq!(number(Some(&json!(42))), Some(42));
    }

    #[test]
    fn a_review_without_an_id_is_rejected() {
        let mut builders = RowBuilders::with_capacity(1);
        let err = builders
            .push(1, &json!({"review": "no id here"}))
            .unwrap_err();
        assert!(matches!(
            err,
            Error::MalformedPayload {
                field: "recommendationid"
            }
        ));
    }

    #[test]
    fn missing_fields_become_nulls_rather_than_failing() {
        let mut builders = RowBuilders::with_capacity(1);
        builders
            .push(1_091_500, &json!({"recommendationid": "1", "review": "ok"}))
            .unwrap();
        let batch = builders.finish(&schema()).unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), schema().fields().len());
    }

    #[test]
    fn raw_json_preserves_fields_the_schema_does_not_model() {
        let mut builders = RowBuilders::with_capacity(1);
        let review = json!({"recommendationid": "7", "some_future_field": [1, 2, 3]});
        builders.push(1, &review).unwrap();
        let batch = builders.finish(&schema()).unwrap();
        let raw = batch
            .column_by_name("raw_json")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap()
            .value(0);
        assert!(raw.contains("some_future_field"), "{raw}");
    }
}
