//! Orchestrating a sharded, resumable walk over a corpus.

use std::{
    collections::HashSet,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde_json::{Value, json};
use tokio::{sync::Semaphore, task::JoinSet};

use crate::{
    CaptureWriter, Result,
    api::SteamClient,
    query::ReviewQuery,
    shard::{self, CORPUS_EPOCH, DEFAULT_SHARD_TARGET, Shard},
    state::CrawlState,
};

#[derive(Debug, Clone)]
pub struct CrawlOptions {
    pub out_dir: PathBuf,
    /// Shards in flight at once. Request pacing is enforced globally by the client, so this
    /// changes how work is ordered, not how hard Valve is hit.
    pub concurrency: usize,
    pub shard_target: u64,
    pub resume: bool,
    /// Fetch only reviews newer than the last completed crawl's watermark.
    pub top_up: bool,
}

impl Default for CrawlOptions {
    fn default() -> Self {
        Self {
            out_dir: PathBuf::from("data"),
            concurrency: 4,
            shard_target: DEFAULT_SHARD_TARGET,
            resume: true,
            top_up: false,
        }
    }
}

/// Why a shard's walk ended. Anything other than [`StopReason::Exhausted`] means the window
/// was not fully served and coverage should be read as a floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    Exhausted,
    /// The cursor stopped advancing. Expected on helpfulness-ranked ordering, and a bug
    /// anywhere else.
    CursorRepeated,
    NoCursor,
}

impl StopReason {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exhausted => "Exhausted",
            Self::CursorRepeated => "CursorRepeated",
            Self::NoCursor => "NoCursor",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Progress {
    pub shards_done: usize,
    pub shards_total: usize,
    pub unique: u64,
    pub valve_total: u64,
}

#[derive(Debug, Clone)]
pub struct CrawlReport {
    pub app_id: u32,
    pub shards: usize,
    pub pages: u32,
    pub unique: u64,
    pub duplicates_this_run: u64,
    pub valve_total: u64,
    pub valve_positive: u64,
    pub valve_negative: u64,
    pub review_score_desc: String,
    pub elapsed: Duration,
    pub dir: PathBuf,
    pub complete: bool,
    pub resumed: bool,
    pub top_up_from: Option<i64>,
}

impl CrawlReport {
    /// Share of Valve's own stated total that was actually retrieved.
    ///
    /// This is the figure that makes the census claim checkable rather than asserted. It is
    /// only meaningful against a total reported for the same request parameters, and it is
    /// meaningless for a top-up, which deliberately fetches only part of the corpus.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "review counts are far below 2^53"
    )]
    pub fn coverage(&self) -> Option<f64> {
        if self.top_up_from.is_some() || self.valve_total == 0 {
            return None;
        }
        Some(self.unique as f64 / self.valve_total as f64)
    }
}

/// Downloads every review Valve will serve for `app_id` into an immutable Parquet capture.
///
/// # Errors
///
/// Propagates transport, throttling, database and Parquet failures. An interrupted crawl
/// leaves its completed shards on disk and resumes from them on the next run.
pub async fn crawl(
    client: &SteamClient,
    app_id: u32,
    options: &CrawlOptions,
    mut on_progress: impl FnMut(Progress),
) -> Result<CrawlReport> {
    let started = Instant::now();
    let state = Arc::new(CrawlState::open(&options.out_dir.join("state.sqlite"))?);

    let summary = client
        .fetch(&ReviewQuery::new(app_id).per_page(0), app_id)
        .await?
        .query_summary;
    let valve_total = summary.as_ref().map_or(0, |s| s.total_reviews);

    let top_up_from = if options.top_up {
        state.watermark(app_id)?
    } else {
        None
    };
    let range_start = top_up_from.map_or(CORPUS_EPOCH, |ts| ts + 1);

    let (crawl_id, snapshot, resumed) =
        prepare(client, app_id, options, &state, range_start, valve_total).await?;

    let dir = options
        .out_dir
        .join(format!("appid={app_id}"))
        .join(format!("snapshot={snapshot}"));
    std::fs::create_dir_all(&dir)?;

    let pending = state.pending_shards(crawl_id)?;
    let shards_total = pending.len();
    let mut duplicates = 0;
    let mut done = 0;

    let permits = Arc::new(Semaphore::new(options.concurrency.max(1)));
    let mut tasks = JoinSet::new();
    for record in pending {
        let client = client.clone();
        let permits = Arc::clone(&permits);
        let state = Arc::clone(&state);
        let path = dir.join(format!("shard-{:04}.parquet", record.idx));
        tasks.spawn(async move {
            let _permit = permits.acquire_owned().await;
            state.mark_running(crawl_id, record.idx)?;
            let outcome = crawl_shard(&client, app_id, record.shard, &path).await?;
            state.mark_done(
                crawl_id,
                record.idx,
                outcome.rows,
                outcome.pages,
                outcome.stop.as_str(),
                outcome.max_created,
            )?;
            Ok::<_, crate::Error>(outcome)
        });
    }

    while let Some(joined) = tasks.join_next().await {
        // A panicking shard task must not be reported as a completed crawl.
        let outcome = joined.map_err(|e| crate::Error::ShardPanicked {
            detail: e.to_string(),
        })??;
        duplicates += outcome.fetched.saturating_sub(outcome.rows);
        done += 1;
        let (unique, _) = state.completed_totals(crawl_id)?;
        on_progress(Progress {
            shards_done: done,
            shards_total,
            unique,
            valve_total,
        });
    }

    let complete = state.all_shards_done(crawl_id)?;
    if complete {
        state.finish_crawl(crawl_id, app_id)?;
    }
    let (unique, pages) = state.completed_totals(crawl_id)?;
    let summary = summary.unwrap_or(crate::QuerySummary {
        total_reviews: 0,
        total_positive: 0,
        total_negative: 0,
        review_score_desc: String::new(),
    });

    let report = CrawlReport {
        app_id,
        shards: shards_total,
        pages,
        unique,
        duplicates_this_run: duplicates,
        valve_total: summary.total_reviews,
        valve_positive: summary.total_positive,
        valve_negative: summary.total_negative,
        review_score_desc: summary.review_score_desc,
        elapsed: started.elapsed(),
        dir,
        complete,
        resumed,
        top_up_from,
    };
    write_sidecar(&report, snapshot)?;
    Ok(report)
}

/// Continues an unfinished crawl where one exists, otherwise plans a new one.
///
/// Resuming reuses the original snapshot timestamp so every shard of one corpus lands in
/// the same directory, however many runs it took to finish.
async fn prepare(
    client: &SteamClient,
    app_id: u32,
    options: &CrawlOptions,
    state: &CrawlState,
    range_start: i64,
    valve_total: u64,
) -> Result<(i64, i64, bool)> {
    if options.resume
        && let Some(found) = state.resumable(app_id)?
    {
        return Ok((found.id, found.snapshot_unix, true));
    }
    let snapshot = now_unix();
    let shards = shard::plan(
        client,
        app_id,
        range_start,
        now_unix(),
        options.shard_target,
    )
    .await?;
    let crawl_id = state.begin_crawl(
        app_id,
        snapshot,
        &ReviewQuery::new(app_id).to_url(),
        valve_total,
    )?;
    state.record_shards(crawl_id, &shards)?;
    Ok((crawl_id, snapshot, false))
}

#[derive(Debug)]
struct ShardOutcome {
    rows: u64,
    fetched: u64,
    pages: u32,
    stop: StopReason,
    max_created: Option<i64>,
}

async fn crawl_shard(
    client: &SteamClient,
    app_id: u32,
    shard: Shard,
    path: &std::path::Path,
) -> Result<ShardOutcome> {
    let mut writer = CaptureWriter::create(path, app_id)?;
    let mut cursor = "*".to_owned();
    let mut seen: HashSet<String> = HashSet::new();
    let mut pages: u32 = 0;
    let mut fetched: u64 = 0;
    let mut max_created: Option<i64> = None;

    let stop = loop {
        let query = ReviewQuery::new(app_id)
            .cursor(cursor.clone())
            .window(shard.start_date, shard.end_date);
        let page = client.fetch(&query, app_id).await?;
        pages = pages.saturating_add(1);

        if page.reviews.is_empty() {
            break StopReason::Exhausted;
        }
        fetched = fetched.saturating_add(count(page.reviews.len()));

        let fresh: Vec<&Value> = page
            .reviews
            .iter()
            .filter(|review| {
                review
                    .get("recommendationid")
                    .and_then(Value::as_str)
                    .is_some_and(|id| seen.insert(id.to_owned()))
            })
            .collect();
        for review in &fresh {
            if let Some(ts) = review.get("timestamp_created").and_then(Value::as_i64) {
                max_created = Some(max_created.map_or(ts, |current: i64| current.max(ts)));
            }
        }
        writer.write(&fresh)?;

        let Some(next) = page.cursor else {
            break StopReason::NoCursor;
        };
        if next == cursor {
            break StopReason::CursorRepeated;
        }
        cursor = next;
    };

    let rows = writer.close()?;
    Ok(ShardOutcome {
        rows,
        fetched,
        pages,
        stop,
        max_created,
    })
}

/// A snapshot that does not record the parameters it was gathered under cannot be compared
/// to any other snapshot, so the metadata travels with the Parquet files.
fn write_sidecar(report: &CrawlReport, snapshot: i64) -> Result<()> {
    let meta = json!({
        "app_id": report.app_id,
        "snapshot_unix": snapshot,
        "tool_version": env!("CARGO_PKG_VERSION"),
        "request_url_template": ReviewQuery::new(report.app_id).to_url(),
        "shards": report.shards,
        "pages": report.pages,
        "rows_unique": report.unique,
        "duplicates_this_run": report.duplicates_this_run,
        "valve_total_reviews": report.valve_total,
        "valve_total_positive": report.valve_positive,
        "valve_total_negative": report.valve_negative,
        "review_score_desc": report.review_score_desc,
        "coverage": report.coverage(),
        "complete": report.complete,
        "resumed": report.resumed,
        "top_up_from": report.top_up_from,
        "elapsed_secs": report.elapsed.as_secs_f64(),
    });
    std::fs::write(
        report.dir.join("crawl.json"),
        serde_json::to_vec_pretty(&meta)?,
    )?;
    Ok(())
}

fn count(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

fn now_unix() -> i64 {
    i64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    )
    .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(unique: u64, total: u64, top_up: Option<i64>) -> CrawlReport {
        CrawlReport {
            app_id: 1,
            shards: 1,
            pages: 1,
            unique,
            duplicates_this_run: 0,
            valve_total: total,
            valve_positive: 0,
            valve_negative: 0,
            review_score_desc: String::new(),
            elapsed: Duration::from_secs(1),
            dir: PathBuf::new(),
            complete: true,
            resumed: false,
            top_up_from: top_up,
        }
    }

    #[test]
    fn coverage_is_the_share_of_valves_own_total() {
        assert_eq!(report(2909, 2909, None).coverage(), Some(1.0));
        let partial = report(21, 2909, None);
        assert!((partial.coverage().unwrap() - 0.007_218).abs() < 1e-6);
    }

    #[test]
    fn a_top_up_reports_no_coverage_because_it_is_not_a_census() {
        assert_eq!(report(12, 2909, Some(1_700_000_000)).coverage(), None);
    }

    #[test]
    fn coverage_is_unknown_rather_than_perfect_when_valve_reports_nothing() {
        assert_eq!(report(100, 0, None).coverage(), None);
    }

    #[test]
    fn stop_reasons_round_trip_to_the_strings_stored_in_state() {
        assert_eq!(StopReason::Exhausted.as_str(), "Exhausted");
        assert_eq!(StopReason::CursorRepeated.as_str(), "CursorRepeated");
        assert_eq!(StopReason::NoCursor.as_str(), "NoCursor");
    }
}
