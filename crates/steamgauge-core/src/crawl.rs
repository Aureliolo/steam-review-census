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
    query::{ReviewQuery, SortOrder},
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
}

impl Default for CrawlOptions {
    fn default() -> Self {
        Self {
            out_dir: PathBuf::from("data"),
            concurrency: 4,
            shard_target: DEFAULT_SHARD_TARGET,
            resume: true,
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
    /// The store's name for the app, where the store would give one. Empty otherwise, and
    /// every caller falls back to the id rather than treating it as a failure.
    pub name: String,
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
    /// Windows Steam stopped serving early, which a second walk then completed.
    pub shards_restarted: usize,
    /// Windows still short of Valve's stated count after every walk. These stay unfinished.
    pub shards_short: usize,
}

impl CrawlReport {
    /// Share of Valve's own stated total that was actually retrieved.
    ///
    /// This is the figure that makes the census claim checkable rather than asserted. It is
    /// only meaningful against a total reported for the same request parameters.
    #[must_use]
    pub fn coverage(&self) -> Option<f64> {
        coverage_of(self.unique, self.valve_total)
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "review counts are far below 2^53"
)]
fn coverage_of(unique: u64, valve_total: u64) -> Option<f64> {
    (valve_total > 0).then(|| unique as f64 / valve_total as f64)
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

    let (crawl_id, snapshot, resumed) =
        prepare(client, app_id, options, &state, valve_total).await?;

    let dir = options
        .out_dir
        .join(format!("appid={app_id}"))
        .join(format!("snapshot={snapshot}"));
    std::fs::create_dir_all(&dir)?;

    let pending = state.pending_shards(crawl_id)?;
    let shards_total = pending.len();
    let Tally {
        duplicates,
        shards_restarted,
        shards_short,
    } = run_shards(
        ShardRun {
            client,
            app_id,
            state: &state,
            crawl_id,
            dir: &dir,
            concurrency: options.concurrency,
        },
        pending,
        |shards_done, unique| {
            on_progress(Progress {
                shards_done,
                shards_total,
                unique,
                valve_total,
            });
        },
    )
    .await?;

    let complete = state.all_shards_done(crawl_id)?;
    if complete {
        state.finish_crawl(crawl_id)?;
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
        name: client.name(app_id).await.unwrap_or_default(),
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
        shards_restarted,
        shards_short,
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
        CORPUS_EPOCH,
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

/// What the shard run added up to, beyond what the state database already records.
struct Tally {
    duplicates: u64,
    shards_restarted: usize,
    shards_short: usize,
}

/// Everything a shard walk needs that is the same for every shard.
struct ShardRun<'a> {
    client: &'a SteamClient,
    app_id: u32,
    state: &'a Arc<CrawlState>,
    crawl_id: i64,
    dir: &'a std::path::Path,
    concurrency: usize,
}

/// Walks every outstanding window, up to `concurrency` at a time.
async fn run_shards(
    run: ShardRun<'_>,
    pending: Vec<crate::state::ShardRecord>,
    mut on_progress: impl FnMut(usize, u64),
) -> Result<Tally> {
    let ShardRun {
        client,
        app_id,
        state,
        crawl_id,
        dir,
        concurrency,
    } = run;
    let permits = Arc::new(Semaphore::new(concurrency.max(1)));
    let mut tasks = JoinSet::new();
    for record in pending {
        let client = client.clone();
        let permits = Arc::clone(&permits);
        let state = Arc::clone(state);
        let path = dir.join(format!("shard-{:04}.parquet", record.idx));
        tasks.spawn(async move {
            let _permit = permits.acquire_owned().await;
            state.mark_running(crawl_id, record.idx)?;
            let outcome = crawl_shard(&client, app_id, record.shard, &path).await?;
            // A window still short after every walk is left unfinished on purpose. Marking
            // it done would fold a known undercount into the corpus and report it as
            // complete; left as it is, the crawl reports itself incomplete and the next run
            // walks it again.
            if !outcome.short {
                state.mark_done(
                    crawl_id,
                    record.idx,
                    outcome.rows,
                    outcome.pages,
                    outcome.stop.as_str(),
                    outcome.max_created,
                )?;
            }
            Ok::<_, crate::Error>(outcome)
        });
    }

    let mut tally = Tally {
        duplicates: 0,
        shards_restarted: 0,
        shards_short: 0,
    };
    let mut done = 0;
    while let Some(joined) = tasks.join_next().await {
        // A panicking shard task must not be reported as a completed crawl.
        let outcome = joined.map_err(|e| crate::Error::ShardPanicked {
            detail: e.to_string(),
        })??;
        tally.duplicates += outcome.fetched.saturating_sub(outcome.rows);
        if outcome.walks > 1 {
            tally.shards_restarted += 1;
        }
        if outcome.short {
            tally.shards_short += 1;
        }
        done += 1;
        let (unique, _) = state.completed_totals(crawl_id)?;
        on_progress(done, unique);
    }
    Ok(tally)
}

/// Walks allowed for one window before its shortfall is treated as real rather than a stall.
///
/// Steam intermittently stops serving a window early. The cursor advances normally, pages
/// come back full, and then an empty page arrives long before the window is exhausted, which
/// is indistinguishable from a genuine end. Measured on a 380,000-review corpus, three of
/// fourteen windows stopped between 69% and 75% of their expected count, and re-walking an
/// identical window immediately afterwards returned 18,159 of 18,160.
///
/// Accepting the first walk is what made that corpus 4.5% short while reporting every shard
/// as finished, which is the one failure this project cannot tolerate quietly: it undercounts
/// the corpus and calls it a census.
const MAX_SHARD_WALKS: u32 = 3;

/// How far below Valve's stated count for a window a walk may land before it is walked again.
///
/// Windows that genuinely finish land within a handful of reviews of expectation, the drift
/// being reviews written or deleted between planning and walking; the largest gap seen across
/// eleven honest shards was eight reviews in 39,278. Windows cut short by a stall miss a
/// quarter or more. Nothing observed falls between, so both bounds have wide margins.
const SHORTFALL_RATIO: f64 = 0.95;

/// Shortfalls smaller than this are drift rather than evidence, however small the window.
const SHORTFALL_FLOOR: u64 = 20;

/// Whether a walk ended so far below expectation that Steam is more likely to have stalled
/// than the window to have run out.
#[expect(
    clippy::cast_precision_loss,
    reason = "review counts are far below 2^53"
)]
fn fell_short(rows: u64, expected: u64) -> bool {
    if expected == 0 {
        return false;
    }
    expected.saturating_sub(rows) > SHORTFALL_FLOOR
        && (rows as f64) < expected as f64 * SHORTFALL_RATIO
}

#[derive(Debug)]
struct ShardOutcome {
    rows: u64,
    fetched: u64,
    pages: u32,
    stop: StopReason,
    max_created: Option<i64>,
    /// Walks taken. More than one means Steam stopped early at least once.
    walks: u32,
    /// Still short after every walk, so the window is not known to be complete.
    short: bool,
}

/// Walks a window, repeating it while it comes back short of what Valve says it holds.
async fn crawl_shard(
    client: &SteamClient,
    app_id: u32,
    shard: Shard,
    path: &std::path::Path,
) -> Result<ShardOutcome> {
    let mut last = walk_shard(client, app_id, shard, path).await?;
    for walk in 2..=MAX_SHARD_WALKS {
        if !fell_short(last.rows, shard.expected) {
            return Ok(ShardOutcome {
                walks: walk - 1,
                short: false,
                ..last
            });
        }
        last = walk_shard(client, app_id, shard, path).await?;
        last.walks = walk;
    }
    let short = fell_short(last.rows, shard.expected);
    Ok(ShardOutcome { short, ..last })
}

async fn walk_shard(
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
        walks: 1,
        short: false,
    })
}

/// A snapshot that does not record the parameters it was gathered under cannot be compared
/// to any other snapshot, so the metadata travels with the Parquet files.
fn write_sidecar(report: &CrawlReport, snapshot: i64) -> Result<()> {
    let meta = json!({
        "app_id": report.app_id,
        "name": report.name,
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
        "shards_restarted": report.shards_restarted,
        "shards_short": report.shards_short,
        "elapsed_secs": report.elapsed.as_secs_f64(),
    });
    std::fs::write(
        report.dir.join("crawl.json"),
        serde_json::to_vec_pretty(&meta)?,
    )?;
    Ok(())
}

/// How far a sweep has got.
#[derive(Debug, Clone, Copy)]
pub struct SweepProgress {
    pub pages: u32,
    pub rows: u64,
}

/// What a sweep brought in.
#[derive(Debug, Clone)]
pub struct SweepReport {
    pub app_id: u32,
    pub dir: PathBuf,
    /// When the sweep started, which is the file it wrote and the next sweep's watermark.
    pub started: i64,
    /// Everything written or edited since this moment was fetched.
    pub watermark: i64,
    pub pages: u32,
    /// Rows written: reviews new since the watermark plus reviews edited since it.
    pub rows: u64,
    /// Of those, reviews the capture had never held.
    pub new: u64,
    /// Of those, reviews the capture already held in an older form.
    pub edited: u64,
    /// The capture's unique reviews after the sweep, against Valve's total now.
    pub unique: u64,
    pub valve_total: u64,
    pub stop: StopReason,
    pub elapsed: Duration,
}

impl SweepReport {
    /// Share of Valve's total the capture now holds.
    #[must_use]
    pub fn coverage(&self) -> Option<f64> {
        coverage_of(self.unique, self.valve_total)
    }
}

/// How far past the watermark a sweep reads before trusting that it has seen everything.
///
/// Valve orders by last edit but not exactly: a review can arrive a few places later than
/// its time says. A day of slack costs a few pages and misses nothing a cursor can reach.
const SWEEP_SLACK: i64 = 24 * 60 * 60;

/// Brings a capture up to date: every review written or edited since it was last brought
/// up to date, or since it was crawled, is fetched again and written beside what is there.
///
/// One walk in last-edit order serves for both. A new review's last edit is its creation,
/// so the same walk that finds edits finds arrivals, and it stops at the watermark rather
/// than at the end of the corpus, which is what makes a sweep a few pages rather than a
/// crawl. The rows land in a sweep file of their own and [`crate::capture::Newest`] records
/// which copy of each review now counts, so nothing already captured is touched.
///
/// # Errors
///
/// Fails if the capture is missing, the network does, or the files cannot be written.
pub async fn sweep(
    client: &SteamClient,
    app_id: u32,
    out_dir: &std::path::Path,
    mut on_progress: impl FnMut(SweepProgress),
) -> Result<SweepReport> {
    let begun = Instant::now();
    let dir = crate::embed::latest_snapshot(out_dir, app_id)?;
    let mut facts: Value = serde_json::from_slice(&std::fs::read(dir.join("crawl.json"))?)?;
    let watermark = facts
        .get("swept_unix")
        .and_then(Value::as_i64)
        .or_else(|| facts.get("snapshot_unix").and_then(Value::as_i64))
        .ok_or(crate::Error::MalformedPayload {
            field: "snapshot_unix",
        })?;
    let started = now_unix();

    let mut newest = crate::capture::Newest::load(&dir)?;
    let path = dir.join(crate::capture::sweep_file(started));
    let walked = walk_since(
        client,
        app_id,
        watermark,
        started,
        &path,
        &mut newest,
        &mut on_progress,
    )
    .await?;
    if walked.rows == 0 {
        std::fs::remove_file(&path)?;
    } else {
        newest.save(&dir)?;
    }

    // Valve's total moves with the corpus, so coverage is re-read against today's figure
    // rather than the one the crawl saw.
    let summary = client
        .fetch(&ReviewQuery::new(app_id).per_page(0), app_id)
        .await?
        .query_summary;
    let unique = facts
        .get("rows_unique")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        .saturating_add(walked.new);
    let valve_total = summary.as_ref().map_or(0, |s| s.total_reviews);
    let sweeps = facts.get("sweeps").and_then(Value::as_u64).unwrap_or(0) + 1;
    let swept_rows = facts
        .get("rows_swept")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        .saturating_add(walked.rows);
    if let Some(object) = facts.as_object_mut() {
        object.insert("swept_unix".to_owned(), json!(started));
        object.insert("sweeps".to_owned(), json!(sweeps));
        object.insert("rows_swept".to_owned(), json!(swept_rows));
        object.insert("rows_unique".to_owned(), json!(unique));
        if let Some(summary) = &summary {
            object.insert(
                "valve_total_reviews".to_owned(),
                json!(summary.total_reviews),
            );
            object.insert(
                "valve_total_positive".to_owned(),
                json!(summary.total_positive),
            );
            object.insert(
                "valve_total_negative".to_owned(),
                json!(summary.total_negative),
            );
            object.insert(
                "review_score_desc".to_owned(),
                json!(summary.review_score_desc),
            );
            object.insert(
                "coverage".to_owned(),
                json!(coverage_of(unique, valve_total)),
            );
        }
    }
    std::fs::write(dir.join("crawl.json"), serde_json::to_vec_pretty(&facts)?)?;

    Ok(SweepReport {
        app_id,
        dir,
        started,
        watermark,
        pages: walked.pages,
        rows: walked.rows,
        new: walked.new,
        edited: walked.edited,
        unique,
        valve_total,
        stop: walked.stop,
        elapsed: begun.elapsed(),
    })
}

/// What one walk in last-edit order brought in.
struct Walked {
    pages: u32,
    rows: u64,
    new: u64,
    edited: u64,
    stop: StopReason,
}

/// Walks the corpus newest edit first, writing every review touched since `watermark` to
/// `path` and noting each in `newest`, until it is safely past the watermark.
async fn walk_since(
    client: &SteamClient,
    app_id: u32,
    watermark: i64,
    started: i64,
    path: &std::path::Path,
    newest: &mut crate::capture::Newest,
    on_progress: &mut impl FnMut(SweepProgress),
) -> Result<Walked> {
    let mut writer = CaptureWriter::create(path, app_id)?;
    let mut seen: HashSet<String> = HashSet::new();
    let mut cursor = "*".to_owned();
    let mut pages: u32 = 0;
    let (mut new, mut edited) = (0_u64, 0_u64);

    let stop = loop {
        let query = ReviewQuery::new(app_id)
            .order(SortOrder::Updated)
            .cursor(cursor.clone());
        let page = client.fetch(&query, app_id).await?;
        pages = pages.saturating_add(1);
        if page.reviews.is_empty() {
            break StopReason::Exhausted;
        }

        let mut newest_on_page = i64::MIN;
        let mut fresh: Vec<&Value> = Vec::new();
        for review in &page.reviews {
            let Some(id) = review.get("recommendationid").and_then(Value::as_str) else {
                continue;
            };
            let stamp = |field: &str| review.get(field).and_then(Value::as_i64).unwrap_or(0);
            let updated = stamp("timestamp_updated");
            newest_on_page = newest_on_page.max(updated);
            if updated < watermark || !seen.insert(id.to_owned()) {
                continue;
            }
            // Created since the watermark and never swept before is a review the capture
            // has not seen; anything else is a copy of one it has.
            if stamp("timestamp_created") >= watermark && !newest.copies.contains_key(id) {
                new += 1;
            } else {
                edited += 1;
            }
            newest.record(id, updated, started);
            fresh.push(review);
        }
        writer.write(&fresh)?;
        on_progress(SweepProgress {
            pages,
            rows: writer.rows(),
        });

        if newest_on_page < watermark - SWEEP_SLACK {
            break StopReason::Exhausted;
        }
        let Some(next) = page.cursor else {
            break StopReason::NoCursor;
        };
        if next == cursor {
            break StopReason::CursorRepeated;
        }
        cursor = next;
    };

    Ok(Walked {
        pages,
        rows: writer.close()?,
        new,
        edited,
        stop,
    })
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

    fn report(unique: u64, total: u64) -> CrawlReport {
        CrawlReport {
            app_id: 1,
            shards: 1,
            pages: 1,
            unique,
            duplicates_this_run: 0,
            valve_total: total,
            valve_positive: 0,
            valve_negative: 0,
            name: String::new(),
            review_score_desc: String::new(),
            elapsed: Duration::from_secs(1),
            dir: PathBuf::new(),
            complete: true,
            resumed: false,
            shards_restarted: 0,
            shards_short: 0,
        }
    }

    #[test]
    fn a_window_that_stalls_a_quarter_of_the_way_short_is_not_accepted() {
        // The shard that exposed this returned 12,600 of an expected 18,184 and was recorded
        // as finished. Walking the identical window again returned all of them.
        assert!(fell_short(12_600, 18_184));
        assert!(fell_short(18_297, 24_340));
        assert!(fell_short(12_699, 17_991));
    }

    #[test]
    fn ordinary_drift_between_planning_and_walking_is_not_a_stall() {
        // Reviews are written and deleted while a crawl runs, so expectation is never exact.
        // The widest honest gap observed was eight reviews in 39,278.
        assert!(!fell_short(39_276, 39_278));
        assert!(!fell_short(29_048, 29_050));
        assert!(!fell_short(2_909, 2_909));
        assert!(!fell_short(35_856, 35_860));
    }

    #[test]
    fn a_small_window_is_judged_by_reviews_missed_rather_than_by_share() {
        // Losing three of twenty reviews is 15% and means nothing; the floor stops a tiny
        // window from being walked three times over noise.
        assert!(!fell_short(17, 20));
        assert!(fell_short(60, 200), "a real shortfall must still be caught");
    }

    #[test]
    fn a_window_valve_reports_nothing_for_can_never_be_short() {
        assert!(!fell_short(0, 0));
    }

    #[test]
    fn coverage_is_the_share_of_valves_own_total() {
        assert_eq!(report(2909, 2909).coverage(), Some(1.0));
        let partial = report(21, 2909);
        assert!((partial.coverage().unwrap() - 0.007_218).abs() < 1e-6);
    }

    #[test]
    fn coverage_is_unknown_rather_than_perfect_when_valve_reports_nothing() {
        assert_eq!(report(100, 0).coverage(), None);
    }

    #[test]
    fn stop_reasons_round_trip_to_the_strings_stored_in_state() {
        assert_eq!(StopReason::Exhausted.as_str(), "Exhausted");
        assert_eq!(StopReason::CursorRepeated.as_str(), "CursorRepeated");
        assert_eq!(StopReason::NoCursor.as_str(), "NoCursor");
    }
}
