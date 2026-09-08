//! Walking a corpus to exhaustion and recording how much of it was reached.

use std::{
    collections::HashSet,
    path::PathBuf,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde_json::{Value, json};

use crate::{CaptureWriter, Result, api::SteamClient, query::ReviewQuery};

#[derive(Debug, Clone)]
pub struct CrawlOptions {
    pub out_dir: PathBuf,
    /// Valve publishes no rate limit for this endpoint, so the crawler paces itself rather
    /// than discovering the ceiling by hitting it.
    pub page_interval: Duration,
}

impl Default for CrawlOptions {
    fn default() -> Self {
        Self {
            out_dir: PathBuf::from("data"),
            page_interval: Duration::from_millis(250),
        }
    }
}

/// Why the walk ended. Anything other than [`StopReason::Exhausted`] means the corpus was
/// not fully served and the coverage figure should be read as a floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Valve returned an empty page, which is how a complete walk ends.
    Exhausted,
    /// The cursor stopped advancing. Expected on helpfulness-ranked ordering, and a bug
    /// anywhere else.
    CursorRepeated,
    NoCursor,
}

#[derive(Debug, Clone, Copy)]
pub struct Progress {
    pub pages: u32,
    pub unique: u64,
    pub valve_total: u64,
}

#[derive(Debug, Clone)]
pub struct CrawlReport {
    pub app_id: u32,
    pub pages: u32,
    pub fetched: u64,
    pub unique: u64,
    pub valve_total: u64,
    pub valve_positive: u64,
    pub valve_negative: u64,
    pub review_score_desc: String,
    pub elapsed: Duration,
    pub path: PathBuf,
    pub stopped_because: StopReason,
}

impl CrawlReport {
    /// Share of Valve's own stated total that was actually retrieved.
    ///
    /// This is the figure that makes the census claim checkable rather than asserted. It is
    /// only meaningful against a total reported for the same request parameters.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "review counts are far below 2^53"
    )]
    pub fn coverage(&self) -> Option<f64> {
        (self.valve_total > 0).then(|| self.unique as f64 / self.valve_total as f64)
    }

    #[must_use]
    pub fn duplicates(&self) -> u64 {
        self.fetched.saturating_sub(self.unique)
    }
}

/// Downloads every review Valve will serve for `app_id` into an immutable Parquet capture.
///
/// # Errors
///
/// Propagates transport, throttling and Parquet failures. A partial crawl leaves the
/// Parquet file behind rather than deleting it, because a partial corpus is still worth
/// more than a re-crawl.
pub async fn crawl(
    client: &SteamClient,
    app_id: u32,
    options: &CrawlOptions,
    mut on_progress: impl FnMut(Progress),
) -> Result<CrawlReport> {
    let started = Instant::now();
    let snapshot = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let dir = options.out_dir.join(format!("appid={app_id}"));
    let path = dir.join(format!("reviews-{snapshot}.parquet"));
    let mut writer = CaptureWriter::create(&path, app_id)?;

    let mut cursor = "*".to_owned();
    let mut seen: HashSet<String> = HashSet::new();
    let mut pages: u32 = 0;
    let mut fetched: u64 = 0;
    let mut summary = None;

    let stopped_because = loop {
        if pages > 0 {
            tokio::time::sleep(options.page_interval).await;
        }

        let page = client
            .fetch(&ReviewQuery::new(app_id).cursor(cursor.clone()), app_id)
            .await?;
        pages = pages.saturating_add(1);

        if summary.is_none() {
            summary = page.query_summary;
        }
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
        writer.write(&fresh)?;

        let valve_total = summary.as_ref().map_or(0, |s| s.total_reviews);
        on_progress(Progress {
            pages,
            unique: count(seen.len()),
            valve_total,
        });

        let Some(next) = page.cursor else {
            break StopReason::NoCursor;
        };
        if next == cursor {
            break StopReason::CursorRepeated;
        }
        cursor = next;
    };

    let unique = writer.close()?;
    let summary = summary.unwrap_or(crate::QuerySummary {
        total_reviews: 0,
        total_positive: 0,
        total_negative: 0,
        review_score_desc: String::new(),
    });

    let report = CrawlReport {
        app_id,
        pages,
        fetched,
        unique,
        valve_total: summary.total_reviews,
        valve_positive: summary.total_positive,
        valve_negative: summary.total_negative,
        review_score_desc: summary.review_score_desc,
        elapsed: started.elapsed(),
        path,
        stopped_because,
    };
    write_sidecar(&report, snapshot)?;
    Ok(report)
}

/// A snapshot that does not record the parameters it was gathered under cannot be compared
/// to any other snapshot, so the metadata travels with the Parquet file.
fn write_sidecar(report: &CrawlReport, snapshot: u64) -> Result<()> {
    let meta = json!({
        "app_id": report.app_id,
        "snapshot_unix": snapshot,
        "tool_version": env!("CARGO_PKG_VERSION"),
        "request_url_template": ReviewQuery::new(report.app_id).to_url(),
        "pages": report.pages,
        "rows_fetched": report.fetched,
        "rows_unique": report.unique,
        "duplicates": report.duplicates(),
        "valve_total_reviews": report.valve_total,
        "valve_total_positive": report.valve_positive,
        "valve_total_negative": report.valve_negative,
        "review_score_desc": report.review_score_desc,
        "coverage": report.coverage(),
        "stopped_because": format!("{:?}", report.stopped_because),
        "elapsed_secs": report.elapsed.as_secs_f64(),
    });
    std::fs::write(
        report.path.with_extension("json"),
        serde_json::to_vec_pretty(&meta)?,
    )?;
    Ok(())
}

fn count(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(unique: u64, total: u64, fetched: u64) -> CrawlReport {
        CrawlReport {
            app_id: 1,
            pages: 1,
            fetched,
            unique,
            valve_total: total,
            valve_positive: 0,
            valve_negative: 0,
            review_score_desc: String::new(),
            elapsed: Duration::from_secs(1),
            path: PathBuf::new(),
            stopped_because: StopReason::Exhausted,
        }
    }

    #[test]
    fn coverage_is_the_share_of_valves_own_total() {
        let full = report(2909, 2909, 2909);
        assert_eq!(full.coverage(), Some(1.0));
        assert_eq!(full.duplicates(), 0);

        let partial = report(21, 2909, 75);
        assert!((partial.coverage().unwrap() - 0.007_218).abs() < 1e-6);
        assert_eq!(partial.duplicates(), 54);
    }

    #[test]
    fn coverage_is_unknown_rather_than_perfect_when_valve_reports_nothing() {
        assert_eq!(report(100, 0, 100).coverage(), None);
    }
}
