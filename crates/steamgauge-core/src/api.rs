//! HTTP access to Valve's `appreviews` endpoint.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use serde::Deserialize;
use serde_json::Value;
use tokio::sync::Mutex;

use crate::{Error, Result, query::ReviewQuery};

/// Valve documents no rate limit for this endpoint, so the ceiling is unknown and can only
/// be found by exceeding it. The client paces itself and treats any push-back as
/// authoritative rather than probing for the real limit.
const MAX_ATTEMPTS: u32 = 5;
const BACKOFF_BASE: Duration = Duration::from_secs(2);
pub const DEFAULT_PACE: Duration = Duration::from_millis(250);

/// Totals as Valve reports them for the query's filters, present only on the first page.
///
/// `total_reviews` is what a crawl's coverage is measured against. It moves with the
/// request parameters, so it is only comparable to a crawl made with the same ones.
#[derive(Debug, Clone, Deserialize)]
pub struct QuerySummary {
    #[serde(default)]
    pub total_reviews: u64,
    #[serde(default)]
    pub total_positive: u64,
    #[serde(default)]
    pub total_negative: u64,
    #[serde(default)]
    pub review_score_desc: String,
}

/// One page of results.
///
/// Reviews stay as raw JSON rather than a typed struct: the capture layer's job is to lose
/// nothing, and Valve adds fields over time that a fixed struct would silently discard.
#[derive(Debug, Deserialize)]
pub struct Page {
    #[serde(default)]
    pub success: u8,
    #[serde(default)]
    pub query_summary: Option<QuerySummary>,
    #[serde(default)]
    pub reviews: Vec<Value>,
    #[serde(default)]
    pub cursor: Option<String>,
}

/// Paces every request through one shared slot, so raising shard concurrency changes how
/// the work is ordered but never how hard Valve is hit.
#[derive(Debug, Clone)]
pub struct SteamClient {
    http: reqwest::Client,
    pace: Duration,
    next_slot: Arc<Mutex<Instant>>,
}

impl SteamClient {
    /// # Errors
    ///
    /// Fails if the HTTP client cannot be constructed, which in practice means a missing or
    /// unusable TLS backend.
    pub fn new(pace: Duration) -> Result<Self> {
        let http = reqwest::Client::builder()
            .user_agent(concat!(
                "steamgauge/",
                env!("CARGO_PKG_VERSION"),
                " (+https://github.com/Aureliolo/steamgauge)"
            ))
            .timeout(Duration::from_secs(60))
            .build()?;
        Ok(Self {
            http,
            pace,
            next_slot: Arc::new(Mutex::new(Instant::now())),
        })
    }

    /// How many reviews Valve reports for a window, without downloading any of them.
    ///
    /// # Errors
    ///
    /// Propagates transport and throttling failures.
    pub async fn count(&self, app_id: u32, window: Option<(i64, i64)>) -> Result<u64> {
        let mut query = ReviewQuery::new(app_id).per_page(0);
        if let Some((start, end)) = window {
            query = query.window(start, end);
        }
        let page = self.fetch(&query, app_id).await?;
        Ok(page.query_summary.map_or(0, |s| s.total_reviews))
    }

    /// The store's name for an app, so a report can say "Helldivers 2" rather than 553850.
    ///
    /// Best effort by design: a delisted or region-locked app answers with no name, and a
    /// missing name is not a reason to refuse a crawl of reviews that are being served
    /// perfectly well. The caller falls back to the id.
    pub async fn name(&self, app_id: u32) -> Option<String> {
        self.wait_turn().await;
        let url =
            format!("https://store.steampowered.com/api/appdetails?appids={app_id}&filters=basic");
        let body: serde_json::Value = self.http.get(&url).send().await.ok()?.json().await.ok()?;
        let name = body
            .get(app_id.to_string())?
            .get("data")?
            .get("name")?
            .as_str()?
            .trim();
        (!name.is_empty()).then(|| name.to_owned())
    }

    /// Fetches one page, retrying on throttling and transient server errors.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Throttled`] if Valve keeps refusing after five attempts, and
    /// [`Error::NoSuchCorpus`] if it answers `success: 0`.
    pub async fn fetch(&self, query: &ReviewQuery, app_id: u32) -> Result<Page> {
        let url = query.to_url();
        let mut attempt = 0;

        loop {
            attempt += 1;
            self.wait_turn().await;

            let response = self.http.get(&url).send().await?;
            let status = response.status();

            if status.is_success() {
                let page: Page = response.json().await?;
                if page.success != 1 {
                    return Err(Error::NoSuchCorpus { app_id });
                }
                return Ok(page);
            }

            let retryable = status.as_u16() == 429 || status.is_server_error();
            if !retryable || attempt >= MAX_ATTEMPTS {
                return Err(Error::Throttled {
                    attempts: attempt,
                    status: status.as_u16(),
                });
            }

            // Valve's own Retry-After wins over any guess the client could make.
            let wait =
                retry_after(&response).unwrap_or_else(|| BACKOFF_BASE * 2_u32.pow(attempt - 1));
            tokio::time::sleep(wait).await;
        }
    }

    async fn wait_turn(&self) {
        let now = Instant::now();
        let slot_at = {
            let mut slot = self.next_slot.lock().await;
            let at = (*slot).max(now);
            *slot = at + self.pace;
            at
        };
        if slot_at > now {
            tokio::time::sleep(slot_at - now).await;
        }
    }
}

fn retry_after(response: &reqwest::Response) -> Option<Duration> {
    response
        .headers()
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}
