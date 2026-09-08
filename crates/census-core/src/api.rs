//! HTTP access to Valve's `appreviews` endpoint.

use std::time::Duration;

use serde::Deserialize;
use serde_json::Value;

use crate::{Error, Result, query::ReviewQuery};

/// Valve documents no rate limit for this endpoint, so the ceiling is unknown and can only
/// be found by exceeding it. The crawler stays well under measured throughput and treats
/// any push-back as authoritative rather than probing for the real limit.
const MAX_ATTEMPTS: u32 = 5;
const BACKOFF_BASE: Duration = Duration::from_secs(2);

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

#[derive(Debug, Clone)]
pub struct SteamClient {
    http: reqwest::Client,
}

impl SteamClient {
    /// # Errors
    ///
    /// Fails if the HTTP client cannot be constructed, which in practice means a missing or
    /// unusable TLS backend.
    pub fn new() -> Result<Self> {
        let http = reqwest::Client::builder()
            .user_agent(concat!(
                "steam-review-census/",
                env!("CARGO_PKG_VERSION"),
                " (+https://github.com/Aureliolo/steam-review-census)"
            ))
            .timeout(Duration::from_secs(60))
            .build()?;
        Ok(Self { http })
    }

    /// Fetches one page, retrying on throttling and transient server errors.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Throttled`] if Valve keeps refusing after [`MAX_ATTEMPTS`], and
    /// [`Error::NoSuchCorpus`] if it answers `success: 0`.
    pub async fn fetch(&self, query: &ReviewQuery, app_id: u32) -> Result<Page> {
        let url = query.to_url();
        let mut attempt = 0;

        loop {
            attempt += 1;
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
