//! Core library for steam-review-census.
//!
//! Nothing here is stable yet. See the repository README for what is being built.

pub mod api;
pub mod capture;
pub mod crawl;
pub mod query;

pub use api::{Page, QuerySummary, SteamClient};
pub use capture::CaptureWriter;
pub use crawl::{CrawlOptions, CrawlReport, Progress, StopReason, crawl};
pub use query::{ReviewQuery, SortOrder};

/// Anything that can go wrong while building a corpus.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("http request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// Valve answers with `success: 0` for unknown app IDs and for apps whose reviews are
    /// not served, which is not an HTTP error and would otherwise look like an empty corpus.
    #[error("steam served no review data for app {app_id}")]
    NoSuchCorpus { app_id: u32 },

    #[error("steam rejected the crawl after {attempts} attempts: {status}")]
    Throttled { attempts: u32, status: u16 },

    #[error("review payload missing {field}")]
    MalformedPayload { field: &'static str },

    #[error(transparent)]
    Parquet(#[from] parquet::errors::ParquetError),

    #[error(transparent)]
    Arrow(#[from] arrow::error::ArrowError),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
