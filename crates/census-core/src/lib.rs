//! Core library for steam-review-census.
//!
//! Nothing here is stable yet. See the repository README for what is being built.

pub mod anchors;
pub mod api;
mod bounded;
pub mod capture;
pub mod claims;
pub mod classify;
pub mod crawl;
pub mod embed;
pub mod evaluate;
pub mod html;
pub mod model;
pub mod query;
pub mod report;
pub mod sample;
pub mod shard;
pub mod state;
pub mod taxonomy;
pub mod time;

pub use anchors::{Anchors, FitOutcome, FitParams};
pub use api::{DEFAULT_PACE, Page, QuerySummary, SteamClient};
pub use capture::CaptureWriter;
pub use classify::{CategoryStats, ClassifyOptions, ClassifyReport, classify_corpus};
pub use crawl::{CrawlOptions, CrawlReport, Progress, StopReason, crawl};
pub use embed::{DEFAULT_BATCH_SIZE, EmbedReport, Embedder, embed_corpus};
pub use evaluate::{AgreementReport, ReferenceSet, Slice, compare};
pub use model::{Encoder, Precision};
pub use query::{ReviewQuery, SortOrder};
pub use sample::{SampleOptions, SampleReport, SampledReview};
pub use shard::{DEFAULT_SHARD_TARGET, Shard};
pub use state::CrawlState;
pub use taxonomy::{CORE_SPINE, CORE_SPINE_VERSION, Category};

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

    /// A shard task died rather than returning an error. Reporting a completed crawl here
    /// would claim coverage the corpus does not have.
    #[error("a shard task failed: {detail}")]
    ShardPanicked { detail: String },

    /// A model file that does not match its pinned hash would change every number the tool
    /// reports without anything appearing to go wrong, so it is refused rather than used.
    #[error("model file {file} failed verification: expected {expected}, got {actual}")]
    ModelChecksum {
        file: &'static str,
        expected: &'static str,
        actual: String,
    },

    #[error("no capture found at {path}; run `census crawl` first")]
    NoCapture { path: std::path::PathBuf },

    #[error("no embeddings found at {path}; run `census embed` first")]
    NoEmbeddings { path: std::path::PathBuf },

    #[error("no classifications found at {path}; run `census classify` first")]
    NoClassifications { path: std::path::PathBuf },

    #[error("no reference set at {path}; run `census sample` then `census ingest` first")]
    NoReferenceSet { path: std::path::PathBuf },

    #[error("no fitted anchors at {path}; run `census fit` first")]
    NoAnchors { path: std::path::PathBuf },

    /// Written by a build that recorded less about a run than this one reads back. The
    /// assignments inside predate whatever is missing, so they are refused rather than
    /// partially interpreted.
    #[error(
        "{path} was written by an older build and is missing {field}; re-run `census classify`"
    )]
    StaleClassifications {
        path: std::path::PathBuf,
        field: &'static str,
    },

    /// Anchors fitted against a different taxonomy or embedding model would shift every
    /// category boundary without anything appearing to go wrong, so they are refused.
    #[error("anchors were fitted for a different {field}: expected {expected}, got {actual}")]
    StaleAnchors {
        field: &'static str,
        expected: String,
        actual: String,
    },

    /// Fitting needs the labelled reviews to be present in the corpus that was embedded.
    #[error("no reference label matched the embedded corpus; re-crawl app {app_id} first")]
    NoTrainingExamples { app_id: u32 },

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error(transparent)]
    Ort(#[from] ort::Error),

    #[error(transparent)]
    Shape(#[from] ndarray::ShapeError),

    #[error(transparent)]
    Parquet(#[from] parquet::errors::ParquetError),

    #[error(transparent)]
    Arrow(#[from] arrow::error::ArrowError),

    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
