//! Fetching, verifying and loading the local embedding model.
//!
//! The model is downloaded on first use rather than bundled, so the installer stays small,
//! and every file is checked against a pinned SHA-256 before it is loaded. A model that
//! silently differs from the pinned one would change every number the tool reports without
//! anything appearing to go wrong.

use std::{
    io::Read,
    path::{Path, PathBuf},
};

use ort::session::Session;
use sha2::{Digest, Sha256};

use crate::{Error, Result};

/// Multilingual by necessity: 57% of a typical Steam corpus is not English, so an
/// English-only model would silently discard most of it.
pub const MODEL_ID: &str = "intfloat/multilingual-e5-small";
pub const EMBEDDING_DIM: usize = 384;

/// The model's own position limit. Reviews longer than this are truncated, which affects
/// roughly the top 1% by length.
pub const MAX_TOKENS: usize = 512;

/// e5 models are trained with an instruction prefix and produce measurably worse vectors
/// without one. The model card specifies `query:` for symmetric similarity, which is what
/// clustering and classification need.
pub const E5_PREFIX: &str = "query: ";

#[derive(Debug, Clone, Copy)]
struct Asset {
    remote: &'static str,
    local: &'static str,
    sha256: &'static str,
}

/// Which build of the graph to run.
///
/// Measured on a 2,875-text corpus rather than assumed, because the obvious assumption was
/// wrong. Quantising was expected to trade a little accuracy for speed and size; it costs
/// accuracy and buys no speed at all.
///
/// | build | `DirectML` | CPU | download | batch-invariant | mean cosine to fp32 |
/// |-------|-----------|-----|----------|-----------------|---------------------|
/// | int8  | 11.8s     | 92.4s  | 112 MB | **no**, 0.9969 | 0.9960 |
/// | fp16  | **4.9s**  | 115.0s | 224 MB | yes, 0.999999  | **0.999999** |
/// | fp32  | 4.9s      | **93.3s** | 448 MB | yes         | reference |
///
/// int8 is the slowest of the three on a GPU, where dynamic quantisation pays for its
/// scales on every layer while fp16 runs on hardware built for it, and it is no faster on
/// this CPU either. It is also the only build whose vectors depend on what a review was
/// embedded alongside: activation scales are taken per tensor, and a tensor spans the batch.
///
/// So fp16 is the default. It is indistinguishable from the full graph, reproducible, the
/// fastest option where a GPU exists, and half the download of fp32. int8 remains for anyone
/// who needs the smallest download and can accept vectors that shift with batching, and fp32
/// for CPU-only runs, where it is the fastest of the three.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    /// Smallest download. Slowest on a GPU, and not batch-invariant.
    Int8,
    /// Indistinguishable from the full graph, and the fastest where a GPU exists.
    #[default]
    Float16,
    /// The graph as exported. The reference the others are judged against, fastest on CPU.
    Float32,
}

impl Precision {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Int8 => "int8",
            Self::Float16 => "fp16",
            Self::Float32 => "fp32",
        }
    }

    fn asset(self) -> Asset {
        match self {
            Self::Int8 => Asset {
                remote: "onnx/model_qint8_avx512_vnni.onnx",
                local: "model_int8.onnx",
                sha256: "dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88",
            },
            Self::Float16 => Asset {
                remote: "onnx/model_O4.onnx",
                local: "model_fp16.onnx",
                sha256: "4654c156f3e4171abc9c716cdb771bf9116455d15ac1aab364aeeede0e3205b0",
            },
            Self::Float32 => Asset {
                remote: "onnx/model.onnx",
                local: "model_fp32.onnx",
                sha256: "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665",
            },
        }
    }
}

const TOKENIZER: Asset = Asset {
    remote: "tokenizer.json",
    local: "tokenizer.json",
    sha256: "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
};

#[derive(Debug, Clone, Copy)]
pub struct DownloadProgress<'a> {
    pub file: &'a str,
    pub downloaded: u64,
    pub total: Option<u64>,
}

/// Default location for the model cache, shared across corpora.
#[must_use]
pub fn default_cache_dir() -> PathBuf {
    dirs_cache().join("steam-review-census").join("models")
}

fn dirs_cache() -> PathBuf {
    // Deliberately dependency-free: the platform conventions are three env vars, and a
    // crate for that is not worth the supply-chain surface.
    if let Some(dir) = std::env::var_os("LOCALAPPDATA") {
        return PathBuf::from(dir);
    }
    if let Some(dir) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(dir);
    }
    if let Some(home) = std::env::var_os("HOME") {
        let home = PathBuf::from(home);
        if cfg!(target_os = "macos") {
            return home.join("Library").join("Caches");
        }
        return home.join(".cache");
    }
    PathBuf::from(".cache")
}

/// Downloads any missing or corrupt model file into `cache_dir`.
///
/// # Errors
///
/// Returns [`Error::ModelChecksum`] if a downloaded file does not match its pinned hash,
/// and propagates transport and filesystem failures.
pub async fn ensure(
    cache_dir: &Path,
    precision: Precision,
    mut on_progress: impl FnMut(DownloadProgress),
) -> Result<()> {
    std::fs::create_dir_all(cache_dir)?;
    let http = reqwest::Client::builder()
        .user_agent(concat!("steam-review-census/", env!("CARGO_PKG_VERSION")))
        .build()?;

    for asset in [TOKENIZER, precision.asset()] {
        let path = cache_dir.join(asset.local);
        if path.is_file() && sha256_file(&path)? == asset.sha256 {
            continue;
        }
        download(&http, asset, &path, &mut on_progress).await?;

        let actual = sha256_file(&path)?;
        if actual != asset.sha256 {
            // A file that fails verification is removed rather than left to be picked up
            // by the next run, which would otherwise skip the download and load it.
            std::fs::remove_file(&path)?;
            return Err(Error::ModelChecksum {
                file: asset.local,
                expected: asset.sha256,
                actual,
            });
        }
    }
    Ok(())
}

async fn download(
    http: &reqwest::Client,
    asset: Asset,
    path: &Path,
    on_progress: &mut impl FnMut(DownloadProgress),
) -> Result<()> {
    let url = format!(
        "https://huggingface.co/{MODEL_ID}/resolve/main/{}",
        asset.remote
    );
    let mut response = http.get(&url).send().await?.error_for_status()?;
    let total = response.content_length();

    // Written beside the target and renamed, so an interrupted download is never mistaken
    // for a complete one on the next run.
    let partial = path.with_extension("partial");
    let mut file = std::fs::File::create(&partial)?;
    let mut downloaded = 0;

    while let Some(chunk) = response.chunk().await? {
        std::io::Write::write_all(&mut file, &chunk)?;
        downloaded += chunk.len() as u64;
        on_progress(DownloadProgress {
            file: asset.local,
            downloaded,
            total,
        });
    }
    drop(file);
    std::fs::rename(&partial, path)?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[must_use]
pub fn model_path(cache_dir: &Path, precision: Precision) -> PathBuf {
    cache_dir.join(precision.asset().local)
}

#[must_use]
pub fn tokenizer_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join("tokenizer.json")
}

/// Builds a session on the fastest backend this build supports and the machine provides.
///
/// GPU backends are opt-in at build time so the default binary needs no vendor runtime to
/// start. Each is *attempted* rather than assumed, because a build that supports `DirectML`
/// still has to run on machines with no suitable adapter. Which backend runs changes only
/// where the arithmetic happens, never which model runs, so results stay comparable.
///
/// # Errors
///
/// Fails if no backend, including the CPU fallback, can load the model.
pub fn session(cache_dir: &Path, precision: Precision) -> Result<(Session, &'static str)> {
    let path = model_path(cache_dir, precision);

    #[cfg(feature = "directml")]
    if let Ok(session) = try_session(&path, ort::ep::DirectML::default().build()) {
        return Ok((session, "directml"));
    }
    #[cfg(feature = "coreml")]
    if let Ok(session) = try_session(&path, ort::ep::CoreML::default().build()) {
        return Ok((session, "coreml"));
    }
    #[cfg(feature = "cuda")]
    if let Ok(session) = try_session(&path, ort::ep::CUDA::default().build()) {
        return Ok((session, "cuda"));
    }

    Ok((try_session(&path, ort::ep::CPU::default().build())?, "cpu"))
}

fn try_session(path: &Path, provider: ort::ep::ExecutionProviderDispatch) -> Result<Session> {
    let mut builder = Session::builder()?
        .with_execution_providers([provider])
        .map_err(ort::Error::from)?;
    Ok(builder.commit_from_file(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_pinned_hash_is_a_sha256() {
        for asset in [
            TOKENIZER,
            Precision::Int8.asset(),
            Precision::Float16.asset(),
            Precision::Float32.asset(),
        ] {
            assert_eq!(asset.sha256.len(), 64, "{}", asset.local);
            assert!(
                asset.sha256.chars().all(|c| c.is_ascii_hexdigit()),
                "{}",
                asset.local
            );
        }
    }

    #[test]
    fn hex_encodes_lowercase_and_pads() {
        assert_eq!(hex(&[0x00, 0x0f, 0xff]), "000fff");
    }

    #[test]
    fn the_cache_directory_is_namespaced_to_this_tool() {
        let dir = default_cache_dir();
        assert!(dir.to_string_lossy().contains("steam-review-census"));
        assert!(dir.ends_with("models"));
    }
}
