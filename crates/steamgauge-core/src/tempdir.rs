//! Scratch directory for tests, removed on drop, so no test writes where another can see it.
//!
//! A dependency for this would be a dependency in the shipped binary's tree for the sake of
//! twenty lines that only ever run under `cargo test`.

use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct Dir(PathBuf);

impl Dir {
    pub fn new() -> Self {
        // Named for the process and thread so two tests running at once cannot share one, and
        // cleared first so a previous run that was killed cannot leave a file behind that a
        // later run then reads as its own.
        let unique = format!(
            "steamgauge-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        );
        let path = std::env::temp_dir().join(unique);
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
