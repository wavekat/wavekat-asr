//! Generic HuggingFace download helper with byte-level progress.
//!
//! This module is backend-agnostic: any backend whose model weights live
//! on HuggingFace Hub can call [`download_files_with_progress`] to fetch
//! them into a caller-chosen directory while a `FnMut` closure receives
//! [`DownloadProgress`] updates as bytes arrive.
//!
//! Today only the [`crate::backends::sherpa_onnx`] backend uses this —
//! `sherpa_onnx::download_preset_with_progress` is a thin wrapper that
//! turns a `ModelPreset` into a `(repo_id, &[&str])` call into here. A
//! future Whisper / Qwen3-ASR / etc. backend can do the same.
//!
//! Paired with [`is_repo_cached`], a pure-filesystem probe that uses
//! the same `(repo_id, files)` pair to answer "are these already on
//! disk?" without touching the network — for consumers that need to
//! decide between rendering a "Download" affordance and a "Ready"
//! state in a UI hot path.
//!
//! Gated behind the `download` Cargo feature; the `sherpa-onnx` feature
//! turns it on transitively.

use std::path::{Path, PathBuf};

use crate::AsrError;

/// Byte-level progress for one file inside a multi-file download.
///
/// Passed to the callback of [`download_files_with_progress`]. Fires at
/// the start of each file (`bytes_done = 0`), repeatedly during streaming
/// as bytes arrive, and once on completion (`bytes_done == bytes_total`).
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    /// Filename inside the HuggingFace repo.
    pub file: String,
    /// 1-indexed position of `file` within the batch, paired with
    /// [`file_count`](Self::file_count) so a UI can render
    /// "file 2 of 4".
    pub file_index: usize,
    /// Total files in the batch.
    pub file_count: usize,
    /// Bytes downloaded so far for the current file. Resets per file.
    pub bytes_done: u64,
    /// Total bytes for the current file, reported by HuggingFace before
    /// streaming begins. `None` only for the very first call before
    /// metadata is known.
    pub bytes_total: Option<u64>,
}

/// Download every file in `files` from `repo_id` on HuggingFace Hub into
/// `dest_dir`, reporting byte progress as it goes.
///
/// On success, `dest_dir` contains every filename in `files` and is
/// directly loadable by a backend that consumes the files from a flat
/// directory. `dest_dir` is created if missing; existing files with the
/// same name are overwritten so a partial previous run retries cleanly.
///
/// `on_progress` runs synchronously on the calling thread inside the
/// hf-hub download loop — keep it cheap (channel send, atomic store) and
/// don't block on it.
///
/// hf-hub's own cache (controlled by the `HF_HOME` env var) still gets
/// populated as a side effect; callers that only want the files in
/// `dest_dir` can ignore it.
pub fn download_files_with_progress<F>(
    repo_id: &str,
    files: &[&str],
    dest_dir: &Path,
    mut on_progress: F,
) -> Result<(), AsrError>
where
    F: FnMut(DownloadProgress),
{
    use hf_hub::api::sync::Api;

    std::fs::create_dir_all(dest_dir).map_err(|e| {
        AsrError::Backend(format!(
            "creating model directory {}: {e}",
            dest_dir.display()
        ))
    })?;

    let file_count = files.len();

    let api = Api::new().map_err(|e| AsrError::Backend(format!("hf-hub init failed: {e}")))?;
    let repo = api.model(repo_id.to_string());

    for (idx, name) in files.iter().enumerate() {
        let file_index = idx + 1;

        // Fresh adapter per file: it borrows `on_progress` for the
        // duration of one `download_with_progress` call, so the next
        // iteration can re-borrow it for the next file.
        let adapter = CallbackProgress {
            file: (*name).to_string(),
            file_index,
            file_count,
            bytes_done: 0,
            bytes_total: None,
            on_progress: &mut on_progress,
        };

        tracing::debug!(
            repo_id,
            file = name,
            file_index,
            file_count,
            "fetching from HuggingFace with progress"
        );

        let src = repo
            .download_with_progress(name, adapter)
            .map_err(|e| AsrError::Backend(format!("hf-hub download of {name} failed: {e}")))?;

        let dest = dest_dir.join(name);
        // copy, not rename — the hf-hub blob is shared with its cache;
        // the user gets their own copy under `dest_dir` so they can move
        // / delete it without breaking the cache.
        std::fs::copy(&src, &dest).map_err(|e| {
            AsrError::Backend(format!(
                "copying {} to {}: {e}",
                src.display(),
                dest.display()
            ))
        })?;
    }

    Ok(())
}

/// Bridge between hf-hub's `Progress` trait and our
/// `FnMut(DownloadProgress)` callback. Borrows the user's closure so a
/// single closure can be reused across every file in the batch.
struct CallbackProgress<'a, F: FnMut(DownloadProgress)> {
    file: String,
    file_index: usize,
    file_count: usize,
    bytes_done: u64,
    bytes_total: Option<u64>,
    on_progress: &'a mut F,
}

impl<F: FnMut(DownloadProgress)> CallbackProgress<'_, F> {
    fn emit(&mut self) {
        (self.on_progress)(DownloadProgress {
            file: self.file.clone(),
            file_index: self.file_index,
            file_count: self.file_count,
            bytes_done: self.bytes_done,
            bytes_total: self.bytes_total,
        });
    }
}

impl<F: FnMut(DownloadProgress)> hf_hub::api::Progress for CallbackProgress<'_, F> {
    fn init(&mut self, size: usize, _filename: &str) {
        self.bytes_total = Some(size as u64);
        self.bytes_done = 0;
        self.emit();
    }

    fn update(&mut self, size: usize) {
        // hf-hub passes the chunk size, not the cumulative total.
        self.bytes_done = self.bytes_done.saturating_add(size as u64);
        self.emit();
    }

    fn finish(&mut self) {
        // Make sure the last emitted value is the file's full size;
        // some backends short the final `update` call.
        if let Some(total) = self.bytes_total {
            if self.bytes_done < total {
                self.bytes_done = total;
                self.emit();
            }
        }
    }
}

// ----- Offline cache probe -----------------------------------------------

/// Returns `true` iff every file in `files` is already present in the
/// local HuggingFace Hub cache for `repo_id`. Pure filesystem — no
/// network metadata fetch, no download — so safe to call in a UI hot
/// path that needs to decide between rendering a "Download" affordance
/// and a "Ready" indicator.
///
/// The cache root is resolved the same way `hf-hub` 0.5 does:
/// `$HF_HOME/hub` if `HF_HOME` is set, otherwise
/// `$HOME/.cache/huggingface/hub`. Returns `false` if neither can be
/// determined.
///
/// Pair with [`download_files_with_progress`] — the same `(repo_id,
/// files)` pair you pass there can be passed here to ask "do I still
/// need to download?".
///
/// # Why this exists
///
/// `hf-hub` 0.5 has no `HF_HUB_OFFLINE` support and no public
/// "lookup-only" download API, so a naïve "try to construct the
/// backend and see if it succeeds" probe will silently kick off a
/// large download whenever the cache is cold. Consumers that want to
/// stay offline during a UI render need a pure-filesystem alternative;
/// this is it.
pub fn is_repo_cached(repo_id: &str, files: &[&str]) -> bool {
    let Some(cache_root) = hf_hub_cache_root() else {
        return false;
    };
    snapshot_satisfies_files(&cache_root, repo_id, files)
}

/// Resolve the HF Hub on-disk cache root the same way `hf-hub` 0.5
/// does. Honors `HF_HOME` (appending `hub`), then falls back to
/// `$HOME/.cache/huggingface/hub`. Returns `None` only when neither
/// env var is set — pathological for any normal Unix or Windows
/// environment.
fn hf_hub_cache_root() -> Option<PathBuf> {
    if let Some(home) = std::env::var_os("HF_HOME") {
        let mut p = PathBuf::from(home);
        p.push("hub");
        return Some(p);
    }
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
    let mut p = PathBuf::from(home);
    p.extend([".cache", "huggingface", "hub"]);
    Some(p)
}

/// Inner probe: walk `<cache_root>/models--<owner>--<repo>/snapshots/`
/// and return `true` iff some snapshot dir contains every file in
/// `files`. Factored out from [`is_repo_cached`] so unit tests can
/// drive it against a temp directory without touching the user's
/// real HF cache or polluting env vars.
fn snapshot_satisfies_files(cache_root: &Path, repo_id: &str, files: &[&str]) -> bool {
    let repo_dir = cache_root.join(format!("models--{}", repo_id.replace('/', "--")));
    let snapshots = repo_dir.join("snapshots");
    let Ok(entries) = std::fs::read_dir(&snapshots) else {
        return false;
    };
    for entry in entries.flatten() {
        let snap = entry.path();
        if !snap.is_dir() {
            continue;
        }
        if files.iter().all(|f| snap.join(f).exists()) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exercises the hf-hub `Progress` bridge directly so we can assert
    /// the callback contract without a real network download.
    #[test]
    fn callback_progress_reports_init_update_and_finish() {
        use hf_hub::api::Progress;

        let events = std::cell::RefCell::new(Vec::<DownloadProgress>::new());
        let mut on_progress = |p: DownloadProgress| events.borrow_mut().push(p);

        let mut adapter = CallbackProgress {
            file: "encoder.onnx".to_string(),
            file_index: 1,
            file_count: 4,
            bytes_done: 0,
            bytes_total: None,
            on_progress: &mut on_progress,
        };

        adapter.init(1000, "encoder.onnx");
        adapter.update(400);
        adapter.update(600);
        adapter.finish();

        let got = events.into_inner();
        // 1 init + 2 updates = 3 emissions. `finish` should NOT add a
        // 4th since bytes_done already equals bytes_total.
        assert_eq!(got.len(), 3);

        assert_eq!(got[0].bytes_done, 0);
        assert_eq!(got[0].bytes_total, Some(1000));
        assert_eq!(got[0].file_index, 1);
        assert_eq!(got[0].file_count, 4);
        assert_eq!(got[0].file, "encoder.onnx");

        assert_eq!(got[1].bytes_done, 400);
        assert_eq!(got[2].bytes_done, 1000);
    }

    /// If hf-hub's last `update` short-counts (we've seen this on retried
    /// downloads), `finish` should synthesize a final 100% event so the UI
    /// can't get stuck at 99%.
    #[test]
    fn callback_progress_finish_synthesizes_completion() {
        use hf_hub::api::Progress;

        let events = std::cell::RefCell::new(Vec::<DownloadProgress>::new());
        let mut on_progress = |p: DownloadProgress| events.borrow_mut().push(p);

        let mut adapter = CallbackProgress {
            file: "tokens.txt".to_string(),
            file_index: 4,
            file_count: 4,
            bytes_done: 0,
            bytes_total: None,
            on_progress: &mut on_progress,
        };

        adapter.init(500, "tokens.txt");
        adapter.update(400); // short
        adapter.finish(); // should push a synthetic 500/500

        let got = events.into_inner();
        assert_eq!(got.len(), 3);
        assert_eq!(got.last().unwrap().bytes_done, 500);
        assert_eq!(got.last().unwrap().bytes_total, Some(500));
    }

    /// Walk the on-disk layout `hf-hub` writes
    /// (`<cache_root>/models--<owner>--<repo>/snapshots/<commit>/<file>`)
    /// and verify the probe (a) returns false for a missing repo,
    /// partial files, or empty snapshot dirs and (b) returns true
    /// once every requested file is present in at least one snapshot.
    /// Pure-fs so it runs in CI without a network or a real model.
    #[test]
    fn snapshot_probe_matches_hf_hub_layout() {
        let root = unique_temp_dir("snapshot-probe");
        let _cleanup = scopeguard_remove(root.clone());
        std::fs::create_dir_all(&root).unwrap();

        let repo_id = "csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20";
        let files = ["encoder.onnx", "decoder.onnx", "tokens.txt"];

        assert!(!snapshot_satisfies_files(&root, repo_id, &files));

        let repo_dir = root.join(format!("models--{}", repo_id.replace('/', "--")));
        let snap_a = repo_dir.join("snapshots").join("abc123");
        std::fs::create_dir_all(&snap_a).unwrap();
        assert!(!snapshot_satisfies_files(&root, repo_id, &files));

        std::fs::write(snap_a.join("encoder.onnx"), b"").unwrap();
        std::fs::write(snap_a.join("decoder.onnx"), b"").unwrap();
        assert!(!snapshot_satisfies_files(&root, repo_id, &files));

        std::fs::write(snap_a.join("tokens.txt"), b"").unwrap();
        assert!(snapshot_satisfies_files(&root, repo_id, &files));

        // A second, half-complete snapshot must not invalidate the
        // first complete one.
        let snap_b = repo_dir.join("snapshots").join("def456");
        std::fs::create_dir_all(&snap_b).unwrap();
        std::fs::write(snap_b.join("encoder.onnx"), b"").unwrap();
        assert!(snapshot_satisfies_files(&root, repo_id, &files));
    }

    /// Slash-bearing repo ids must be flattened to the `<owner>--<repo>`
    /// form `hf-hub` writes; without that the probe would silently look
    /// in a nonexistent directory and always report `false`.
    #[test]
    fn snapshot_probe_flattens_slashes_in_repo_id() {
        let root = unique_temp_dir("slash-flatten");
        let _cleanup = scopeguard_remove(root.clone());
        let repo_id = "owner/repo";
        let snap = root
            .join("models--owner--repo")
            .join("snapshots")
            .join("c0");
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::write(snap.join("model.onnx"), b"").unwrap();
        assert!(snapshot_satisfies_files(&root, repo_id, &["model.onnx"]));
    }

    /// Unique-per-test scratch directory under the OS temp dir. We
    /// don't pull `tempfile` in just for this — pid + nanosecond
    /// timestamp is enough collision protection for a unit test, and
    /// the [`scopeguard_remove`] guard cleans up regardless of
    /// pass/fail.
    fn unique_temp_dir(label: &str) -> std::path::PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let pid = std::process::id();
        std::env::temp_dir().join(format!("wavekat-asr-{label}-{pid}-{nanos}"))
    }

    fn scopeguard_remove(path: std::path::PathBuf) -> impl Drop {
        struct Guard(std::path::PathBuf);
        impl Drop for Guard {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        Guard(path)
    }
}
