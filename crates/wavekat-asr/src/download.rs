//! Generic HuggingFace download helper with byte-level progress.
//!
//! This module is backend-agnostic: any backend whose model weights live
//! on HuggingFace Hub can call [`download_files_with_progress`] to fetch
//! them into a caller-chosen directory while a `FnMut` closure receives
//! [`DownloadProgress`] updates as bytes arrive.
//!
//! The sherpa-onnx backend's `download_preset_with_progress` is a thin
//! wrapper that turns a `ModelPreset` into a `(repo_id, &[&str])` call
//! into here; the R2T2 backend uses [`download_pinned_to_cache`] for
//! its GGUF weights.
//!
//! Paired with [`is_repo_cached`], a pure-filesystem probe that uses
//! the same `(repo_id, files)` pair to answer "are these already on
//! disk?" without touching the network — for consumers that need to
//! decide between rendering a "Download" affordance and a "Ready"
//! state in a UI hot path.
//!
//! For networks where `huggingface.co` is unreachable,
//! [`download_pinned_to_cache`] fetches a pinned snapshot from an
//! ordered list of [`DownloadSource`]s (HuggingFace, or any mirror that
//! serves the same `resolve` URL layout), verifies each file's SHA-256,
//! and writes the result into the same cache `hf-hub` reads.
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

// ----- Pinned downloads from any HF-layout host ---------------------------

/// One file of a pinned model snapshot: its name inside the repo plus
/// the exact size and SHA-256 it must have.
///
/// Pinning makes a download verifiable independent of where the bytes
/// came from, which is what lets [`download_pinned_to_cache`] fall back
/// to a mirror without trusting it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PinnedFile {
    /// Filename inside the repo.
    pub name: &'static str,
    /// Lowercase hex SHA-256 of the file's contents.
    pub sha256: &'static str,
    /// Exact size in bytes.
    pub size: u64,
}

/// A host that serves repo files in the HuggingFace `resolve` layout:
/// `{base_url}/{repo_id}/resolve/{revision}/{file}`.
///
/// HuggingFace itself is one such host; a mirror (a community one, or a
/// static bucket populated with the same paths) is another. Useful for
/// networks where `huggingface.co` is unreachable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadSource {
    base_url: String,
}

impl DownloadSource {
    /// Base URL of HuggingFace Hub.
    pub const HUGGING_FACE_URL: &'static str = "https://huggingface.co";

    /// HuggingFace Hub itself.
    pub fn hugging_face() -> Self {
        Self::mirror(Self::HUGGING_FACE_URL)
    }

    /// Any other host serving the HuggingFace `resolve` layout under
    /// `base_url`. A trailing slash is ignored.
    pub fn mirror(base_url: impl Into<String>) -> Self {
        let base_url = base_url.into().trim_end_matches('/').to_string();
        Self { base_url }
    }

    /// Base URL, without a trailing slash.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Full URL of `file` at `revision` in `repo_id` on this host.
    pub fn file_url(&self, repo_id: &str, revision: &str, file: &str) -> String {
        format!("{}/{repo_id}/resolve/{revision}/{file}", self.base_url)
    }
}

/// How long to wait for a TCP/TLS connection before giving up on a
/// source. Short on purpose: a blocked host should fall through to the
/// next source quickly instead of stalling the download.
const CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
/// How long to wait for response headers once connected.
const RESPONSE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Download a pinned snapshot of `repo_id` at `revision` into the local
/// HuggingFace Hub cache, trying `sources` in order and verifying every
/// file's size and SHA-256.
///
/// Files land where `hf-hub` itself would look for them
/// (`<cache>/models--<owner>--<repo>/snapshots/<revision>/<file>`, with
/// `refs/main` pointing at `revision`), so after this returns
/// [`is_repo_cached`] reports `true` and a backend that resolves its
/// files through `hf-hub` finds them without touching the network.
///
/// Source fallback is sticky: when a source fails (connection error,
/// HTTP error, or a file that doesn't match its pin) the current file is
/// retried from the next source, and later files start from that source
/// too — a host that refused one connection is unlikely to accept the
/// next. Files already present at their pinned size are skipped, so an
/// interrupted run resumes at file granularity.
///
/// `on_progress` fires as in [`download_files_with_progress`];
/// `bytes_total` is always the pinned size.
pub fn download_pinned_to_cache<F>(
    repo_id: &str,
    revision: &str,
    files: &[PinnedFile],
    sources: &[DownloadSource],
    on_progress: F,
) -> Result<PathBuf, AsrError>
where
    F: FnMut(DownloadProgress),
{
    let cache_root = hf_hub_cache_root()
        .ok_or_else(|| AsrError::Backend("cannot locate the HuggingFace cache directory".into()))?;
    download_pinned_into(&cache_root, repo_id, revision, files, sources, on_progress)
}

/// [`download_pinned_to_cache`] against an explicit cache root, so tests
/// can use a temp directory.
fn download_pinned_into<F>(
    cache_root: &Path,
    repo_id: &str,
    revision: &str,
    files: &[PinnedFile],
    sources: &[DownloadSource],
    mut on_progress: F,
) -> Result<PathBuf, AsrError>
where
    F: FnMut(DownloadProgress),
{
    let repo_dir = cache_root.join(repo_folder_name(repo_id));
    let snapshot = repo_dir.join("snapshots").join(revision);
    let file_count = files.len();
    let agent = http_agent();
    let mut source_idx = 0;
    let mut failures: Vec<String> = Vec::new();

    for (idx, file) in files.iter().enumerate() {
        let dest = snapshot.join(file.name);
        let emit = |on_progress: &mut F, bytes_done: u64| {
            on_progress(DownloadProgress {
                file: file.name.to_string(),
                file_index: idx + 1,
                file_count,
                bytes_done,
                bytes_total: Some(file.size),
            })
        };

        if std::fs::metadata(&dest).is_ok_and(|m| m.is_file() && m.len() == file.size) {
            emit(&mut on_progress, file.size);
            continue;
        }

        loop {
            let Some(source) = sources.get(source_idx) else {
                return Err(AsrError::Backend(if failures.is_empty() {
                    "no download source configured".to_string()
                } else {
                    format!(
                        "download of {} failed from every source: {}",
                        file.name,
                        failures.join("; ")
                    )
                }));
            };
            let url = source.file_url(repo_id, revision, file.name);
            tracing::debug!(
                url,
                file_index = idx + 1,
                file_count,
                "fetching pinned file"
            );
            match fetch_verified(&agent, &url, &dest, file, |done| {
                emit(&mut on_progress, done)
            }) {
                Ok(()) => break,
                Err(FetchError::Local(err)) => return Err(err),
                Err(FetchError::Remote(msg)) => {
                    tracing::warn!(url, error = %msg, "download source failed, trying the next one");
                    failures.push(format!("{}: {msg}", source.base_url()));
                    source_idx += 1;
                }
            }
        }
    }

    let refs = repo_dir.join("refs");
    std::fs::create_dir_all(&refs)?;
    std::fs::write(refs.join("main"), revision)?;
    Ok(snapshot)
}

/// Why fetching one file failed: the source's fault (try another one)
/// or ours (a disk error that no other source would fix).
#[derive(Debug)]
enum FetchError {
    Remote(String),
    Local(AsrError),
}

impl From<std::io::Error> for FetchError {
    fn from(err: std::io::Error) -> Self {
        FetchError::Local(AsrError::Io(err))
    }
}

/// Stream `url` into `dest` via a `.part` file, hashing as it goes, and
/// move it into place only if size and SHA-256 match `file`.
fn fetch_verified(
    agent: &ureq::Agent,
    url: &str,
    dest: &Path,
    file: &PinnedFile,
    mut on_bytes: impl FnMut(u64),
) -> Result<(), FetchError> {
    use sha2::{Digest, Sha256};
    use std::io::{Read, Write};

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let part = dest.with_file_name(format!(
        "{}.part",
        dest.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("download")
    ));

    on_bytes(0);
    let mut response = agent
        .get(url)
        .call()
        .map_err(|e| FetchError::Remote(e.to_string()))?;
    let mut reader = response.body_mut().as_reader();

    let result = (|| {
        let mut out = std::fs::File::create(&part)?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 64 * 1024];
        let mut done: u64 = 0;
        loop {
            let n = reader
                .read(&mut buf)
                .map_err(|e| FetchError::Remote(e.to_string()))?;
            if n == 0 {
                break;
            }
            done += n as u64;
            if done > file.size {
                return Err(FetchError::Remote(format!(
                    "{} is larger than expected ({} bytes)",
                    file.name, file.size
                )));
            }
            hasher.update(&buf[..n]);
            out.write_all(&buf[..n])?;
            on_bytes(done);
        }
        out.flush()?;
        drop(out);
        if done != file.size {
            return Err(FetchError::Remote(format!(
                "{} is incomplete ({done} of {} bytes)",
                file.name, file.size
            )));
        }
        let digest = to_hex(&hasher.finalize());
        if !digest.eq_ignore_ascii_case(file.sha256) {
            return Err(FetchError::Remote(format!(
                "{} failed its checksum (got {digest})",
                file.name
            )));
        }
        // Rename can't replace an existing file on every platform; a
        // wrong-sized leftover is the only thing that could be there.
        match std::fs::remove_file(dest) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
        std::fs::rename(&part, dest)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&part);
    }
    result
}

fn http_agent() -> ureq::Agent {
    use ureq::tls::{TlsConfig, TlsProvider};
    ureq::Agent::config_builder()
        .tls_config(
            TlsConfig::builder()
                .provider(TlsProvider::NativeTls)
                .build(),
        )
        .timeout_connect(Some(CONNECT_TIMEOUT))
        .timeout_recv_response(Some(RESPONSE_TIMEOUT))
        .build()
        .into()
}

fn to_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

/// `hf-hub`'s per-repo cache directory name: `models--<owner>--<repo>`.
fn repo_folder_name(repo_id: &str) -> String {
    format!("models--{}", repo_id.replace('/', "--"))
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
    let repo_dir = cache_root.join(repo_folder_name(repo_id));
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

    // ----- pinned downloads ------------------------------------------

    /// Minimal HTTP/1.1 server: serves `routes` (path → (status, body))
    /// until the test ends, one request per connection. Returns its base
    /// URL and a log of the paths it was asked for.
    fn serve(
        routes: Vec<(&'static str, u16, Vec<u8>)>,
    ) -> (String, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
        use std::io::{BufRead, BufReader, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let log_t = log.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                let path = line.split_whitespace().nth(1).unwrap_or("").to_string();
                loop {
                    let mut h = String::new();
                    if reader.read_line(&mut h).unwrap() <= 2 {
                        break;
                    }
                }
                log_t.lock().unwrap().push(path.clone());
                let (status, body) = routes
                    .iter()
                    .find(|(p, _, _)| *p == path)
                    .map(|(_, s, b)| (*s, b.clone()))
                    .unwrap_or((404, b"not found".to_vec()));
                let head = format!(
                    "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(&body);
            }
        });
        (base, log)
    }

    /// A base URL nothing listens on — connecting fails immediately,
    /// like a blocked host that actively refuses.
    fn refused_base() -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        format!("http://{addr}")
    }

    const REPO: &str = "owner/model";
    const REV: &str = "0123456789abcdef0123456789abcdef01234567";

    fn pinned(name: &'static str, body: &[u8]) -> PinnedFile {
        use sha2::{Digest, Sha256};
        let sha = to_hex(&Sha256::digest(body));
        PinnedFile {
            name,
            sha256: Box::leak(sha.into_boxed_str()),
            size: body.len() as u64,
        }
    }

    fn route(name: &str) -> &'static str {
        Box::leak(format!("/{REPO}/resolve/{REV}/{name}").into_boxed_str())
    }

    #[test]
    fn source_file_url_uses_resolve_layout_and_trims_slash() {
        let src = DownloadSource::mirror("https://example.com/models/");
        assert_eq!(src.base_url(), "https://example.com/models");
        assert_eq!(
            src.file_url("owner/model", "abc", "encoder.onnx"),
            "https://example.com/models/owner/model/resolve/abc/encoder.onnx"
        );
        assert_eq!(
            DownloadSource::hugging_face().file_url("a/b", "main", "t.txt"),
            "https://huggingface.co/a/b/resolve/main/t.txt"
        );
    }

    /// The case that motivated this: the first host refuses the
    /// connection, so every file comes from the next one — and lands
    /// where `hf-hub` and the cache probe look.
    #[test]
    fn falls_back_to_next_source_and_populates_hf_cache() {
        let root = unique_temp_dir("pinned-fallback");
        let _cleanup = scopeguard_remove(root.clone());
        let files = [
            pinned("encoder.onnx", b"enc-bytes"),
            pinned("tokens.txt", b"a 0\nb 1\n"),
        ];
        let (mirror, log) = serve(vec![
            (route("encoder.onnx"), 200, b"enc-bytes".to_vec()),
            (route("tokens.txt"), 200, b"a 0\nb 1\n".to_vec()),
        ]);
        let sources = [
            DownloadSource::mirror(refused_base()),
            DownloadSource::mirror(mirror),
        ];

        let mut events = Vec::new();
        let snap =
            download_pinned_into(&root, REPO, REV, &files, &sources, |p| events.push(p)).unwrap();

        assert_eq!(snap, root.join("models--owner--model/snapshots").join(REV));
        assert_eq!(
            std::fs::read(snap.join("encoder.onnx")).unwrap(),
            b"enc-bytes"
        );
        assert_eq!(
            std::fs::read_to_string(root.join("models--owner--model/refs/main")).unwrap(),
            REV
        );
        assert!(snapshot_satisfies_files(
            &root,
            REPO,
            &["encoder.onnx", "tokens.txt"]
        ));
        assert!(!snap.join("encoder.onnx.part").exists());
        // Sticky: the refused host isn't retried for the second file.
        assert_eq!(log.lock().unwrap().len(), 2);
        let last = events.last().unwrap();
        assert_eq!((last.file_index, last.file_count), (2, 2));
        assert_eq!(last.bytes_done, 8);
        assert_eq!(last.bytes_total, Some(8));
    }

    /// A source that serves the wrong bytes is not trusted: the file is
    /// rejected and the next source is used.
    #[test]
    fn checksum_mismatch_falls_through_to_next_source() {
        let root = unique_temp_dir("pinned-mismatch");
        let _cleanup = scopeguard_remove(root.clone());
        let files = [pinned("model.onnx", b"genuine!")];
        let (bad, _) = serve(vec![(route("model.onnx"), 200, b"tampered".to_vec())]);
        let (good, _) = serve(vec![(route("model.onnx"), 200, b"genuine!".to_vec())]);
        let sources = [DownloadSource::mirror(bad), DownloadSource::mirror(good)];

        let snap = download_pinned_into(&root, REPO, REV, &files, &sources, |_| {}).unwrap();
        assert_eq!(std::fs::read(snap.join("model.onnx")).unwrap(), b"genuine!");
    }

    /// With no good source left, nothing is left behind that the cache
    /// probe would mistake for a finished download.
    #[test]
    fn fails_when_every_source_fails_and_leaves_no_file() {
        let root = unique_temp_dir("pinned-allfail");
        let _cleanup = scopeguard_remove(root.clone());
        let files = [pinned("model.onnx", b"genuine!")];
        let (bad, _) = serve(vec![(route("model.onnx"), 200, b"tampered".to_vec())]);
        let (missing, _) = serve(vec![]);
        let sources = [DownloadSource::mirror(bad), DownloadSource::mirror(missing)];

        let err = download_pinned_into(&root, REPO, REV, &files, &sources, |_| {}).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("checksum"), "{msg}");
        assert!(msg.contains("404"), "{msg}");
        let snap = root.join("models--owner--model/snapshots").join(REV);
        assert!(!snap.join("model.onnx").exists());
        assert!(!snap.join("model.onnx.part").exists());
        assert!(!root.join("models--owner--model/refs/main").exists());
    }

    #[test]
    fn truncated_body_is_rejected() {
        let root = unique_temp_dir("pinned-short");
        let _cleanup = scopeguard_remove(root.clone());
        let mut file = pinned("model.onnx", b"full body");
        file.size = 100;
        let (host, _) = serve(vec![(route("model.onnx"), 200, b"full body".to_vec())]);
        let err = download_pinned_into(
            &root,
            REPO,
            REV,
            &[file],
            &[DownloadSource::mirror(host)],
            |_| {},
        )
        .unwrap_err();
        assert!(err.to_string().contains("incomplete"), "{err}");
    }

    /// Files already at their pinned size are not fetched again, so a
    /// re-run after an interruption only downloads what's missing.
    #[test]
    fn skips_files_already_present_at_pinned_size() {
        let root = unique_temp_dir("pinned-resume");
        let _cleanup = scopeguard_remove(root.clone());
        let files = [pinned("a.onnx", b"aaaa"), pinned("b.txt", b"bb")];
        let snap = root.join("models--owner--model/snapshots").join(REV);
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::write(snap.join("a.onnx"), b"aaaa").unwrap();
        let (host, log) = serve(vec![(route("b.txt"), 200, b"bb".to_vec())]);

        download_pinned_into(
            &root,
            REPO,
            REV,
            &files,
            &[DownloadSource::mirror(host)],
            |_| {},
        )
        .unwrap();
        assert_eq!(*log.lock().unwrap(), vec![route("b.txt").to_string()]);
    }

    #[test]
    fn no_sources_is_an_error() {
        let root = unique_temp_dir("pinned-nosource");
        let _cleanup = scopeguard_remove(root.clone());
        let err =
            download_pinned_into(&root, REPO, REV, &[pinned("x", b"x")], &[], |_| {}).unwrap_err();
        assert!(err.to_string().contains("no download source"), "{err}");
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
