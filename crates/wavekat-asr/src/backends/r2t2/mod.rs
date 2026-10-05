//! Confucius4-R2T2 streaming ASR backend (llama.cpp).
//!
//! [Confucius4-R2T2] is NetEase Youdao's streaming fine-tune of
//! Qwen3-ASR-1.7B. Its output is **append-only**: committed text is
//! never revised, so partials can be acted on immediately. Chinese and
//! English are the primary languages; others work with lower accuracy.
//!
//! This backend runs the official GGUF export through llama.cpp's
//! multimodal (`mtmd`) API, so the same code runs on macOS (Metal,
//! enabled automatically), Linux and Windows (CPU by default; enable
//! `llama-cpp-2`'s `cuda` or `vulkan` feature in your own manifest for
//! a GPU build). Weights download from HuggingFace on first use
//! ([`R2T2_Q8_0`] by default, ~2.5 GB).
//!
//! # How it streams
//!
//! Audio is decoded in fixed chunks (160 ms by default, 80 ms–2 s
//! supported). Every step re-feeds the whole utterance, prompted with
//! the text committed so far; see [`stream`] for the loop. Because the
//! cost of a step grows with utterance length, the backend splits the
//! stream into utterances with a WebRTC VAD endpointer and resets after
//! each one (or after [`R2t2Config::max_utterance_ms`]).
//!
//! Decoding runs on a worker thread: [`StreamingAsr::push_audio`] only
//! enqueues audio. When decoding falls behind real time, queued chunks
//! are merged into one larger step (up to 2 s) instead of building a
//! backlog — see [`R2t2Config::catch_up`]. [`R2t2Asr::stats`] reports
//! per-step timing so callers can see whether a machine keeps up.
//!
//! # Example
//!
//! ```no_run
//! use wavekat_asr::backends::r2t2::R2t2Asr;
//! use wavekat_asr::{AudioFrame, Channel, StreamingAsr, TranscriptEvent};
//!
//! let (mut asr, rx) = R2t2Asr::new().unwrap();
//! let samples = vec![0.0f32; 16_000];
//! asr.push_audio(&AudioFrame::new(samples.as_slice(), 16_000), Channel::Local)
//!     .unwrap();
//! asr.finish().unwrap();
//! for event in rx.try_iter() {
//!     if let TranscriptEvent::Final { text, .. } = event {
//!         println!("final: {text}");
//!     }
//! }
//! println!("{:?}", asr.stats());
//! ```
//!
//! [Confucius4-R2T2]: https://github.com/netease-youdao/Confucius4-R2T2
//!
//! # Licence
//!
//! The inference code here is Apache-2.0 like the rest of the crate. The
//! model weights are governed by the NetEase Youdao Model Use License
//! Agreement, which consumers must review before shipping them.

mod engine;
pub mod stream;
mod text;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use wavekat_vad::backends::webrtc::WebRtcVad;
use wavekat_vad::VoiceActivityDetector;

pub use engine::R2t2Model;
pub use wavekat_vad::backends::webrtc::WebRtcVadMode as VadMode;

pub use crate::download::{DownloadProgress, DownloadSource, PinnedFile};
use crate::{AsrError, AudioFrame, Channel, StreamingAsr, TranscriptEvent};

use engine::{Engine, EngineConfig};
use stream::{Schedule, Utterance, SAMPLE_RATE};

/// A pinned pair of GGUF files: the language model and the audio
/// projector (`mmproj`, the audio encoder).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct R2t2Weights {
    /// HuggingFace repo id.
    pub model_id: &'static str,
    /// HuggingFace commit the hashes were taken from.
    pub revision: &'static str,
    /// Language-model GGUF.
    pub model: PinnedFile,
    /// Audio-projector GGUF.
    pub mmproj: PinnedFile,
}

impl R2t2Weights {
    /// Both files, model first.
    pub fn pinned_files(&self) -> [PinnedFile; 2] {
        [self.model, self.mmproj]
    }

    /// Filenames, in the same order as [`pinned_files`](Self::pinned_files).
    pub fn files(&self) -> [&'static str; 2] {
        [self.model.name, self.mmproj.name]
    }

    /// Exact download size in bytes.
    pub fn size_bytes(&self) -> u64 {
        self.model.size + self.mmproj.size
    }

    /// `true` iff both files are in the local HuggingFace cache. Pure
    /// filesystem; never touches the network.
    pub fn is_cached(&self) -> bool {
        crate::download::is_repo_cached(self.model_id, &self.files())
    }
}

const GGUF_REPO: &str = "netease-youdao/Confucius4-R2T2-GGUF";
const GGUF_REVISION: &str = "86ff0251cb9f456b63aeef5f80137f104e22869a";

/// The f16 audio encoder is used with every preset: the upstream notes
/// that llama.cpp's encoder is the less robust half on quiet onsets, so
/// it isn't quantised further.
const MMPROJ_F16: PinnedFile = PinnedFile {
    name: "mmproj-Confucius4-R2T2-f16.gguf",
    sha256: "0057b28b8814e431a28e3d1002343eef2d1676bbf8b0625351ae4550ac4e595d",
    size: 641_773_984,
};

/// 8-bit language model + f16 encoder (~2.5 GB). Default; accuracy is
/// effectively that of the f16 weights.
pub const R2T2_Q8_0: R2t2Weights = R2t2Weights {
    model_id: GGUF_REPO,
    revision: GGUF_REVISION,
    model: PinnedFile {
        name: "Confucius4-R2T2-Q8_0.gguf",
        sha256: "151097e43957a19984ea7de66e8144ce69b95039eb31c93da4f58db367e455c3",
        size: 1_834_422_208,
    },
    mmproj: MMPROJ_F16,
};

/// 4-bit language model + f16 encoder (~1.7 GB), for smaller devices.
pub const R2T2_Q4_K_M: R2t2Weights = R2t2Weights {
    model_id: GGUF_REPO,
    revision: GGUF_REVISION,
    model: PinnedFile {
        name: "Confucius4-R2T2-Q4_K_M.gguf",
        sha256: "fa3cb46c8c3a66a58812b9098ba6e96a0266d4e8c9b3cf5ba34432fd2f9f6466",
        size: 1_107_404_736,
    },
    mmproj: MMPROJ_F16,
};

/// Unquantised f16 language model + f16 encoder (~4.1 GB). Reference
/// accuracy; mostly useful for comparisons.
pub const R2T2_F16: R2t2Weights = R2t2Weights {
    model_id: GGUF_REPO,
    revision: GGUF_REVISION,
    model: PinnedFile {
        name: "Confucius4-R2T2-f16.gguf",
        sha256: "4fa26b8f2d6db9122752d3e28694c26083bad95f6263d2f7619c7f98f2480b97",
        size: 3_447_345_088,
    },
    mmproj: MMPROJ_F16,
};

/// Speech/silence segmentation used to split the stream into utterances.
#[derive(Debug, Clone)]
pub struct Endpointing {
    /// WebRTC VAD aggressiveness.
    pub vad_mode: VadMode,
    /// Consecutive speech needed to open an utterance.
    pub min_speech_ms: u32,
    /// Trailing silence that closes an utterance.
    pub trailing_silence_ms: u32,
    /// Audio kept from before the speech onset so the first syllable
    /// isn't clipped.
    pub preroll_ms: u32,
}

impl Default for Endpointing {
    fn default() -> Self {
        Self {
            vad_mode: VadMode::Aggressive,
            min_speech_ms: 90,
            trailing_silence_ms: 800,
            preroll_ms: 300,
        }
    }
}

/// Configuration for [`R2t2Asr`].
///
/// Model files resolve in this order:
/// 1. [`model_dir`](Self::model_dir), if set — both files of
///    [`weights`](Self::weights) are looked up by name inside it.
/// 2. The `WAVEKAT_ASR_R2T2_DIR` environment variable, same lookup.
/// 3. The HuggingFace cache, downloading (and verifying) on first use.
#[derive(Debug, Clone)]
pub struct R2t2Config {
    /// Which GGUF files to run.
    pub weights: R2t2Weights,
    /// Local directory holding the weights. Skips the download.
    pub model_dir: Option<PathBuf>,
    /// Force a language (`"Chinese"`, `"English"`, …). `None` lets the
    /// model detect it, which costs a few tokens at the start of each
    /// utterance.
    pub language: Option<String>,
    /// Context / hotword hint, placed in the system prompt.
    pub context: String,
    /// Audio added per decode step. 80–2000 ms; the model was tuned
    /// around 160 ms.
    pub chunk_ms: u32,
    /// Extra audio collected before the first decode of an utterance.
    pub lookahead_ms: u32,
    /// Tokens held back from the end of each hypothesis before
    /// committing (the rollback window).
    pub unfixed_token_num: usize,
    /// Merge queued chunks into one step (up to 2 s of audio) when
    /// decoding falls behind real time. Disable to reproduce the
    /// reference's fixed schedule exactly, e.g. for offline evaluation.
    pub catch_up: bool,
    /// VAD endpointing. `None` treats the whole stream as speech and
    /// relies on [`max_utterance_ms`](Self::max_utterance_ms) alone.
    pub endpointing: Option<Endpointing>,
    /// Hard cap on utterance length. Step cost grows with utterance
    /// length, so long monologues are cut here.
    pub max_utterance_ms: u32,
    /// Offload to the GPU when the build has one (Metal on macOS).
    pub use_gpu: bool,
    /// CPU threads for llama.cpp.
    pub n_threads: i32,
}

impl Default for R2t2Config {
    fn default() -> Self {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get().min(8))
            .unwrap_or(4);
        Self {
            weights: R2T2_Q8_0,
            model_dir: None,
            language: None,
            context: String::new(),
            chunk_ms: 160,
            lookahead_ms: 160,
            unfixed_token_num: 1,
            catch_up: true,
            endpointing: Some(Endpointing::default()),
            max_utterance_ms: 30_000,
            use_gpu: true,
            n_threads: threads as i32,
        }
    }
}

impl R2t2Config {
    fn validate(&self) -> Result<(), AsrError> {
        if !(80..=2000).contains(&self.chunk_ms) {
            return Err(AsrError::Backend(format!(
                "chunk_ms must be within 80..=2000, got {}",
                self.chunk_ms
            )));
        }
        if self.max_utterance_ms < self.chunk_ms + self.lookahead_ms {
            return Err(AsrError::Backend(
                "max_utterance_ms must cover at least one chunk plus lookahead".into(),
            ));
        }
        Ok(())
    }

    fn schedule(&self) -> Schedule {
        let chunk = ms_to_samples(self.chunk_ms);
        Schedule {
            chunk,
            lookahead: ms_to_samples(self.lookahead_ms),
            max_chunks_per_step: if self.catch_up {
                (ms_to_samples(2000) / chunk).max(1)
            } else {
                1
            },
            unfixed_token_num: self.unfixed_token_num,
        }
    }

    /// Context size: 13 audio tokens per second of utterance, plus room
    /// for the prompt and transcript.
    fn n_ctx(&self) -> u32 {
        let audio_tokens = self.max_utterance_ms.div_ceil(1000) * 13;
        (audio_tokens + 1024).max(2048)
    }

    fn resolve_files(&self) -> Result<(PathBuf, PathBuf), AsrError> {
        let dir = self
            .model_dir
            .clone()
            .or_else(|| std::env::var_os("WAVEKAT_ASR_R2T2_DIR").map(PathBuf::from));
        let dir = match dir {
            Some(dir) => dir,
            None => {
                download_weights_to_cache(&self.weights, &[DownloadSource::hugging_face()], |_| {})?
            }
        };
        let model = dir.join(self.weights.model.name);
        let mmproj = dir.join(self.weights.mmproj.name);
        for path in [&model, &mmproj] {
            if !path.exists() {
                return Err(AsrError::Backend(format!(
                    "model file not found: {}",
                    path.display()
                )));
            }
        }
        Ok((model, mmproj))
    }
}

/// Download `weights` at their pinned revision into the HuggingFace
/// cache, verifying each file's SHA-256, and return the snapshot
/// directory. Files already present at the right size are skipped, so
/// this is also a cheap way to locate a cached snapshot.
pub fn download_weights_to_cache<F>(
    weights: &R2t2Weights,
    sources: &[DownloadSource],
    on_progress: F,
) -> Result<PathBuf, AsrError>
where
    F: FnMut(DownloadProgress),
{
    crate::download::download_pinned_to_cache(
        weights.model_id,
        weights.revision,
        &weights.pinned_files(),
        sources,
        on_progress,
    )
}

/// Running performance counters. Snapshot with [`R2t2Asr::stats`].
#[derive(Debug, Clone, Default)]
pub struct R2t2Stats {
    /// Decode steps run (each re-feeds the current utterance).
    pub decode_steps: u64,
    /// Steps that merged more than one chunk because decoding was
    /// behind real time.
    pub merged_steps: u64,
    /// Wall time of the most recent step.
    pub last_step_ms: f64,
    /// Mean step wall time.
    pub mean_step_ms: f64,
    /// Slowest step so far.
    pub max_step_ms: f64,
    /// Total decode wall time, steps and utterance flushes.
    pub decode_ms: f64,
    /// Audio advanced by decoding, in ms.
    pub audio_ms: f64,
    /// Real-time factor: `decode_ms / audio_ms`. Below 1.0 keeps up.
    pub rtf: f64,
    /// Time from the newest decoded sample arriving in `push_audio` to
    /// its step finishing, for the most recent step: queueing plus
    /// compute. Excludes the model's own look-ahead.
    pub last_lag_ms: f64,
    /// Mean of [`last_lag_ms`](Self::last_lag_ms) over all steps.
    pub mean_lag_ms: f64,
    /// Worst [`last_lag_ms`](Self::last_lag_ms) so far.
    pub max_lag_ms: f64,
    /// Audio received but not yet decoded at the end of the latest
    /// step, in ms. Grows when the machine can't keep up.
    pub backlog_ms: f64,
    /// Utterances finalised.
    pub utterances: u64,
    /// Length of the utterance currently being decoded, in ms.
    pub current_utterance_ms: f64,
    /// Language of the latest utterance: the forced
    /// [`R2t2Config::language`], or the model's own header in auto mode.
    /// The model commits to that header on the utterance's first step,
    /// so an utterance that starts quietly can carry a wrong label (the
    /// transcript itself is unaffected). Force the language when the
    /// label matters.
    pub language: String,
}

impl R2t2Stats {
    fn record_step(&mut self, ms: f64, audio_samples: usize, chunks: usize) {
        self.decode_steps += 1;
        if chunks > 1 {
            self.merged_steps += 1;
        }
        self.last_step_ms = ms;
        self.max_step_ms = self.max_step_ms.max(ms);
        self.decode_ms += ms;
        self.audio_ms += samples_to_ms(audio_samples);
        self.mean_step_ms = self.decode_ms / self.decode_steps as f64;
        self.update_rtf();
    }

    fn update_rtf(&mut self) {
        if self.audio_ms > 0.0 {
            self.rtf = self.decode_ms / self.audio_ms;
        }
    }
}

enum Command {
    Audio(Vec<f32>, Instant),
    Reset,
    Finish,
}

/// Streaming ASR session backed by Confucius4-R2T2.
///
/// Construct with [`R2t2Asr::new`], [`R2t2Asr::with_config`], or
/// [`R2t2Asr::with_model`] to share one set of loaded weights between
/// sessions. Each returns a [`Receiver<TranscriptEvent>`].
///
/// Events: `SpeechStarted` at each VAD onset, `Partial` whenever the
/// committed (append-only) text grows, `Final` with the utterance text
/// at each endpoint, then `SpeechEnded`.
///
/// Like the sherpa-onnx backend, only [`Channel::Local`] and 16 kHz
/// input are accepted for now.
pub struct R2t2Asr {
    tx: Option<Sender<Command>>,
    worker: Option<JoinHandle<()>>,
    stats: Arc<Mutex<R2t2Stats>>,
    failure: Arc<Mutex<Option<String>>>,
    finished: bool,
}

impl R2t2Asr {
    /// Default config: [`R2T2_Q8_0`] weights, 160 ms chunks, automatic
    /// language detection, VAD endpointing.
    pub fn new() -> Result<(Self, Receiver<TranscriptEvent>), AsrError> {
        Self::with_config(R2t2Config::default())
    }

    /// Load (and download, if needed) the weights for `config`, then
    /// start a session.
    pub fn with_config(config: R2t2Config) -> Result<(Self, Receiver<TranscriptEvent>), AsrError> {
        let model = Self::load_model(&config)?;
        Self::with_model(model, config)
    }

    /// Load the weights `config` points at, for sharing between several
    /// sessions via [`with_model`](Self::with_model). Each session still
    /// allocates its own context and audio encoder (~1 GB).
    pub fn load_model(config: &R2t2Config) -> Result<Arc<R2t2Model>, AsrError> {
        config.validate()?;
        let (model_path, mmproj_path) = config.resolve_files()?;
        R2t2Model::load(&EngineConfig {
            model_path,
            mmproj_path,
            use_gpu: config.use_gpu,
            n_threads: config.n_threads,
            n_ctx: config.n_ctx(),
        })
    }

    /// Start a session on already-loaded weights. The model must have
    /// been loaded from a config with the same `max_utterance_ms` or
    /// larger (it sizes the context).
    pub fn with_model(
        model: Arc<R2t2Model>,
        config: R2t2Config,
    ) -> Result<(Self, Receiver<TranscriptEvent>), AsrError> {
        config.validate()?;
        let engine = Engine::new(model)?;
        let vad = match &config.endpointing {
            Some(ep) => Some(
                WebRtcVad::new(SAMPLE_RATE as u32, ep.vad_mode)
                    .map_err(|e| AsrError::Backend(format!("webrtc vad: {e}")))?,
            ),
            None => None,
        };

        let (events_tx, events_rx) = channel();
        let (tx, rx) = channel();
        let stats = Arc::new(Mutex::new(R2t2Stats {
            language: config.language.clone().unwrap_or_default(),
            ..R2t2Stats::default()
        }));
        let failure = Arc::new(Mutex::new(None));

        let mut worker = Worker::new(engine, vad, config, events_tx, stats.clone());
        let failure_w = failure.clone();
        let handle = std::thread::Builder::new()
            .name("wavekat-asr-r2t2".into())
            .spawn(move || {
                if let Err(e) = worker.run(rx) {
                    tracing::error!(error = %e, "r2t2 worker stopped");
                    let _ = worker.events.send(TranscriptEvent::Warning(format!(
                        "r2t2 decoding stopped: {e}"
                    )));
                    *failure_w.lock().unwrap_or_else(|p| p.into_inner()) = Some(e.to_string());
                }
            })?;

        Ok((
            Self {
                tx: Some(tx),
                worker: Some(handle),
                stats,
                failure,
                finished: false,
            },
            events_rx,
        ))
    }

    /// Snapshot of the session's performance counters.
    pub fn stats(&self) -> R2t2Stats {
        self.stats.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    fn send(&self, cmd: Command) -> Result<(), AsrError> {
        let sent = self.tx.as_ref().is_some_and(|tx| tx.send(cmd).is_ok());
        if sent {
            return Ok(());
        }
        let failure = self
            .failure
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        Err(AsrError::Backend(
            failure.unwrap_or_else(|| "r2t2 worker is not running".into()),
        ))
    }
}

impl StreamingAsr for R2t2Asr {
    fn push_audio(&mut self, frame: &AudioFrame, channel: Channel) -> Result<(), AsrError> {
        if self.finished {
            return Err(AsrError::AlreadyFinished);
        }
        if channel != Channel::Local {
            return Err(AsrError::InvalidFrame(
                "r2t2 backend supports Channel::Local only".into(),
            ));
        }
        if frame.sample_rate() != SAMPLE_RATE as u32 {
            return Err(AsrError::InvalidFrame(format!(
                "r2t2 backend requires 16 kHz audio, got {} Hz",
                frame.sample_rate()
            )));
        }
        self.send(Command::Audio(frame.samples().to_vec(), Instant::now()))
    }

    /// Flush: decodes everything still queued and emits the last
    /// `Final`. Blocks until the worker is done.
    fn finish(&mut self) -> Result<(), AsrError> {
        if self.finished {
            return Err(AsrError::AlreadyFinished);
        }
        self.finished = true;
        let sent = self.send(Command::Finish);
        self.tx = None;
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
        sent
    }

    /// Drop the current utterance without emitting a `Final`.
    fn reset(&mut self, channel: Channel) -> Result<(), AsrError> {
        if channel != Channel::Local {
            return Ok(());
        }
        self.send(Command::Reset)
    }
}

impl Drop for R2t2Asr {
    fn drop(&mut self) {
        // Closing the channel stops the worker after its current step.
        self.tx = None;
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }
}

/// WebRTC VAD frame: 30 ms at 16 kHz.
const VAD_FRAME: usize = 480;

enum Phase {
    /// Waiting for speech; keeps a short pre-roll.
    Idle {
        preroll: VecDeque<f32>,
        speech_run: usize,
    },
    /// Inside an utterance.
    Speaking {
        utterance: Box<Utterance>,
        /// Absolute sample index of the utterance's first sample.
        start: u64,
        silence_run: usize,
    },
}

struct Worker {
    engine: Engine,
    vad: Option<WebRtcVad>,
    config: R2t2Config,
    events: Sender<TranscriptEvent>,
    stats: Arc<Mutex<R2t2Stats>>,
    phase: Phase,
    /// Audio waiting to fill a VAD frame.
    frame_buf: Vec<f32>,
    /// Samples routed through the VAD so far.
    routed: u64,
    /// Samples received so far.
    received: u64,
    /// `(cumulative samples received, arrival time)` per pushed frame,
    /// for lag measurement.
    arrivals: VecDeque<(u64, Instant)>,
    finishing: bool,
}

impl Worker {
    fn new(
        engine: Engine,
        vad: Option<WebRtcVad>,
        config: R2t2Config,
        events: Sender<TranscriptEvent>,
        stats: Arc<Mutex<R2t2Stats>>,
    ) -> Self {
        Self {
            engine,
            vad,
            config,
            events,
            stats,
            phase: Phase::Idle {
                preroll: VecDeque::new(),
                speech_run: 0,
            },
            frame_buf: Vec::with_capacity(VAD_FRAME),
            routed: 0,
            received: 0,
            arrivals: VecDeque::new(),
            finishing: false,
        }
    }

    fn run(&mut self, rx: Receiver<Command>) -> Result<(), AsrError> {
        loop {
            // Block for the next command, then take everything else
            // already queued so a slow step catches up in one go.
            let cmd = match rx.recv_timeout(Duration::from_millis(200)) {
                Ok(cmd) => cmd,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => return Ok(()),
            };
            self.handle(cmd)?;
            loop {
                while let Ok(cmd) = rx.try_recv() {
                    self.handle(cmd)?;
                }
                if self.finishing {
                    self.finish()?;
                    return Ok(());
                }
                if !self.step()? {
                    break;
                }
            }
        }
    }

    fn handle(&mut self, cmd: Command) -> Result<(), AsrError> {
        match cmd {
            Command::Audio(samples, at) => {
                self.received += samples.len() as u64;
                self.arrivals.push_back((self.received, at));
                self.frame_buf.extend_from_slice(&samples);
                let mut offset = 0;
                while self.frame_buf.len() - offset >= VAD_FRAME {
                    let frame: Vec<f32> = self.frame_buf[offset..offset + VAD_FRAME].to_vec();
                    offset += VAD_FRAME;
                    self.route(&frame)?;
                    if !self.config.catch_up {
                        // Fixed schedule: decode each chunk as soon as
                        // it is complete, never merging.
                        while self.step()? {}
                    }
                }
                self.frame_buf.drain(..offset);
            }
            Command::Reset => {
                self.phase = Phase::Idle {
                    preroll: VecDeque::new(),
                    speech_run: 0,
                };
                if let Some(vad) = self.vad.as_mut() {
                    vad.reset();
                }
            }
            Command::Finish => self.finishing = true,
        }
        Ok(())
    }

    fn is_speech(&mut self, frame: &[f32]) -> bool {
        let Some(vad) = self.vad.as_mut() else {
            return true;
        };
        let pcm: Vec<i16> = frame
            .iter()
            .map(|s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .collect();
        vad.process(&pcm, SAMPLE_RATE as u32)
            .map(|p| p >= 0.5)
            .unwrap_or(true)
    }

    /// Feed one VAD frame through the speech/silence state machine.
    fn route(&mut self, frame: &[f32]) -> Result<(), AsrError> {
        let speech = self.is_speech(frame);
        self.routed += frame.len() as u64;
        let ep = self.config.endpointing.clone().unwrap_or(Endpointing {
            min_speech_ms: 0,
            preroll_ms: 0,
            ..Endpointing::default()
        });

        match &mut self.phase {
            Phase::Idle {
                preroll,
                speech_run,
            } => {
                preroll.extend(frame);
                let keep = ms_to_samples(ep.preroll_ms).max(frame.len());
                while preroll.len() > keep {
                    preroll.pop_front();
                }
                *speech_run = if speech { *speech_run + frame.len() } else { 0 };
                if *speech_run >= ms_to_samples(ep.min_speech_ms) {
                    let audio: Vec<f32> = preroll.drain(..).collect();
                    let start = self.routed - audio.len() as u64;
                    let mut utterance = Box::new(Utterance::new(
                        self.config.schedule(),
                        &self.config.context,
                        self.config.language.as_deref(),
                    ));
                    utterance.push(&audio);
                    let _ = self.events.send(TranscriptEvent::SpeechStarted {
                        channel: Channel::Local,
                        ts_ms: samples_to_ms_u64(start),
                    });
                    self.phase = Phase::Speaking {
                        utterance,
                        start,
                        silence_run: 0,
                    };
                }
            }
            Phase::Speaking {
                utterance,
                silence_run,
                ..
            } => {
                utterance.push(frame);
                *silence_run = if speech {
                    0
                } else {
                    *silence_run + frame.len()
                };
                let endpoint = self.config.endpointing.is_some()
                    && *silence_run >= ms_to_samples(ep.trailing_silence_ms);
                let too_long =
                    utterance.total_samples() >= ms_to_samples(self.config.max_utterance_ms);
                if endpoint || too_long {
                    return self.end_utterance();
                }
            }
        }
        Ok(())
    }

    /// Run one decode step on the current utterance. Returns `false`
    /// when there's nothing to decode yet.
    fn step(&mut self) -> Result<bool, AsrError> {
        let Phase::Speaking {
            utterance, start, ..
        } = &mut self.phase
        else {
            return Ok(false);
        };
        let t0 = Instant::now();
        let Some(out) = utterance.step(&mut self.engine)? else {
            return Ok(false);
        };
        let elapsed = ms_since(t0);
        let decoded_end = *start + utterance.decoded_samples() as u64;
        let lag = lag_ms(&mut self.arrivals, decoded_end);
        {
            let mut stats = self.stats.lock().unwrap_or_else(|p| p.into_inner());
            stats.record_step(elapsed, out.samples, out.chunks);
            stats.last_lag_ms = lag;
            stats.max_lag_ms = stats.max_lag_ms.max(lag);
            let n = stats.decode_steps as f64;
            stats.mean_lag_ms += (lag - stats.mean_lag_ms) / n;
            stats.backlog_ms = samples_to_ms(utterance.pending_samples());
            stats.current_utterance_ms = samples_to_ms(utterance.decoded_samples());
            if !utterance.language().is_empty() {
                stats.language = utterance.language().to_string();
            }
        }
        if !out.appended.is_empty() {
            let _ = self.events.send(TranscriptEvent::Partial {
                channel: Channel::Local,
                ts_ms: samples_to_ms_u64(decoded_end),
                text: utterance.committed().to_string(),
            });
        }
        Ok(true)
    }

    /// Flush the current utterance and emit its `Final`.
    fn end_utterance(&mut self) -> Result<(), AsrError> {
        let phase = std::mem::replace(
            &mut self.phase,
            Phase::Idle {
                preroll: VecDeque::new(),
                speech_run: 0,
            },
        );
        let Phase::Speaking {
            mut utterance,
            start,
            ..
        } = phase
        else {
            return Ok(());
        };
        let t0 = Instant::now();
        let out = utterance.finish(&mut self.engine)?;
        let elapsed = ms_since(t0);
        let end = start + utterance.decoded_samples() as u64;
        {
            let mut stats = self.stats.lock().unwrap_or_else(|p| p.into_inner());
            // Steps run inside the flush are timed together with it.
            let per_step =
                elapsed / (out.steps.len() + usize::from(out.tail_samples > 0)).max(1) as f64;
            for step in &out.steps {
                stats.record_step(per_step, step.samples, step.chunks);
            }
            if out.tail_samples > 0 {
                stats.decode_ms += per_step;
                stats.audio_ms += samples_to_ms(out.tail_samples);
                stats.update_rtf();
            }
            stats.utterances += 1;
            stats.current_utterance_ms = 0.0;
            if !utterance.language().is_empty() {
                stats.language = utterance.language().to_string();
            }
        }
        if !out.text.is_empty() {
            let _ = self.events.send(TranscriptEvent::Final {
                channel: Channel::Local,
                ts_ms: samples_to_ms_u64(start),
                end_ms: samples_to_ms_u64(end),
                text: out.text,
                confidence: 1.0,
            });
        }
        let _ = self.events.send(TranscriptEvent::SpeechEnded {
            channel: Channel::Local,
            ts_ms: samples_to_ms_u64(end),
        });
        Ok(())
    }

    fn finish(&mut self) -> Result<(), AsrError> {
        let rest = std::mem::take(&mut self.frame_buf);
        if let Phase::Speaking { utterance, .. } = &mut self.phase {
            utterance.push(&rest);
            self.routed += rest.len() as u64;
        }
        self.end_utterance()
    }
}

/// Wall time since the sample at absolute index `sample` arrived,
/// dropping arrival records that are no longer needed.
fn lag_ms(arrivals: &mut VecDeque<(u64, Instant)>, sample: u64) -> f64 {
    while arrivals.len() > 1 && arrivals[0].0 < sample {
        arrivals.pop_front();
    }
    arrivals.front().map(|(_, at)| ms_since(*at)).unwrap_or(0.0)
}

fn ms_to_samples(ms: u32) -> usize {
    ms as usize * SAMPLE_RATE / 1000
}

fn samples_to_ms(samples: usize) -> f64 {
    samples as f64 * 1000.0 / SAMPLE_RATE as f64
}

fn samples_to_ms_u64(samples: u64) -> u64 {
    samples * 1000 / SAMPLE_RATE as u64
}

fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// Convenience for examples and tests: resolve the default weights'
/// cached snapshot directory without downloading. `None` if missing.
pub fn cached_weights_dir(weights: &R2t2Weights) -> Option<PathBuf> {
    if !weights.is_cached() {
        return None;
    }
    download_weights_to_cache(weights, &[], |_| {}).ok()
}

// `StreamingAsr` requires `Send`; keep that a compile-time guarantee.
const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<R2t2Asr>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_pin_their_files() {
        for w in [R2T2_Q8_0, R2T2_Q4_K_M, R2T2_F16] {
            assert_eq!(w.revision.len(), 40);
            for f in w.pinned_files() {
                assert_eq!(f.sha256.len(), 64);
                assert!(f.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
                assert!(f.name.ends_with(".gguf"));
                assert!(f.size > 100_000_000);
            }
            assert_eq!(w.files(), [w.model.name, w.mmproj.name]);
        }
    }

    #[test]
    fn config_validation_and_schedule() {
        let cfg = R2t2Config::default();
        cfg.validate().unwrap();
        let s = cfg.schedule();
        assert_eq!(s.chunk, 2560);
        assert_eq!(s.lookahead, 2560);
        assert_eq!(s.max_chunks_per_step, 12);
        assert!(cfg.n_ctx() >= 2048);

        let bad = R2t2Config {
            chunk_ms: 40,
            ..R2t2Config::default()
        };
        assert!(bad.validate().is_err());

        let exact = R2t2Config {
            catch_up: false,
            ..R2t2Config::default()
        };
        assert_eq!(exact.schedule().max_chunks_per_step, 1);
    }

    #[test]
    fn model_dir_must_contain_the_weights() {
        let cfg = R2t2Config {
            model_dir: Some(std::env::temp_dir().join("wavekat-asr-r2t2-missing")),
            ..R2t2Config::default()
        };
        let err = cfg.resolve_files().unwrap_err();
        assert!(err.to_string().contains("model file not found"), "{err}");
    }
}
