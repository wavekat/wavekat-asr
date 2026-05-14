//! sherpa-onnx streaming ASR backend.
//!
//! Wraps a [`sherpa_onnx::OnlineRecognizer`] around a streaming Zipformer
//! transducer. The default configuration loads the bilingual EN+ZH
//! checkpoint from the sherpa-onnx pretrained zoo; if the model files
//! aren't already on disk they're downloaded from HuggingFace on first
//! use.
//!
//! This is Phase 1 of the rollout in `docs/03-sherpa-onnx-backend.md`:
//! single-channel only ([`Channel::Local`]), 16 kHz f32 input only.
//! Dual-channel routing and 8 kHz → 16 kHz resampling land in Phase 2.
//!
//! # Example
//!
//! ```no_run
//! use wavekat_asr::backends::sherpa_onnx::SherpaOnnxAsr;
//! use wavekat_asr::{AudioFrame, Channel, StreamingAsr, TranscriptEvent};
//!
//! let (mut asr, rx) = SherpaOnnxAsr::new().unwrap();
//! let samples = vec![0.0f32; 16_000];
//! let frame = AudioFrame::new(samples.as_slice(), 16_000);
//! asr.push_audio(&frame, Channel::Local).unwrap();
//! asr.finish().unwrap();
//! for event in rx.try_iter() {
//!     if let TranscriptEvent::Final { text, .. } = event {
//!         println!("final: {text}");
//!     }
//! }
//! ```

use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};

use sherpa_onnx::{
    OnlineModelConfig, OnlineRecognizer, OnlineRecognizerConfig, OnlineStream,
    OnlineTransducerModelConfig,
};

use crate::{AsrError, AudioFrame, Channel, StreamingAsr, TranscriptEvent};

const SAMPLE_RATE: i32 = 16_000;

/// Default HuggingFace repo id for the v1 bilingual EN+ZH streaming Zipformer.
pub const DEFAULT_MODEL_ID: &str =
    "csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20";

/// Default filename for the encoder ONNX (int8 quantized).
pub const DEFAULT_ENCODER: &str = "encoder-epoch-99-avg-1.int8.onnx";
/// Default filename for the decoder ONNX.
pub const DEFAULT_DECODER: &str = "decoder-epoch-99-avg-1.onnx";
/// Default filename for the joiner ONNX (int8 quantized).
pub const DEFAULT_JOINER: &str = "joiner-epoch-99-avg-1.int8.onnx";
/// Default filename for the tokens table.
pub const DEFAULT_TOKENS: &str = "tokens.txt";

/// Decoding method passed through to sherpa-onnx.
#[derive(Debug, Clone, Copy)]
pub enum DecodingMethod {
    /// Greedy search — lowest latency, default.
    Greedy,
    /// Modified beam search — slightly better quality, ~30% more CPU.
    ModifiedBeamSearch,
}

impl DecodingMethod {
    fn as_str(self) -> &'static str {
        match self {
            DecodingMethod::Greedy => "greedy_search",
            DecodingMethod::ModifiedBeamSearch => "modified_beam_search",
        }
    }
}

/// Configuration for [`SherpaOnnxAsr`].
///
/// Resolution order for the model files:
/// 1. `model_dir` if `Some` — files are looked up under it by the
///    `*_filename` fields.
/// 2. `WAVEKAT_ASR_MODEL_DIR` env var — same lookup, useful for tests
///    and power users who don't want a HuggingFace download.
/// 3. HuggingFace Hub — fetched from `model_id` into hf-hub's cache.
#[derive(Debug, Clone)]
pub struct SherpaOnnxConfig {
    /// Directory containing the four model files. If `None`, falls back
    /// to the env var, then HuggingFace.
    pub model_dir: Option<PathBuf>,
    /// HuggingFace repo id used when no local directory is found.
    pub model_id: String,
    /// Encoder filename inside the model directory / HF repo.
    pub encoder_filename: String,
    /// Decoder filename inside the model directory / HF repo.
    pub decoder_filename: String,
    /// Joiner filename inside the model directory / HF repo.
    pub joiner_filename: String,
    /// Tokens filename inside the model directory / HF repo.
    pub tokens_filename: String,
    /// Number of threads for ONNX Runtime.
    pub num_threads: i32,
    /// Decoding strategy.
    pub decoding_method: DecodingMethod,
    /// Whether to enable sherpa-onnx's built-in endpointer.
    pub enable_endpoint: bool,
    /// Trailing silence (seconds) for endpoint rule 2.
    pub rule2_min_trailing_silence: f32,
}

impl Default for SherpaOnnxConfig {
    fn default() -> Self {
        Self {
            model_dir: None,
            model_id: DEFAULT_MODEL_ID.to_string(),
            encoder_filename: DEFAULT_ENCODER.to_string(),
            decoder_filename: DEFAULT_DECODER.to_string(),
            joiner_filename: DEFAULT_JOINER.to_string(),
            tokens_filename: DEFAULT_TOKENS.to_string(),
            num_threads: 2,
            decoding_method: DecodingMethod::Greedy,
            enable_endpoint: true,
            rule2_min_trailing_silence: 0.8,
        }
    }
}

/// Streaming ASR session backed by sherpa-onnx.
///
/// Construct via [`SherpaOnnxAsr::new`] or [`SherpaOnnxAsr::with_config`].
/// Both return a paired [`Receiver<TranscriptEvent>`] alongside the
/// session, matching the shape of the mock backend.
///
/// # Phase 1 limitations
///
/// - Only `Channel::Local` is supported. Pushing `Channel::Remote`
///   returns [`AsrError::InvalidFrame`].
/// - Only 16 kHz frames are accepted. Resampling lands in Phase 2.
pub struct SherpaOnnxAsr {
    recognizer: OnlineRecognizer,
    stream: OnlineStream,
    tx: Sender<TranscriptEvent>,
    last_emitted: String,
    samples_pushed: u64,
    last_utt_start_ms: u64,
    finished: bool,
}

impl SherpaOnnxAsr {
    /// Construct a session with default config (auto-downloads the
    /// bilingual EN+ZH model from HuggingFace if not cached).
    pub fn new() -> Result<(Self, Receiver<TranscriptEvent>), AsrError> {
        Self::with_config(SherpaOnnxConfig::default())
    }

    /// Construct a session with the given config.
    pub fn with_config(
        config: SherpaOnnxConfig,
    ) -> Result<(Self, Receiver<TranscriptEvent>), AsrError> {
        let files = resolve_model_files(&config)?;

        let sys_config = OnlineRecognizerConfig {
            model_config: OnlineModelConfig {
                transducer: OnlineTransducerModelConfig {
                    encoder: Some(path_to_string(&files.encoder)?),
                    decoder: Some(path_to_string(&files.decoder)?),
                    joiner: Some(path_to_string(&files.joiner)?),
                },
                tokens: Some(path_to_string(&files.tokens)?),
                num_threads: config.num_threads.max(1),
                provider: Some("cpu".to_string()),
                ..Default::default()
            },
            decoding_method: Some(config.decoding_method.as_str().to_string()),
            enable_endpoint: config.enable_endpoint,
            rule1_min_trailing_silence: 2.4,
            rule2_min_trailing_silence: config.rule2_min_trailing_silence,
            rule3_min_utterance_length: 300.0,
            ..Default::default()
        };

        let recognizer = OnlineRecognizer::create(&sys_config)
            .ok_or_else(|| AsrError::Backend("OnlineRecognizer::create returned null".into()))?;
        let stream = recognizer.create_stream();

        let (tx, rx) = channel();

        Ok((
            Self {
                recognizer,
                stream,
                tx,
                last_emitted: String::new(),
                samples_pushed: 0,
                last_utt_start_ms: 0,
                finished: false,
            },
            rx,
        ))
    }

    fn current_ms(&self) -> u64 {
        self.samples_pushed * 1000 / SAMPLE_RATE as u64
    }

    fn pump(&mut self) -> Result<(), AsrError> {
        while self.recognizer.is_ready(&self.stream) {
            self.recognizer.decode(&self.stream);
        }
        if let Some(result) = self.recognizer.get_result(&self.stream) {
            if !result.text.is_empty() && result.text != self.last_emitted {
                let _ = self.tx.send(TranscriptEvent::Partial {
                    channel: Channel::Local,
                    ts_ms: self.current_ms(),
                    text: result.text.clone(),
                });
                self.last_emitted = result.text;
            }
        }
        if self.recognizer.is_endpoint(&self.stream) {
            let end_ms = self.current_ms();
            if !self.last_emitted.is_empty() {
                let _ = self.tx.send(TranscriptEvent::Final {
                    channel: Channel::Local,
                    ts_ms: self.last_utt_start_ms,
                    end_ms,
                    text: std::mem::take(&mut self.last_emitted),
                    confidence: 1.0,
                });
            }
            self.recognizer.reset(&self.stream);
            self.last_utt_start_ms = end_ms;
        }
        Ok(())
    }
}

impl StreamingAsr for SherpaOnnxAsr {
    fn push_audio(&mut self, frame: &AudioFrame, channel: Channel) -> Result<(), AsrError> {
        if self.finished {
            return Err(AsrError::AlreadyFinished);
        }
        if channel != Channel::Local {
            return Err(AsrError::InvalidFrame(
                "sherpa-onnx Phase 1 backend supports Channel::Local only".into(),
            ));
        }
        if frame.sample_rate() != SAMPLE_RATE as u32 {
            return Err(AsrError::InvalidFrame(format!(
                "sherpa-onnx Phase 1 backend requires 16 kHz audio, got {} Hz",
                frame.sample_rate()
            )));
        }

        let samples = frame.samples();
        self.stream.accept_waveform(SAMPLE_RATE, samples);
        self.samples_pushed += samples.len() as u64;
        self.pump()
    }

    fn finish(&mut self) -> Result<(), AsrError> {
        if self.finished {
            return Err(AsrError::AlreadyFinished);
        }
        self.finished = true;
        self.stream.input_finished();
        self.pump()?;
        // Flush any pending hypothesis that endpointing didn't promote.
        if !self.last_emitted.is_empty() {
            let _ = self.tx.send(TranscriptEvent::Final {
                channel: Channel::Local,
                ts_ms: self.last_utt_start_ms,
                end_ms: self.current_ms(),
                text: std::mem::take(&mut self.last_emitted),
                confidence: 1.0,
            });
        }
        let _ = self.tx.send(TranscriptEvent::SpeechEnded {
            channel: Channel::Local,
            ts_ms: self.current_ms(),
        });
        Ok(())
    }

    fn reset(&mut self, channel: Channel) -> Result<(), AsrError> {
        if channel != Channel::Local {
            return Ok(());
        }
        self.recognizer.reset(&self.stream);
        self.last_emitted.clear();
        self.last_utt_start_ms = self.current_ms();
        Ok(())
    }
}

#[derive(Debug)]
struct ModelFiles {
    encoder: PathBuf,
    decoder: PathBuf,
    joiner: PathBuf,
    tokens: PathBuf,
}

fn resolve_model_files(config: &SherpaOnnxConfig) -> Result<ModelFiles, AsrError> {
    if let Some(dir) = config.model_dir.as_deref() {
        return load_from_dir(dir, config);
    }
    if let Ok(env_dir) = std::env::var("WAVEKAT_ASR_MODEL_DIR") {
        return load_from_dir(Path::new(&env_dir), config);
    }
    download_from_hf(config)
}

fn load_from_dir(dir: &Path, config: &SherpaOnnxConfig) -> Result<ModelFiles, AsrError> {
    let resolve = |filename: &str| -> Result<PathBuf, AsrError> {
        let path = dir.join(filename);
        if !path.exists() {
            return Err(AsrError::Backend(format!(
                "model file not found: {}",
                path.display()
            )));
        }
        Ok(path)
    };
    Ok(ModelFiles {
        encoder: resolve(&config.encoder_filename)?,
        decoder: resolve(&config.decoder_filename)?,
        joiner: resolve(&config.joiner_filename)?,
        tokens: resolve(&config.tokens_filename)?,
    })
}

fn download_from_hf(config: &SherpaOnnxConfig) -> Result<ModelFiles, AsrError> {
    use hf_hub::api::sync::Api;

    let api = Api::new()
        .map_err(|e| AsrError::Backend(format!("hf-hub init failed: {e}")))?;
    let repo = api.model(config.model_id.clone());
    let fetch = |name: &str| -> Result<PathBuf, AsrError> {
        tracing::debug!(model_id = %config.model_id, file = name, "fetching from HuggingFace");
        repo.get(name)
            .map_err(|e| AsrError::Backend(format!("hf-hub download of {name} failed: {e}")))
    };
    Ok(ModelFiles {
        encoder: fetch(&config.encoder_filename)?,
        decoder: fetch(&config.decoder_filename)?,
        joiner: fetch(&config.joiner_filename)?,
        tokens: fetch(&config.tokens_filename)?,
    })
}

fn path_to_string(path: &Path) -> Result<String, AsrError> {
    path.to_str()
        .map(|s| s.to_string())
        .ok_or_else(|| AsrError::Backend(format!("non-UTF-8 path: {}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_uses_bilingual_model() {
        let cfg = SherpaOnnxConfig::default();
        assert_eq!(cfg.model_id, DEFAULT_MODEL_ID);
        assert_eq!(cfg.encoder_filename, DEFAULT_ENCODER);
        assert!(cfg.enable_endpoint);
    }

    #[test]
    fn load_from_dir_errors_on_missing_files() {
        let cfg = SherpaOnnxConfig::default();
        let tmp = std::env::temp_dir().join("wavekat-asr-missing");
        let err = load_from_dir(&tmp, &cfg).unwrap_err();
        match err {
            AsrError::Backend(msg) => assert!(msg.contains("model file not found")),
            other => panic!("expected Backend error, got {other:?}"),
        }
    }
}
