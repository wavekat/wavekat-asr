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
    OnlineModelConfig, OnlineParaformerModelConfig, OnlineRecognizer, OnlineRecognizerConfig,
    OnlineStream, OnlineTransducerModelConfig,
};

use crate::{AsrError, AudioFrame, Channel, StreamingAsr, TranscriptEvent};

const SAMPLE_RATE: i32 = 16_000;

/// Streaming model family.
///
/// Different architectures need different `OnlineModelConfig` slots —
/// transducer uses encoder+decoder+joiner, Paraformer uses just
/// encoder+decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    /// k2 Zipformer transducer (encoder + decoder + joiner).
    Transducer,
    /// Alibaba/FunASR Paraformer (encoder + decoder; no joiner).
    Paraformer,
}

/// A bundled-model preset that fills in `model_id`, model family, and
/// the per-file names for [`SherpaOnnxConfig`].
///
/// Use [`SherpaOnnxConfig::from_preset`] (or [`SherpaOnnxAsr::with_preset`])
/// to construct a backend from one of the constants below.
#[derive(Debug, Clone)]
pub struct ModelPreset {
    /// HuggingFace repo id.
    pub model_id: &'static str,
    /// Architecture family — determines which sherpa-onnx config slot is used.
    pub family: ModelFamily,
    /// Encoder filename inside the repo / model directory.
    pub encoder: &'static str,
    /// Decoder filename inside the repo / model directory.
    pub decoder: &'static str,
    /// Joiner filename. `Some(...)` for [`Transducer`](ModelFamily::Transducer),
    /// `None` for [`Paraformer`](ModelFamily::Paraformer).
    pub joiner: Option<&'static str>,
    /// Tokens filename.
    pub tokens: &'static str,
}

/// Bilingual EN+ZH streaming Zipformer (default). Handles mixed-language
/// speech but can over-produce Hanzi for English-only audio.
pub const BILINGUAL_ZH_EN: ModelPreset = ModelPreset {
    model_id: "csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
    family: ModelFamily::Transducer,
    encoder: "encoder-epoch-99-avg-1.int8.onnx",
    decoder: "decoder-epoch-99-avg-1.onnx",
    joiner: Some("joiner-epoch-99-avg-1.int8.onnx"),
    tokens: "tokens.txt",
};

/// English-only streaming Zipformer. Best WER on English; will hallucinate
/// on Chinese input.
pub const ZIPFORMER_EN: ModelPreset = ModelPreset {
    model_id: "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26",
    family: ModelFamily::Transducer,
    encoder: "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    decoder: "decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
    joiner: Some("joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx"),
    tokens: "tokens.txt",
};

/// Chinese-only streaming Paraformer (FunASR). Often beats the bilingual
/// Zipformer on Mandarin WER; useless for English.
pub const PARAFORMER_ZH: ModelPreset = ModelPreset {
    model_id: "csukuangfj/sherpa-onnx-streaming-paraformer-zh",
    family: ModelFamily::Paraformer,
    encoder: "encoder.int8.onnx",
    decoder: "decoder.int8.onnx",
    joiner: None,
    tokens: "tokens.txt",
};

/// Bilingual EN+ZH streaming Paraformer (FunASR). ZH-leaning bilingual
/// alternative to the default Zipformer.
pub const PARAFORMER_BILINGUAL_ZH_EN: ModelPreset = ModelPreset {
    model_id: "csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en",
    family: ModelFamily::Paraformer,
    encoder: "encoder.int8.onnx",
    decoder: "decoder.int8.onnx",
    joiner: None,
    tokens: "tokens.txt",
};

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
/// Most users want [`SherpaOnnxConfig::from_preset`] to pick one of the
/// bundled [`ModelPreset`] constants. Fields are exposed for downstream
/// crates that want non-bundled checkpoints.
///
/// Resolution order for the model files:
/// 1. `model_dir` if `Some` — files are looked up under it by the
///    `*_filename` fields.
/// 2. `WAVEKAT_ASR_MODEL_DIR` env var — same lookup, useful for tests
///    and power users who don't want a HuggingFace download.
/// 3. HuggingFace Hub — fetched from `model_id` into hf-hub's cache.
#[derive(Debug, Clone)]
pub struct SherpaOnnxConfig {
    /// Directory containing the model files. If `None`, falls back to
    /// the env var, then HuggingFace.
    pub model_dir: Option<PathBuf>,
    /// HuggingFace repo id used when no local directory is found.
    pub model_id: String,
    /// Architecture family — determines which sherpa-onnx config slot is used.
    pub family: ModelFamily,
    /// Encoder filename inside the model directory / HF repo.
    pub encoder_filename: String,
    /// Decoder filename inside the model directory / HF repo.
    pub decoder_filename: String,
    /// Joiner filename. `Some` for transducers, `None` for Paraformer.
    pub joiner_filename: Option<String>,
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

impl SherpaOnnxConfig {
    /// Build a config from one of the bundled [`ModelPreset`] constants.
    /// All other fields are filled with sensible defaults.
    pub fn from_preset(preset: ModelPreset) -> Self {
        Self {
            model_dir: None,
            model_id: preset.model_id.to_string(),
            family: preset.family,
            encoder_filename: preset.encoder.to_string(),
            decoder_filename: preset.decoder.to_string(),
            joiner_filename: preset.joiner.map(str::to_string),
            tokens_filename: preset.tokens.to_string(),
            num_threads: 2,
            decoding_method: DecodingMethod::Greedy,
            enable_endpoint: true,
            rule2_min_trailing_silence: 0.8,
        }
    }
}

impl Default for SherpaOnnxConfig {
    /// Defaults to the bilingual EN+ZH Zipformer ([`BILINGUAL_ZH_EN`]).
    fn default() -> Self {
        Self::from_preset(BILINGUAL_ZH_EN.clone())
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
    /// bilingual EN+ZH Zipformer from HuggingFace if not cached).
    pub fn new() -> Result<(Self, Receiver<TranscriptEvent>), AsrError> {
        Self::with_config(SherpaOnnxConfig::default())
    }

    /// Construct a session from one of the bundled [`ModelPreset`]s.
    pub fn with_preset(preset: ModelPreset) -> Result<(Self, Receiver<TranscriptEvent>), AsrError> {
        Self::with_config(SherpaOnnxConfig::from_preset(preset))
    }

    /// Construct a session with the given config.
    pub fn with_config(
        config: SherpaOnnxConfig,
    ) -> Result<(Self, Receiver<TranscriptEvent>), AsrError> {
        let files = resolve_model_files(&config)?;

        let (transducer, paraformer) = match config.family {
            ModelFamily::Transducer => {
                let joiner = files.joiner.as_ref().ok_or_else(|| {
                    AsrError::Backend("transducer model family requires a joiner file".into())
                })?;
                (
                    OnlineTransducerModelConfig {
                        encoder: Some(path_to_string(&files.encoder)?),
                        decoder: Some(path_to_string(&files.decoder)?),
                        joiner: Some(path_to_string(joiner)?),
                    },
                    OnlineParaformerModelConfig::default(),
                )
            }
            ModelFamily::Paraformer => (
                OnlineTransducerModelConfig::default(),
                OnlineParaformerModelConfig {
                    encoder: Some(path_to_string(&files.encoder)?),
                    decoder: Some(path_to_string(&files.decoder)?),
                },
            ),
        };

        let sys_config = OnlineRecognizerConfig {
            model_config: OnlineModelConfig {
                transducer,
                paraformer,
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
    /// `Some` for transducer family, `None` for Paraformer.
    joiner: Option<PathBuf>,
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
    let joiner = match (&config.joiner_filename, config.family) {
        (Some(name), ModelFamily::Transducer) => Some(resolve(name)?),
        (_, ModelFamily::Paraformer) => None,
        (None, ModelFamily::Transducer) => {
            return Err(AsrError::Backend(
                "transducer family requires joiner_filename".into(),
            ))
        }
    };
    Ok(ModelFiles {
        encoder: resolve(&config.encoder_filename)?,
        decoder: resolve(&config.decoder_filename)?,
        joiner,
        tokens: resolve(&config.tokens_filename)?,
    })
}

fn download_from_hf(config: &SherpaOnnxConfig) -> Result<ModelFiles, AsrError> {
    use hf_hub::api::sync::Api;

    let api = Api::new().map_err(|e| AsrError::Backend(format!("hf-hub init failed: {e}")))?;
    let repo = api.model(config.model_id.clone());
    let fetch = |name: &str| -> Result<PathBuf, AsrError> {
        tracing::debug!(model_id = %config.model_id, file = name, "fetching from HuggingFace");
        repo.get(name)
            .map_err(|e| AsrError::Backend(format!("hf-hub download of {name} failed: {e}")))
    };
    let joiner = match (&config.joiner_filename, config.family) {
        (Some(name), ModelFamily::Transducer) => Some(fetch(name)?),
        (_, ModelFamily::Paraformer) => None,
        (None, ModelFamily::Transducer) => {
            return Err(AsrError::Backend(
                "transducer family requires joiner_filename".into(),
            ))
        }
    };
    Ok(ModelFiles {
        encoder: fetch(&config.encoder_filename)?,
        decoder: fetch(&config.decoder_filename)?,
        joiner,
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
    fn default_config_uses_bilingual_zipformer() {
        let cfg = SherpaOnnxConfig::default();
        assert_eq!(cfg.model_id, BILINGUAL_ZH_EN.model_id);
        assert_eq!(cfg.family, ModelFamily::Transducer);
        assert_eq!(cfg.encoder_filename, BILINGUAL_ZH_EN.encoder);
        assert_eq!(
            cfg.joiner_filename.as_deref(),
            Some(BILINGUAL_ZH_EN.joiner.unwrap())
        );
        assert!(cfg.enable_endpoint);
    }

    #[test]
    fn paraformer_preset_has_no_joiner() {
        let cfg = SherpaOnnxConfig::from_preset(PARAFORMER_ZH);
        assert_eq!(cfg.family, ModelFamily::Paraformer);
        assert!(cfg.joiner_filename.is_none());
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
