# 03 — sherpa-onnx backend implementation plan

**Status:** Proposed
**Date:** 2026-05-14
**Depends on:** [`02-streaming-asr-backend-survey.md`](02-streaming-asr-backend-survey.md)

---

## Overview

Implement `SherpaOnnxAsr` as the first real backend in `wavekat-asr`, wrapping
the [`sherpa-onnx`](https://crates.io/crates/sherpa-onnx) Rust crate
(currently `1.13.2`) around a streaming Zipformer transducer. The backend
holds one shared `OnlineRecognizer` and two per-channel `OnlineStream`
instances, so a single call's local + remote audio can be transcribed by one
backend instance.

This is the analogue of
[`wavekat-tts/docs/03-qwen3-tts-backend.md`](../../wavekat-tts/docs/03-qwen3-tts-backend.md)
— sibling repos use the same doc shape.

---

## Why sherpa-onnx (not raw `ort`)

`wavekat-tts` chose to wrap Qwen3-TTS ONNX models directly with the `ort`
crate, because the upstream codebase didn't ship a streaming inference
pipeline — every layer (tokenizer, talker LM, code predictor, vocoder) had
to be glued together in Rust.

Streaming ASR is the opposite. `sherpa-onnx` already ships a
production-quality streaming pipeline (feature extractor, encoder/decoder/
joiner stepping, endpointer, RNN-T greedy/beam decoder, hotword biasing) with
a C API and Rust bindings. Re-implementing that on top of `ort` would buy us
nothing for v1 and cost weeks.

Trade-off: we pull in sherpa-onnx's native libraries (vendored ONNX Runtime)
instead of going through `ort`. That means our execution-provider story is
*sherpa-onnx's* providers (CPU, CUDA, CoreML, DirectML) rather than `ort`'s.
For v1 that's fine. If we later need to share an `ort` session across crates,
the natural second backend is **Qwen3-ASR** (`feature = "qwen3-asr"`) wrapped
the same way `wavekat-tts` wraps Qwen3-TTS — see
[`02 §2`](02-streaming-asr-backend-survey.md#2-qwen3-asr-06b--17b-alibaba-qwen--follow-up-ecosystem-fit).
Nemotron Speech Streaming is the other follow-up candidate for English-only
SOTA.

---

## Trait changes landing alongside this backend

The trait in `01-plan-asr.md` listed `reset()` as an open question. To match
how real backends (sherpa-onnx, future commercial) want to be used, we add
one method to `StreamingAsr`:

```rust
pub trait StreamingAsr: Send {
    fn push_audio(&mut self, frame: &AudioFrame, channel: Channel) -> Result<(), AsrError>;
    fn finish(&mut self) -> Result<(), AsrError>;

    /// Reset per-channel utterance state. Cheap on local backends; for
    /// commercial backends, implementations may choose to drop and recreate
    /// their socket — the trait only promises "the next push_audio starts a
    /// fresh utterance on `channel`."
    fn reset(&mut self, channel: Channel) -> Result<(), AsrError>;
}
```

`MockAsr` gains a no-op `reset` returning `Ok(())`. This is a breaking
change to the trait, but the crate is `0.0.x` and the only known consumer
(`wavekat-voice`) hasn't shipped against it yet — acceptable.

---

## Cargo wiring

```toml
# crates/wavekat-asr/Cargo.toml additions

[features]
default = []
mock = []                                          # unchanged
sherpa-onnx = ["dep:sherpa-onnx", "dep:hf-hub"]    # new

# Execution providers — composable, same shape as wavekat-tts
cuda    = ["sherpa-onnx?/cuda"]
coreml  = ["sherpa-onnx?/coreml"]
directml = ["sherpa-onnx?/directml"]

[dependencies]
# existing
wavekat-core = "0.0.7"
thiserror = "2"
tracing = "0.1"

# new — local-backend
sherpa-onnx = { version = "1.13", optional = true, default-features = false, features = ["sys"] }
hf-hub      = { version = "0.5", optional = true, default-features = false, features = ["ureq"] }
```

`hf-hub` is the same dependency `wavekat-tts` already uses for model
downloads — we follow the same convention. The `sys` feature on `sherpa-onnx`
pulls in the vendored native library; we deliberately avoid the `download`
features it ships with because we want our own model-management story.

---

## Model directory convention

```
~/.cache/wavekat/asr/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/
├── encoder.onnx                                   # ~50 MB int8
├── decoder.onnx                                   # ~12 MB
├── joiner.onnx                                    # ~10 MB
└── tokens.txt
```

Resolution order, identical to `wavekat-tts` Qwen3 backend:

1. Explicit path in `SherpaOnnxConfig`.
2. `WAVEKAT_ASR_MODEL_DIR` env var (for tests / power users).
3. `$XDG_CACHE_HOME/wavekat/asr/<model-id>/` (or platform equivalent).
4. Download from HuggingFace into (3) if missing.

Default model for v1: `sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20`
— the current best-supported EN+ZH bilingual streaming Zipformer in the
sherpa-onnx pretrained model zoo. We ship pointer constants for the other
useful checkpoints too; selecting them is a `SherpaOnnxConfig` field:

| Constant | Underlying model | When to pick it |
|----------|------------------|------------------|
| `SHERPA_BILINGUAL_ZH_EN` (default) | `sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20` (Zipformer, k2) | mixed-language calls, default |
| `SHERPA_PARAFORMER_ZH` | `sherpa-onnx-streaming-paraformer-zh-2024-03-28` (FunASR Paraformer-online, Alibaba) | predominantly Mandarin calls; ZH-specialized, may beat the bilingual model on ZH WER |
| `SHERPA_ZIPFORMER_EN` | `sherpa-onnx-streaming-zipformer-en-2023-06-26` | English-only calls |

The Paraformer choice is what makes this backend more than "just Zipformer":
sherpa-onnx supports both architectures behind the same `OnlineRecognizer`
type ([Paraformer model zoo](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/paraformer-models.html)),
and FunASR is the Alibaba ASR family that parallels the Qwen3-TTS family
already in wavekat-tts. Switching models is config, not code — see
[`02 §3`](02-streaming-asr-backend-survey.md#3-funasr-paraformer-online--fun-asr-nano-alibabamodelscope--model-choice-within-sherpa-onnx).

---

## Public API

```rust
// src/backends/sherpa_onnx.rs
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver};

use crate::{AsrError, AudioFrame, Channel, StreamingAsr, TranscriptEvent};

/// Configuration for the sherpa-onnx streaming Zipformer backend.
#[derive(Debug, Clone)]
pub struct SherpaOnnxConfig {
    /// Path to the model directory (encoder/decoder/joiner/tokens). If
    /// `None`, falls back to `WAVEKAT_ASR_MODEL_DIR`, then cache, then
    /// downloads `model_id`.
    pub model_dir: Option<PathBuf>,
    /// HuggingFace model id used when no local directory is found.
    /// Defaults to a curated EN+ZH bilingual streaming Zipformer.
    pub model_id: String,
    /// Number of threads for ONNX Runtime.
    pub num_threads: usize,
    /// Decoding method: `"greedy_search"` (fastest) or `"modified_beam_search"`.
    pub decoding_method: DecodingMethod,
    /// Endpointing — chunk silence detection (sherpa's own rules; not VAD).
    pub enable_endpoint: bool,
    /// Minimum trailing silence in seconds before declaring an endpoint.
    pub rule2_min_trailing_silence: f32,
    /// Hardware provider.
    pub provider: Provider,
}

#[derive(Debug, Clone, Copy)]
pub enum DecodingMethod { Greedy, ModifiedBeamSearch }

#[derive(Debug, Clone, Copy)]
pub enum Provider { Cpu, Cuda, CoreMl, DirectMl }

pub struct SherpaOnnxAsr {
    // Private fields — see "Internal structure" below.
}

impl SherpaOnnxAsr {
    /// Construct a backend with default config (auto-download EN+ZH model).
    pub fn new() -> Result<(Self, Receiver<TranscriptEvent>), AsrError>;

    /// Construct with explicit config.
    pub fn with_config(config: SherpaOnnxConfig) -> Result<(Self, Receiver<TranscriptEvent>), AsrError>;
}

impl StreamingAsr for SherpaOnnxAsr {
    fn push_audio(&mut self, frame: &AudioFrame, channel: Channel) -> Result<(), AsrError>;
    fn finish(&mut self) -> Result<(), AsrError>;
    fn reset(&mut self, channel: Channel) -> Result<(), AsrError>;
}
```

The `(Self, Receiver<TranscriptEvent>)` shape mirrors `MockAsr::new()`.

---

## Internal structure

```rust
struct SherpaOnnxAsr {
    recognizer: sherpa_onnx::OnlineRecognizer,    // shared by both channels
    local: ChannelState,
    remote: ChannelState,
    tx: std::sync::mpsc::Sender<TranscriptEvent>,
    finished: bool,
}

struct ChannelState {
    stream: sherpa_onnx::OnlineStream,
    /// The last full transcript string we emitted for this channel; used
    /// to deduplicate partials.
    last_emitted: String,
    /// Stream wall-clock origin in ms. Set on first audio push.
    stream_start_ms: Option<u64>,
    /// Cumulative samples received on this channel, in the model's sample
    /// rate (post-resample) — used to derive `ts_ms`.
    samples_pushed_at_16k: u64,
}
```

### One recognizer, two streams — why

sherpa-onnx's design intentionally separates "model" (`OnlineRecognizer`)
from "per-utterance state" (`OnlineStream`). The recognizer holds the ONNX
sessions and tokens; each stream holds its own feature extractor, decoder
state, and KV-like caches. Sharing the recognizer between channels is the
documented way to multiplex.

Result: the cost of supporting a second channel is "one extra
`OnlineStream`" (low MB), not "double the ONNX session memory" (hundreds of
MB). Answers open question #1 from `01-plan-asr.md`.

---

## Audio path

`AudioFrame` carries its native sample rate (telephony: 8 kHz, mic: 16 or
48 kHz). sherpa-onnx Zipformer models expect 16 kHz mono f32 in `[-1, 1]`.
We resample inside `push_audio`:

```rust
fn push_audio(&mut self, frame: &AudioFrame, channel: Channel) -> Result<(), AsrError> {
    let resampled = if frame.sample_rate() == 16_000 {
        std::borrow::Cow::Borrowed(frame.as_f32())
    } else {
        std::borrow::Cow::Owned(frame.resample(16_000).into_f32_owned())
    };

    let state = self.channel_mut(channel);
    state.stream.accept_waveform(16_000, &resampled);

    while self.recognizer.is_ready(&state.stream) {
        self.recognizer.decode(&mut state.stream);
    }

    let result = self.recognizer.get_result(&state.stream);
    if !result.text.is_empty() && result.text != state.last_emitted {
        // Partial event with delta — see "Event emission" below.
        emit_partial(&self.tx, channel, &state, &result.text)?;
        state.last_emitted = result.text;
    }

    if self.recognizer.is_endpoint(&state.stream) {
        emit_final(&self.tx, channel, &state, &result.text)?;
        self.recognizer.reset(&mut state.stream);
        state.last_emitted.clear();
    }

    Ok(())
}
```

`wavekat-core`'s `AudioFrame` already exposes `resample()` behind the
`resample` feature — same dependency wavekat-tts uses. No new resampler
needed.

### Event emission rules

| sherpa-onnx state | event we emit |
|-------------------|----------------|
| `get_result(...)` returns non-empty text that differs from `last_emitted` | `Partial { channel, ts_ms, text }` |
| `is_endpoint(...)` is true | `Final { channel, ts_ms, end_ms, text, confidence: 1.0 }`, then call `recognizer.reset(&stream)` and clear `last_emitted` |
| `finish()` is called | flush any non-empty pending result as `Final`, then `SpeechEnded` per channel |
| internal error decoding | propagate `AsrError::Backend(...)` from `push_audio` |

`SpeechStarted` / `SpeechEnded` are optional in the trait; we emit
`SpeechEnded` only on `finish()`. `SpeechStarted` is skipped because
sherpa-onnx's endpointer doesn't fire a "speech began" signal that lines up
with `Partial` arrival anyway.

### Timestamps

`ts_ms` for `Partial` and `Final` is computed from cumulative samples
received on the channel since construction, at 16 kHz post-resample:

```rust
fn ts_ms(state: &ChannelState) -> u64 {
    state.samples_pushed_at_16k * 1000 / 16_000
}
```

This is monotone per channel. `end_ms` on a `Final` is the timestamp at the
moment of endpoint.

---

## Configuration defaults

```rust
impl Default for SherpaOnnxConfig {
    fn default() -> Self {
        Self {
            model_dir: None,
            model_id: "csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20".into(),
            num_threads: 2,
            decoding_method: DecodingMethod::Greedy,
            enable_endpoint: true,
            rule2_min_trailing_silence: 0.8,
            provider: Provider::Cpu,
        }
    }
}
```

Rationale:

- `num_threads = 2`: enough for real-time on a single channel; we have two
  channels, so 4 cores worth of CPU under load. Configurable upward.
- `greedy`: lowest latency. Beam search costs ~30% more CPU; not needed for
  v1.
- `enable_endpoint = true`, `rule2_min_trailing_silence = 0.8`: phone calls
  have natural pauses; we want a `Final` per utterance, not per call.
- `provider = Cpu`: safe default. CoreML is opt-in via the `coreml` feature
  + `Provider::CoreMl`.

---

## Phased rollout

### Phase 1 — single-channel, default model, greedy

Goal: push 16 kHz audio from a WAV file into one stream, get partials and a
final out. Add an `examples/transcribe_wav.rs` that mirrors
`wavekat-tts/examples/synthesize.rs` in shape.

Acceptance:
- `cargo test -p wavekat-asr --features sherpa-onnx` passes.
- The example transcribes a known short WAV and prints partials + final.

### Phase 2 — dual channel + resampling

Goal: route `Channel::Local` and `Channel::Remote` to separate streams.
Accept 8 kHz input from the remote leg and resample internally.

Acceptance:
- A test that pushes two interleaved 8 kHz / 16 kHz frames and asserts
  events arrive on the correct channel.

### Phase 3 — endpoint + reset

Goal: drive endpointing rules from a real conversation recording, validate
`Final` boundaries. Implement `StreamingAsr::reset()`.

Acceptance:
- A 30 s sample with three utterances produces three `Final`s on the right
  channel with monotone timestamps.

### Phase 4 — execution providers

Goal: gate CoreML / CUDA / DirectML behind features the same way
`wavekat-tts` does. CoreML first because most of the team is on macOS.

Acceptance:
- `cargo test -p wavekat-asr --features sherpa-onnx,coreml` builds on
  macOS.

### Phase 5 — model download via hf-hub

Goal: if no local model dir is found, download the default model from HF
into the cache directory. Reuse the same pattern as wavekat-tts.

Acceptance:
- First-run with no `WAVEKAT_ASR_MODEL_DIR` set downloads the model and
  uses it; second run skips the download.

---

## Testing strategy

### Unit tests (no model files)

- `SherpaOnnxConfig::default()` returns expected defaults.
- Sample-rate resampling glue routes correctly between 8 kHz / 16 kHz / 48 kHz.
- `reset(Channel::Local)` clears `last_emitted` for local only, not remote.

### Integration tests (gated on `SHERPA_ONNX_MODEL_DIR` or default cache)

```rust
#[test]
fn transcribe_known_phrase() {
    let model_dir = std::env::var("SHERPA_ONNX_MODEL_DIR")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(default_cache_dir);
    if !model_dir.exists() {
        eprintln!("Skipping: model dir not found at {model_dir:?}");
        return;
    }
    // load WAV → push frames → assert final transcript contains "hello"
}
```

Same gate-on-existence pattern as `wavekat-tts/docs/03`.

---

## Open questions deferred

- **Hotword biasing.** sherpa-onnx supports per-stream hotwords for
  domain-specific vocabulary (contact names, technical terms). Expose this
  through `SherpaOnnxConfig` once we have a real consumer asking for it.
- **Custom endpointer rules.** sherpa's `rule1`/`rule2`/`rule3` thresholds
  could move out of the config into a builder once tuned.
- **Multi-language switching mid-call.** The bilingual model handles
  EN+ZH out of the box; switching to a third language mid-call would
  require swapping the recognizer, which we don't support and probably
  shouldn't.
- **Word-level timestamps.** sherpa-onnx exposes per-token timestamps on
  the result; we drop them today. Add to `TranscriptEvent::Final` only
  when the renderer wants to use them.

---

## References

- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [sherpa-onnx Rust crate docs](https://docs.rs/sherpa-onnx/)
- [Streaming Zipformer pre-trained model zoo](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html)
- [aivo0/rust-asr-server](https://github.com/aivo0/rust-asr-server) — reference Rust integration
- Sibling pattern: [`wavekat-tts/docs/03-qwen3-tts-backend.md`](../../wavekat-tts/docs/03-qwen3-tts-backend.md)
- Sibling pattern: [`wavekat-tts/docs/02-execution-providers.md`](../../wavekat-tts/docs/02-execution-providers.md)
