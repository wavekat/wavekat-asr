<p align="center">
  <a href="https://github.com/wavekat/wavekat-asr">
    <img src="https://github.com/wavekat/wavekat-brand/raw/main/assets/banners/wavekat-asr-narrow.svg" alt="WaveKat ASR">
  </a>
</p>

[![Crates.io](https://img.shields.io/crates/v/wavekat-asr.svg)](https://crates.io/crates/wavekat-asr)
[![docs.rs](https://docs.rs/wavekat-asr/badge.svg)](https://docs.rs/wavekat-asr)
[![CI](https://github.com/wavekat/wavekat-asr/actions/workflows/ci.yml/badge.svg)](https://github.com/wavekat/wavekat-asr/actions/workflows/ci.yml)

Unified streaming speech-to-text for [WaveKat](https://wavekat.com) voice pipelines, wrapping multiple
ASR engines behind common Rust traits. Same pattern as
[wavekat-vad](https://github.com/wavekat/wavekat-vad),
[wavekat-turn](https://github.com/wavekat/wavekat-turn), and
[wavekat-tts](https://github.com/wavekat/wavekat-tts).

> [!WARNING]
> **Pre-1.0.** The trait surface may iterate as more backends land. Pin
> to an exact patch version.

## Backends

| Backend | Feature flag | Transport | Languages | Status | License |
|---------|-------------|-----------|-----------|--------|---------|
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (streaming Zipformer / Paraformer) | `sherpa-onnx` | Local ONNX | EN, ZH, EN+ZH | ✅ Available | Apache 2.0 |
| [Confucius4-R2T2](https://github.com/netease-youdao/Confucius4-R2T2) (append-only streaming LLM-ASR) | `r2t2` | Local llama.cpp (Metal / CUDA / Vulkan / CPU) | ZH, EN (+ others) | 🧪 Experimental | Code Apache 2.0; weights [NetEase model licence](https://github.com/netease-youdao/Confucius4-R2T2/blob/master/MODEL_LICENSE) |

Local-first by design: both backends run entirely on-device.

## Quick start

```sh
cargo add wavekat-asr --features sherpa-onnx
```

```rust
use wavekat_asr::backends::sherpa_onnx::SherpaOnnxAsr;
use wavekat_asr::{AudioFrame, Channel, StreamingAsr, TranscriptEvent};

let (mut asr, rx) = SherpaOnnxAsr::new()?;  // auto-downloads bilingual model on first run

let samples = vec![0.0f32; 16_000];          // 1 s of 16 kHz mono audio
let frame = AudioFrame::new(samples.as_slice(), 16_000);
asr.push_audio(&frame, Channel::Local)?;
asr.finish()?;

for event in rx.try_iter() {
    if let TranscriptEvent::Final { text, confidence, .. } = event {
        println!("final ({confidence:.2}): {text}");
    }
}
```

## The `StreamingAsr` trait

All backends implement a common trait so you can write code generic over
backends:

```rust
pub trait StreamingAsr: Send {
    fn push_audio(&mut self, frame: &AudioFrame, channel: Channel) -> Result<(), AsrError>;
    fn finish(&mut self) -> Result<(), AsrError>;
    fn reset(&mut self, channel: Channel) -> Result<(), AsrError>;
}
```

Transcript events come back through an `mpsc::Receiver<TranscriptEvent>`
the backend hands you at construction time:

```rust
pub enum TranscriptEvent {
    SpeechStarted { channel, ts_ms },
    SpeechEnded   { channel, ts_ms },
    Partial       { channel, ts_ms, text },
    Final         { channel, ts_ms, end_ms, text, confidence },
    Warning(String),
}
```

`Channel::{Local, Remote}` tags which side of a two-channel call each
event belongs to — the daemon tees both RTP directions through one ASR
instance.

## Architecture

```
wavekat-vad   →  "is someone speaking?"
wavekat-turn  →  "are they done speaking?"
wavekat-asr   →  "what did they say?"
wavekat-tts   →  "synthesize the response"
     │                   │                     │                    │
     └───────────────────┴─────────────────────┴────────────────────┘
                                  │
                            AudioFrame (wavekat-core)
```

The trait surface stays deliberately small. Backends own their own
resampling, network state, and tokenizer.

```text
   AudioFrame ──▶  push_audio(frame, channel)  ──▶  ┌───────────┐
                                                    │  Backend  │
   end of call ─▶  finish()                    ──▶  │           │
                                                    │           │
                                  TranscriptEvent ◀─│           │
                                  on Receiver       └───────────┘
```

Why sync push + receiver, rather than `async fn`? The intended consumer
already runs an event loop and fans events out to clients; matching that
shape avoids forcing a tokio runtime through the trait. Backends that
need their own runtime spawn one internally.

## sherpa-onnx backend

Local streaming Zipformer / Paraformer via
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). Auto-downloads the
selected model from HuggingFace on first use; cached under `$HF_HOME/hub/`
(default `~/.cache/huggingface/hub/`).

Where HuggingFace is unreachable, pre-fetch with
`download_preset_to_cache(&preset, &sources, on_progress)`: it tries each
`DownloadSource` in order (HuggingFace, or any mirror serving the same
`{base}/{repo}/resolve/{revision}/{file}` layout), checks every file against
the SHA-256 pinned in the preset, and writes into the same cache.

### Model presets

Model choice is a construction-time call — the ONNX files load into the
recognizer, so switching models requires rebuilding the backend.

| `WAVEKAT_ASR_PRESET` | Constant | HF repo | Best for |
|----------------------|----------|---------|----------|
| `bilingual` *(default)* | `BILINGUAL_ZH_EN` | `csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20` | Mixed EN+ZH calls |
| `en` | `ZIPFORMER_EN` | `csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26` | English-only |
| `zh` | `PARAFORMER_ZH` | `csukuangfj/sherpa-onnx-streaming-paraformer-zh` | Mandarin-only (often beats bilingual on ZH WER) |
| `paraformer-zh-en` | `PARAFORMER_BILINGUAL_ZH_EN` | `csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en` | ZH-leaning bilingual alternative |

### Examples

Two runnable examples ship behind `--features sherpa-onnx`. First run
auto-downloads the selected model.

```sh
# Transcribe a 16 kHz mono WAV file
cargo run --release --example transcribe_wav --features sherpa-onnx -- audio.wav

# Live mic transcription (Ctrl-C to stop)
cargo run --release --example transcribe_mic --features sherpa-onnx

# Pick a different model (default is `bilingual`)
WAVEKAT_ASR_PRESET=en cargo run --release --example transcribe_mic --features sherpa-onnx
```

## Confucius4-R2T2 backend

[Confucius4-R2T2](https://github.com/netease-youdao/Confucius4-R2T2) is
NetEase Youdao's streaming fine-tune of Qwen3-ASR-1.7B. Its output is
**append-only** — committed text never changes — and it decodes in
chunks of 80 ms to 2 s. This backend runs the official GGUF export
through llama.cpp's multimodal (`mtmd`) API, so the same code runs on
macOS, Linux and Windows:

- **macOS:** Metal, automatically.
- **Linux / Windows:** CPU by default. For a GPU build, enable the
  matching `llama-cpp-2` feature in your own manifest (Cargo unifies it):
  `llama-cpp-2 = { version = "=0.1.158", features = ["cuda"] }` (or `vulkan`).

```rust
use wavekat_asr::backends::r2t2::{R2t2Asr, R2t2Config};

let config = R2t2Config {
    language: Some("Chinese".into()), // None = auto-detect
    chunk_ms: 160,
    ..R2t2Config::default()
};
let (mut asr, rx) = R2t2Asr::with_config(config)?; // downloads ~2.5 GB on first run
// push_audio / finish as with any backend; asr.stats() reports step time, lag and RTF.
```

How it streams: every chunk re-feeds the whole utterance to the model,
prompted with the text committed so far (minus a one-token rollback
window); the continuation is cut at R2T2's `|` stable-prefix marker and
whatever extends the committed text is emitted as a `Partial`. A WebRTC
VAD endpointer closes utterances (`Final`) so step cost stays bounded.
Decoding runs on a worker thread — `push_audio` never blocks — and when a
machine can't keep up, queued chunks merge into one larger step instead
of building a backlog. Details and measurements:
[`docs/05-r2t2-backend.md`](docs/05-r2t2-backend.md).

| Weights constant | Files (from [`netease-youdao/Confucius4-R2T2-GGUF`](https://huggingface.co/netease-youdao/Confucius4-R2T2-GGUF)) | Size |
|------------------|------|------|
| `R2T2_Q8_0` *(default)* | `Confucius4-R2T2-Q8_0.gguf` + `mmproj-Confucius4-R2T2-f16.gguf` | 2.5 GB |
| `R2T2_Q4_K_M` | `Confucius4-R2T2-Q4_K_M.gguf` + `mmproj-Confucius4-R2T2-f16.gguf` | 1.7 GB |
| `R2T2_F16` | `Confucius4-R2T2-f16.gguf` + `mmproj-Confucius4-R2T2-f16.gguf` | 4.1 GB |

```sh
# Live mic, with per-step timing on the status line (Ctrl-C to stop)
cargo run --release --example r2t2_mic --features r2t2

# Stream a WAV at real-time pace and print latency / RTF stats
cargo run --release --example r2t2_wav --features r2t2 -- audio.wav

# Knobs: WAVEKAT_R2T2_WEIGHTS=q8|q4|f16, WAVEKAT_R2T2_LANGUAGE=Chinese,
#        WAVEKAT_R2T2_CHUNK_MS=320, WAVEKAT_R2T2_CONTEXT="hotwords…"
WAVEKAT_R2T2_CHUNK_MS=320 cargo run --release --example r2t2_mic --features r2t2
```

> [!NOTE]
> The R2T2 **weights** are not Apache-2.0: they are governed by the
> NetEase Youdao Model Use License Agreement (royalty-free with
> conditions). Review it before shipping the weights in a product.

## Feature flags

| Flag | Default | Description |
|------|---------|-------------|
| `sherpa-onnx` | No | Local streaming Zipformer / Paraformer via sherpa-onnx; pulls in `hf-hub` for first-run model download |
| `r2t2` | No | Local Confucius4-R2T2 via llama.cpp (`llama-cpp-2` with `mtmd`) plus WebRTC VAD endpointing; downloads pinned, SHA-256-verified GGUF weights on first use |

## Building from source

Enabling `sherpa-onnx` pulls in `sherpa-onnx-sys`, which builds vendored
ONNX Runtime through CMake. You'll need:

- A C++ toolchain (`clang` or `gcc`) and `cmake` on PATH.
- **Linux only — and only for the `transcribe_mic` example:** ALSA dev
  headers (`libasound2-dev` on Debian/Ubuntu, `alsa-lib-devel` on Fedora).
  The library itself has no system audio dependency.

The first build of `sherpa-onnx-sys` is slow (5–10 min); subsequent
builds are cached by Cargo.

Enabling `r2t2` compiles llama.cpp from source (via `llama-cpp-sys-2`),
which needs `cmake`, a C++ toolchain and `libclang` for bindgen
(`libclang-dev` on Debian/Ubuntu; bundled with Xcode on macOS). First
build takes a few minutes.

## Important notes

- **Sample rate.** The `StreamingAsr` trait accepts any `AudioFrame`
  sample rate; backends resample internally. The sherpa-onnx backend
  currently expects 16 kHz f32 input — 8 kHz telephony resampling lands
  in a follow-up (see [`docs/03-sherpa-onnx-backend.md`](docs/03-sherpa-onnx-backend.md)).
- **Dual-channel routing.** `Channel::{Local, Remote}` is wired through
  the trait today; per-channel state isolation in sherpa-onnx is Phase 2.

## About WaveKat

`wavekat-asr` is part of WaveKat, an open-source ecosystem of Rust crates for building real-time voice pipelines. It handles streaming speech-to-text, alongside sibling crates for voice activity detection, turn detection, and text-to-speech.

See [wavekat.com](https://wavekat.com) for the full project.

## Stars

<a href="https://stars.wavekat.com/wavekat/wavekat-asr">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://stars.wavekat.com/wavekat/wavekat-asr/chart.svg?theme=dark">
    <img alt="wavekat/wavekat-asr stars" src="https://stars.wavekat.com/wavekat/wavekat-asr/chart.svg?theme=light">
  </picture>
</a>

## License

Licensed under [Apache 2.0](LICENSE).

Copyright 2026 WaveKat.

### Acknowledgements

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — streaming ASR runtime by the k2-fsa team (Apache 2.0)
- Pretrained model checkpoints from the [sherpa-onnx pretrained zoo](https://huggingface.co/csukuangfj) on HuggingFace
