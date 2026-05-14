<p align="center">
  <a href="https://github.com/wavekat/wavekat-asr">
    <img src="https://github.com/wavekat/wavekat-brand/raw/main/assets/banners/wavekat-asr-narrow.svg" alt="WaveKat ASR">
  </a>
</p>

[![Crates.io](https://img.shields.io/crates/v/wavekat-asr.svg)](https://crates.io/crates/wavekat-asr)
[![docs.rs](https://docs.rs/wavekat-asr/badge.svg)](https://docs.rs/wavekat-asr)
[![CI](https://github.com/wavekat/wavekat-asr/actions/workflows/ci.yml/badge.svg)](https://github.com/wavekat/wavekat-asr/actions/workflows/ci.yml)

Streaming ASR trait surface for voice pipelines, intended to wrap one or
more speech-to-text backends behind a common Rust API. Same pattern as
[wavekat-vad](https://github.com/wavekat/wavekat-vad) and
[wavekat-turn](https://github.com/wavekat/wavekat-turn).

> [!WARNING]
> **Scaffold release.** This crate ships only the trait shape and a
> scripted-event `mock` backend so downstream consumers can wire
> integration tests against the contract. No real ASR backends are
> bundled yet — the trait may iterate before the first one lands. Pin to
> an exact patch version.

## What's included

| Item | Feature flag |
|------|--------------|
| `StreamingAsr` trait, `TranscriptEvent`, `Channel`, `AsrError` | always |
| `MockAsr` — scripted partials → final, paired with an `mpsc::Receiver` | `mock` |
| `SherpaOnnxAsr` — local streaming Zipformer (EN+ZH bilingual by default); auto-downloads model from HuggingFace on first use | `sherpa-onnx` |

## Quick start

```sh
cargo add wavekat-asr --features mock
```

```rust
use wavekat_asr::{AudioFrame, Channel, StreamingAsr, TranscriptEvent};
use wavekat_asr::backends::mock::MockAsr;

let (mut asr, rx) = MockAsr::new();
let samples = vec![0i16; 160];
let frame = AudioFrame::new(&samples, 16_000);

asr.push_audio(&frame, Channel::Local).unwrap();
asr.finish().unwrap();

for event in rx.try_iter() {
    match event {
        TranscriptEvent::Final { text, .. } => println!("final: {text}"),
        TranscriptEvent::Partial { text, .. } => println!("partial: {text}"),
        _ => {}
    }
}
```

## Examples

Two runnable examples ship behind `--features sherpa-onnx`. First run
auto-downloads the selected model into hf-hub's cache.

```sh
# Transcribe a 16 kHz mono WAV file
cargo run --release --example transcribe_wav --features sherpa-onnx -- audio.wav

# Live mic transcription (Ctrl-C to stop)
cargo run --release --example transcribe_mic --features sherpa-onnx

# Pick a different model (default is `bilingual`)
WAVEKAT_ASR_PRESET=en cargo run --release --example transcribe_mic --features sherpa-onnx
```

Bundled model presets — model choice is a construction-time call (the
ONNX files load into the recognizer); switching models requires
rebuilding the backend.

| `WAVEKAT_ASR_PRESET` | Constant | HF repo | Best for |
|----------------------|----------|---------|----------|
| `bilingual` *(default)* | `BILINGUAL_ZH_EN` | `csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20` | Mixed EN+ZH calls |
| `en` | `ZIPFORMER_EN` | `csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26` | English-only |
| `zh` | `PARAFORMER_ZH` | `csukuangfj/sherpa-onnx-streaming-paraformer-zh` | Mandarin-only (often beats bilingual on ZH WER) |
| `paraformer-zh-en` | `PARAFORMER_BILINGUAL_ZH_EN` | `csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en` | ZH-leaning bilingual alternative |

## Architecture

The crate exposes one trait — `StreamingAsr` — and one event enum —
`TranscriptEvent`. The trait keeps the surface that consumers see as
small as possible; backends will own their own resampling, network
state, and tokenizer.

```text
   AudioFrame ──▶  push_audio(frame, channel)  ──▶  ┌───────────┐
                                                    │  Backend  │
   end of call ─▶  finish()                    ──▶  │           │
                                                    │           │
                                  TranscriptEvent ◀─│           │
                                  on Receiver       └───────────┘
```

Why a sync push + receiver pair, rather than `async fn`? The daemon that
will consume this (`wavekat-voice`) already runs an event loop and fans
events out over SSE; matching that shape avoids forcing a tokio runtime
through the trait. Backends that need their own runtime will spawn one
internally.

## License

Apache-2.0. See [LICENSE](LICENSE).
