<p align="center">
  <a href="https://github.com/wavekat/wavekat-asr">
    <img src="https://github.com/wavekat/wavekat-brand/raw/main/assets/banners/wavekat-asr-narrow.svg" alt="WaveKat ASR">
  </a>
</p>

[![Crates.io](https://img.shields.io/crates/v/wavekat-asr.svg)](https://crates.io/crates/wavekat-asr)
[![docs.rs](https://docs.rs/wavekat-asr/badge.svg)](https://docs.rs/wavekat-asr)
[![CI](https://github.com/wavekat/wavekat-asr/actions/workflows/ci.yml/badge.svg)](https://github.com/wavekat/wavekat-asr/actions/workflows/ci.yml)

Unified streaming speech-to-text for voice pipelines, wrapping multiple
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

Local-first by design: the bundled sherpa-onnx backend ships today and
runs entirely on-device.

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

## Feature flags

| Flag | Default | Description |
|------|---------|-------------|
| `sherpa-onnx` | No | Local streaming Zipformer / Paraformer via sherpa-onnx; pulls in `hf-hub` for first-run model download |

## Important notes

- **Sample rate.** The `StreamingAsr` trait accepts any `AudioFrame`
  sample rate; backends resample internally. The sherpa-onnx backend
  currently expects 16 kHz f32 input — 8 kHz telephony resampling lands
  in a follow-up (see [`docs/03-sherpa-onnx-backend.md`](docs/03-sherpa-onnx-backend.md)).
- **Dual-channel routing.** `Channel::{Local, Remote}` is wired through
  the trait today; per-channel state isolation in sherpa-onnx is Phase 2.

## License

Licensed under [Apache 2.0](LICENSE).

Copyright 2026 WaveKat.

### Acknowledgements

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — streaming ASR runtime by the k2-fsa team (Apache 2.0)
- Pretrained model checkpoints from the [sherpa-onnx pretrained zoo](https://huggingface.co/csukuangfj) on HuggingFace
