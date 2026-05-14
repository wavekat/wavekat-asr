# CLAUDE.md

## Project overview

wavekat-asr is a unified streaming ASR library for the WaveKat voice pipeline
ecosystem. It wraps multiple speech-to-text backends behind a common Rust trait,
consuming `AudioFrame` from `wavekat-core` and emitting `TranscriptEvent`s on a
receiver.

## Build & test

```bash
make check          # cargo check, all features
make test           # cargo test --features sherpa-onnx
make lint           # clippy with -D warnings
make ci             # everything CI runs
```

On Linux, the `transcribe_mic` example needs ALSA dev headers (`libasound2-dev`
on Debian/Ubuntu). The library itself does not.

## Architecture

- `src/lib.rs` — `StreamingAsr` trait, `TranscriptEvent`, `Channel`
- `src/error.rs` — `AsrError`
- `src/backends/` — one module per backend, gated by feature flags
- Backends return events on `std::sync::mpsc::Receiver<TranscriptEvent>` handed
  out at construction time, not via async fn

## Key design decisions

1. **Sync push + receiver** — `push_audio` is sync; events flow back on an
   mpsc receiver. Avoids forcing a tokio runtime through the trait. Matches
   the consumption shape of the downstream voice daemon.
2. **AudioFrame is the only audio type** — consumed from `wavekat-core`, same
   type produced by wavekat-vad/turn/tts.
3. **Per-channel state** — `Channel::{Local, Remote}` tags each frame and
   event so one backend instance can serve both legs of a call.
4. **Backends own their resampling/network/tokenizer** — trait surface stays
   tiny on purpose. Phase 1 sherpa-onnx requires 16 kHz; 8 kHz telephony
   resampling lands in Phase 2.
5. **Feature flag per backend** — same pattern as wavekat-vad/turn/tts.
