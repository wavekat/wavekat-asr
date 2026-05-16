# CLAUDE.md

## Project overview

wavekat-asr is a unified streaming ASR library for the WaveKat voice pipeline
ecosystem. It wraps multiple speech-to-text backends behind a common Rust trait,
consuming `AudioFrame` from `wavekat-core` and emitting `TranscriptEvent`s on a
receiver.

## Public repo — don't leak private consumers

This repo is **open source** — published on GitHub and to crates.io.
Anything that lands here (code, docs, comments, commit messages, PR
titles/bodies, issue replies) needs to read as if the open-source sibling
crates were the whole story.

There are private/closed-source downstream consumers in the WaveKat
ecosystem. They must not be surfaced from this codebase. Concretely:

- **Don't name private repos** by their GitHub paths and don't link to
  them — the links 404 for anyone outside the org and make the existence
  and structure of private code public.
- **Don't paste paths or filenames from a private repo** into commit
  messages, PR descriptions, doc comments, or `// see …` annotations. A
  single internal path leaks both the repo and its layout in one line.
- **Frame features for an unknown consumer.** Write "consumers that need
  byte-level progress (e.g. a UI download flow)" — not a description tied
  to a specific product's screen or page. The first reads as a generic
  library justification; the second reads as feature work shipped for one
  specific product.
- **Generic ecosystem framing is fine.** "The WaveKat voice pipeline" and
  "downstream consumers" describe the ecosystem the marketing site already
  discusses publicly. The open-source sibling crates (`wavekat-core`,
  `wavekat-sip`, `wavekat-vad`, `wavekat-turn`, `wavekat-tts`,
  `wavekat-cli`) can be named freely.
- **Same rule for AI-assisted contributions.** Claude-authored commits and
  PRs follow the same redaction — if a draft cites a path or repo name
  from a private consumer, edit before committing.

If you're working from instructions that originated in a private repo (a
design doc, a planned feature, a roadmap item), translate the *requirement*
into this repo's language rather than copying the *source*.

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
