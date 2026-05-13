# 01 — Plan: streaming ASR trait

**Status:** Scaffold landed (mock only)
**Date:** 2026-05-14

---

## Goal

Stand up a single-purpose `wavekat-asr` crate that gives downstream
voice pipelines (first consumer: [`wavekat-voice`](https://github.com/wavekat/wavekat-voice))
a stable abstraction over speech-to-text backends, matching the pattern
set by [`wavekat-vad`](https://github.com/wavekat/wavekat-vad) and
[`wavekat-turn`](https://github.com/wavekat/wavekat-turn).

This doc captures what `0.0.1` actually ships and what's left open. It
deliberately does **not** commit to a specific backend roadmap — we'll
add that doc when we pick the first real backend to build.

---

## What 0.0.1 ships

- The `StreamingAsr` trait — `push_audio(&frame, channel)` + `finish()`.
- The `TranscriptEvent` enum — `SpeechStarted`, `SpeechEnded`, `Partial`,
  `Final`, `Warning`.
- The `Channel` enum — `Local` vs `Remote`, so a single ASR instance can
  serve both sides of a call.
- The `AsrError` enum (thiserror).
- The `mock` backend — emits a scripted sequence of events on each
  `push_audio` call. Pairs the impl with a `std::sync::mpsc::Receiver`.
- Workspace + CI + release-plz wired up so subsequent versions cut
  themselves on merge to `main`.

That's everything. No real backends, no network code, no model code, no
resamplers.

---

## Open questions for the first real backend

These need answers before we ship a real backend behind its own feature:

1. **Channel multiplexing.** One session per `Channel` (cleanest,
   doubles cost) or one bidirectional session with multichannel input
   (cheaper, requires careful interleaving)? Depends on what the chosen
   backend supports.
2. **Backpressure.** `push_audio` returns `Ok(())` synchronously today.
   If a backend's transport falls behind, we either need to buffer
   internally (memory growth) or surface a backpressure signal on the
   trait. Watch what real traffic does first.
3. **Reset semantics.** No `reset()` on the trait yet. When a call ends
   and a new one starts, the daemon currently has to drop and recreate
   the `StreamingAsr`. Acceptable for lightweight backends; expensive
   for ones that load big models on construction.
4. **Configuration shape.** Each backend will need its own constructor
   (`MockAsr::new()`, `XxxAsr::new(XxxConfig)`, …). No common builder
   trait yet — resist abstracting until at least two real backends
   exist.
5. **Confidence reporting.** Backends that don't report per-segment
   confidence currently emit `1.0`. Alternative is `Option<f32>` — costs
   ergonomics, gained accuracy. Decide when there's a backend that
   actually has the data.

---

## What this scaffold is **not** trying to do

- Re-implement an ASR model. The point of multiple backends is to wrap
  upstream work, not to publish our own.
- Live up to a stability promise yet. `0.0.x` will iterate fast.
- Define a benchmarking framework. We'll add one (mirroring
  `wavekat-vad`'s benchmark table) once there's something to compare.
- Promise any specific backend on a timeline.

---

## Why a separate repo, not a crate inside `wavekat-voice`

Same reasoning as the existing sibling crates: each WaveKat voice
primitive lives in its own repo, releases on its own cadence via
release-plz, and is consumable by any third party without pulling in our
daemon. The split is deliberate — see the table in
`wavekat-voice/CLAUDE.md`.
