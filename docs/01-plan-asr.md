# 01 — Streaming ASR trait: design notes

Historical planning note from the scaffold landing on 2026-05-14. Kept for
context on why the trait surface looks the way it does. For the live
trait reference, read the rustdoc on
[`StreamingAsr`](https://docs.rs/wavekat-asr/latest/wavekat_asr/trait.StreamingAsr.html).

---

## What this crate is

A single-purpose Rust crate exposing one trait, `StreamingAsr`, and a
small event vocabulary (`TranscriptEvent`, `Channel`, `AsrError`).
Backends live behind Cargo features so dependants pull in only what they
ship. Same pattern as
[`wavekat-vad`](https://github.com/wavekat/wavekat-vad) and
[`wavekat-turn`](https://github.com/wavekat/wavekat-turn).

## What this crate is not

- Not an ASR model. Backends wrap upstream work; we don't publish our
  own checkpoints.
- Not a stability promise yet. `0.0.x` will iterate as more backends
  land — pin to an exact patch version.
- Not a benchmarking framework. One will follow once there's more than
  one real backend to compare.

## Why a separate repo

Each WaveKat voice primitive lives in its own repo, releases on its own
cadence via release-plz, and is consumable by any third party without
pulling in a larger daemon.

## Open questions left to future backends

The current trait was shaped against the sherpa-onnx backend
([`docs/03`](03-sherpa-onnx-backend.md)). Real commercial backends
([`docs/04`](04-commercial-backends.md)) will pressure-test it on:

1. **Channel multiplexing.** One session per `Channel`, or one
   bidirectional session with multichannel input? Depends on backend
   support.
2. **Backpressure.** `push_audio` returns `Ok(())` synchronously today.
   Network backends that fall behind need either internal buffering or
   a backpressure signal on the trait.
3. **Configuration shape.** Each backend has its own constructor; no
   common builder trait yet. Resist abstracting until at least two real
   network backends exist.
4. **Confidence reporting.** Backends without per-segment confidence
   emit `1.0`. `Option<f32>` is the alternative — decide when a backend
   that actually reports it lands.
