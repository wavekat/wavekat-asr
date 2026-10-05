# 05 — Confucius4-R2T2 backend

**Status:** Experimental (behind the `r2t2` feature)
**Date:** 2026-10-06

---

## Overview

`R2t2Asr` runs NetEase Youdao's
[Confucius4-R2T2](https://github.com/netease-youdao/Confucius4-R2T2), a
streaming fine-tune of Qwen3-ASR-1.7B, behind the `StreamingAsr` trait.
R2T2's selling point is **append-only output**: it learns to emit only a
*stable prefix* of the transcript and to wait for more audio otherwise,
so committed text never needs revising. Upstream reports 200–600 ms
average latency at accuracy close to offline Qwen3-ASR, with decode
chunks configurable from 80 ms to 2 s.

## Which weights

Upstream publishes several formats; only one fits a portable Rust crate:

| Repo | Runtime | Platforms | Used here |
|------|---------|-----------|-----------|
| `netease-youdao/Confucius4-R2T2` (safetensors, 4.1 GB) | vLLM / PyTorch | NVIDIA (Linux) | No |
| **`netease-youdao/Confucius4-R2T2-GGUF`** | **llama.cpp (`mtmd`)** | **macOS (Metal), Linux, Windows (CPU / CUDA / Vulkan)** | **Yes** |
| Community MLX conversions | MLX | Apple Silicon only | No |
| Community ONNX export | onnxruntime / WebGPU | Broad, but unofficial | No |

The GGUF repo ships the language model at f16 / Q8_0 / Q4_K_M and the
audio projector (`mmproj`, the audio encoder) at f16 / Q8_0. Presets pin
the language model at a given quantisation and always use the **f16
projector**: upstream notes that llama.cpp's encoder is the less robust
half on quiet onsets, so it isn't quantised further. Files are pinned to
a HuggingFace revision with SHA-256s and fetched through the existing
`download_pinned_to_cache` path (mirrors work the same way as for
sherpa-onnx).

## Runtime: llama.cpp via `llama-cpp-2`

[`llama-cpp-2`](https://crates.io/crates/llama-cpp-2) 0.1.158 bundles a
llama.cpp with the Qwen3-ASR audio encoder (`mtmd/models/qwen3a.cpp`) and
exposes the `mtmd` API. It's pinned exactly because `mtmd`'s C API moves
between llama.cpp releases. Default features are off (no OpenMP, so no
`libomp` on macOS); Metal is on by llama.cpp's own default on Apple
targets. GPU backends elsewhere are opt-in through Cargo feature
unification in the consumer's manifest, which keeps `--all-features`
builds of this crate (CI) free of CUDA / Vulkan SDK requirements.

Per decode the engine follows upstream's `r2t2_llama/native_ext.cpp`:
clear the KV cache → replace `<|audio_start|><|audio_pad|><|audio_end|>`
with mtmd's media marker → `mtmd_tokenize` with the PCM as an audio
bitmap → `mtmd_helper_eval_chunks` (runs the encoder and prefill) →
greedy decode to end-of-generation or the token budget.

`R2t2Model` (the loaded language model) is `Arc`-shareable across
sessions; each session owns its own llama context and `mtmd` context
because neither is safe to use from two threads at once.

## The streaming loop

`stream.rs` is a port of `R2T2ASRModel.streaming_transcribe` /
`finish_streaming_transcribe` and the per-chunk scheduling in upstream's
`ws_server.py`, kept independent of llama.cpp (a `Decoder` trait) so it
can be unit-tested with a scripted fake:

1. Wait for `chunk + lookahead` (160 + 160 ms) before the first decode,
   then decode every `chunk`.
2. Each step re-feeds **all** utterance audio, prompted with the
   previous hypothesis minus its last `unfixed_token_num` (1) tokens.
   The rollback widens while the cut lands inside a multi-byte
   character.
3. The continuation is cut at the `|` stable-prefix marker, normalised
   (punctuation by script, no spaces between Hanzi, repetition
   clean-up), rolled back again, and whatever extends the committed text
   is committed.
4. Token budget per step follows the reference server's adaptive rule
   (1 token per 80 ms of audio, +0.5 while stalled on English, ×2 after
   a Chinese word, capped).

One deliberate deviation: in auto-language mode, a `language None`
header with no text (the model's answer for near-silent audio) is not
carried into the next prompt. Upstream does carry it, which pins the
whole utterance to "no language" when it starts with a short silence.

## Segmentation and threading

Step cost grows with utterance length (the encoder and prefill rerun on
all of it), so the backend segments the stream:

- **WebRTC VAD** (`wavekat-vad`, 30 ms frames): 90 ms of speech opens an
  utterance (with 300 ms pre-roll), 800 ms of trailing silence closes it
  with a `Final`. Silence costs no inference at all.
- **Hard cap** at `max_utterance_ms` (30 s default).

Inference runs on a worker thread, so `push_audio` only enqueues. With
`catch_up` (default) the worker merges all complete queued chunks (up to
2 s) into one step when it falls behind, rather than queueing work it
can't finish; R2T2 is trained for chunks up to 2 s, so merged steps stay
in-distribution. `catch_up: false` reproduces the reference's fixed
schedule exactly, for offline evaluation.

`R2t2Asr::stats()` reports step time (mean / max), real-time factor,
lag (time from a sample's arrival in `push_audio` to the step that
decoded it finishing) and current backlog.

## Measurements

Apple M4 (10-core GPU, 16 GB), Metal, real-time-paced streaming of
upstream's `resources/test.wav` (6.7 s Mandarin) via the `r2t2_wav`
example. Every configuration produced the same transcript,
`之前有顾客自己带酒水也没加收钱或者不让喝。`

| Weights | Chunk | Steps (merged) | Mean step | RTF | Mean lag | Max lag |
|---------|-------|----------------|-----------|-----|----------|---------|
| Q8_0 | 160 ms | 30 (10) | 223 ms | 1.04* | 319 ms | 405 ms |
| Q8_0 | 320 ms | 20 (0) | 251 ms | 0.79 | 264 ms | 346 ms |
| Q8_0 | 480 ms | 13 (0) | 257 ms | 0.54 | 278 ms | 321 ms |
| Q4_K_M | 160 ms | 31 (9) | 213 ms | 1.02* | 316 ms | 388 ms |
| Q4_K_M | 320 ms | 20 (0) | 235 ms | 0.74 | 248 ms | 305 ms |
| Q4_K_M | 480 ms | 13 (0) | 238 ms | 0.50 | 259 ms | 308 ms |

\* At 160 ms the M4 can't decode faster than real time, so catch-up
merges chunks and the worker is busy continuously (RTF ≈ 1 by
construction); lag stays bounded rather than growing.

Where a step's time goes (Q8_0, 160 ms schedule): encoder + prefill
grows from ~110 ms at 0.3 s of audio to ~190 ms at 6.7 s; each generated
token costs ~20 ms. Quantising the language model further barely helps
because the step is dominated by re-encoding and prefilling the audio.

Lag here is pipeline lag only. The model's own stable-prefix behaviour
adds its look-ahead on top (it waits until a word is unambiguous), which
is the latency upstream's 200–600 ms figure refers to.

Synthetic English and Mandarin clips (macOS `say`) transcribed exactly,
including punctuation, at ~300 ms mean lag on the default settings.

## Limitations and follow-ups

- `Channel::Local` and 16 kHz input only, as with sherpa-onnx today.
- The auto-detected language label is decided on an utterance's first
  step and can be wrong when the utterance starts quietly; transcripts
  are unaffected. Force `language` when the label matters.
- Performance ideas not yet tried: caching encoder output for completed
  8 s encoder windows, reusing the KV cache for the system-prompt prefix,
  and running the encoder on a different device than the decoder.
- The CPU-only path works but has not been benchmarked here.
