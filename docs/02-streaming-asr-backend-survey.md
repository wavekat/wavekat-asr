# 02 — Streaming ASR backend survey

**Status:** Selected **sherpa-onnx** as the first local backend runtime,
**bilingual EN+ZH Zipformer** (k2) as the default model, with FunASR
Paraformer-online as a swappable model for ZH-heavy calls. Qwen3-ASR 0.6B
(Alibaba, Jan 2026) and Nemotron Speech Streaming 0.6B (NVIDIA, early 2026)
are tracked as follow-up backends.
**Date:** 2026-05-14

---

## Why this doc

The scaffold in [`01-plan-asr.md`](01-plan-asr.md) intentionally avoided
committing to a backend. We now need one — the consumer is
[`wavekat-voice`](https://github.com/wavekat/wavekat-voice), which wants live
transcription of both call legs (local mic + remote RTP) rendered into the
desktop UI as a phone call unfolds.

This doc captures the candidates that were evaluated, the criteria, and the
choice.

---

## What "live ASR" means here

The first consumer is a softphone. Concretely:

- **Two channels per call.** `Channel::Local` is the user's mic; `Channel::Remote`
  is incoming RTP audio (typically G.711 µ-law / A-law from the SIP peer,
  decoded and resampled by `wavekat-voice`). Both run concurrently.
- **Phone-grade audio.** 8 kHz narrowband from the RTP side, 16 or 48 kHz from
  the mic side. The trait already accepts any [`AudioFrame`] rate; the backend
  resamples internally.
- **Latency target: ≤ 600 ms partial.** Partial transcripts should land in the
  renderer within ~600 ms of the spoken word; finals are allowed to lag further
  (segment boundary + a flush). Anything above ~1 s for partials feels
  laggy in side-by-side conversation UI.
- **Multilingual.** The WaveKat workspace already runs Chinese in TTS
  (`Qwen3-TTS` ships ZH/EN/JA/KO voices). The first ASR backend must at minimum
  cover EN and ZH; a bilingual EN+ZH model is acceptable for v1.
- **CPU-first.** Users running the desktop app are on Macs and PCs without
  guaranteed GPUs. Anything that won't transcribe in real-time on a modern
  consumer CPU is disqualified for the default local backend.
- **No daemons we don't own.** No "spin up a Triton server" requirement. The
  backend has to be a Rust crate that loads from a local model directory.

---

## Criteria

Each candidate was scored against:

1. **Truly streaming** — emits partials within a chunk window, not "wait for
   silence, then run Whisper on the buffer."
2. **Multilingual (at least EN + ZH)** — see above.
3. **CPU real-time** — RTF < 1.0 on a recent x86_64 or Apple Silicon core.
4. **Rust integration story** — published Rust bindings, or a clean ONNX/C-API
   we can wrap behind `ort` the same way `wavekat-tts` wraps Qwen3-TTS.
5. **Model size sensible for desktop ship** — under ~500 MB so we don't
   balloon the installer.
6. **Recently maintained** — checkpoints or releases in 2025–2026, active repo.
7. **Licensing compatible with a closed-source consumer** — Apache-2.0 / MIT
   ideal; CC-BY-NC and "research only" disqualify it for the Voice product.

---

## Candidates evaluated

### 1. sherpa-onnx streaming Zipformer (k2-fsa) — **chosen**

[k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) ships streaming
Zipformer transducer models (encoder / decoder / joiner ONNX triple) plus a
C API and pre-built Rust bindings on crates.io. Pre-trained checkpoints
include English, Chinese, EN+ZH bilingual, Japanese, Korean, and a 2026
multilingual release.

- **Streaming:** ✅ Native — transducer architecture, emits partials per chunk.
- **Multilingual:** ✅ EN + ZH bilingual checkpoint exists, plus dedicated
  language-specific checkpoints.
- **CPU real-time:** ✅ ~100 MB models, RTF well under 1.0 on a single core.
- **Rust:** ✅ [`sherpa-onnx`](https://docs.rs/sherpa-onnx) and
  [`sherpa-transducers`](https://crates.io/crates/sherpa-transducers) crates,
  thin RAII wrappers around the C API. Sample-based push API, per-stream
  state, `is_endpoint()` / `reset()`.
- **Model size:** ✅ ~100 MB per language.
- **Maintenance:** ✅ Active, 2026-dated checkpoints.
- **License:** ✅ Apache-2.0 on the runtime; checkpoints are Apache-2.0 / CC-BY.

**Verdict:** Best dev velocity. Genuine streaming, multilingual, Rust-ready,
small. This is what `0.0.2` (or wherever the version sits at landing time)
ships.

### 2. Qwen3-ASR 0.6B / 1.7B (Alibaba Qwen) — follow-up, ecosystem fit

[`QwenLM/Qwen3-ASR`](https://github.com/QwenLM/Qwen3-ASR) was open-sourced on
**29 January 2026** alongside Qwen3-ForcedAligner. Two sizes (0.6B / 1.7B);
the 1.7B variant claims SOTA among open-source ASR models and is competitive
with proprietary commercial APIs on the team's published benchmarks.

- **Streaming:** ✅ "Streaming/offline unified inference with a single model."
  The repo ships a `demo_streaming.py` and a `vllm_streaming` example.
- **Multilingual:** ✅ 52 languages / dialects, with explicit language ID
  + timestamp prediction.
- **CPU real-time:** ⚠️ Architecturally an LLM-style decoder over audio
  tokens — closer to Qwen3-TTS in inference shape than to a small
  transducer. 0.6B should be runnable on CPU with int4 quant; 1.7B almost
  certainly wants a GPU for real-time.
- **Rust:** ⚠️ No first-party Rust bindings. Path is the same one
  `wavekat-tts` already proved out for Qwen3-TTS: export to ONNX, drive
  with `ort` ourselves, port the tokenizer and embedding lookups in Rust.
  Non-trivial but well-trodden in this workspace.
- **Model size:** ⚠️ 0.6B is ~1.2–1.5 GB ONNX uncompressed; expect
  ~300–400 MB after int4 quantization.
- **Maintenance:** ✅ Active Qwen team, public ModelScope + HuggingFace
  weights.
- **License:** ✅ Apache-2.0.

**Verdict:** Strong candidate as a **second** local backend. Skipped for v1
because: (a) wrapping it is a multi-week effort comparable to the Qwen3-TTS
port in wavekat-tts, (b) CPU real-time on the larger variant is not a
sure thing, and (c) we want a *something works today* path before a
research port. The right shape is: ship sherpa-onnx + Zipformer in 0.0.2,
then add `Qwen3Asr` as a second backend (feature `qwen3-asr`) in a later
release.

This is the natural parallel to how wavekat-tts treats Qwen3-TTS — same
ecosystem, same export pipeline, same `ort` runtime.

### 3. FunASR Paraformer-online / Fun-ASR-Nano (Alibaba/ModelScope) — model choice within sherpa-onnx

[`modelscope/FunASR`](https://github.com/modelscope/FunASR) is the older
Alibaba ASR toolkit. The relevant streaming model is
[`funasr/paraformer-zh-streaming`](https://huggingface.co/funasr/paraformer-zh-streaming)
— a non-autoregressive end-to-end model with configurable chunk windows
(`[0, 10, 5]` = 600 ms, `[0, 8, 4]` = 480 ms). The newer
**Fun-ASR-Nano-2512** (late 2025) extends to 31 languages with real-time
streaming.

The crucial fact: **sherpa-onnx supports Paraformer models directly**,
including online variants. So FunASR isn't a competing *runtime*, it's an
alternative *model* you can drop into the same sherpa-onnx runtime that
hosts Zipformer.

- **Streaming:** ✅ Native chunk-streaming.
- **Multilingual:** Paraformer-zh-streaming: ZH only. Fun-ASR-Nano-2512:
  31 languages, but ONNX export maturity unclear at time of writing.
- **CPU real-time:** ✅
- **Rust:** ✅ Same `sherpa-onnx` crate as the Zipformer path — switching
  models is config, not code.
- **Model size:** ~220 MB for paraformer-zh-streaming.
- **License:** Apache-2.0 toolkit; model weights mostly Apache-2.0 / model-
  specific.

**Verdict:** Useful **alternative model**, not a separate backend.
Specifically: when a user's calls are predominantly Mandarin Chinese,
Paraformer-zh-streaming may outperform the bilingual Zipformer on ZH WER
because it's ZH-specialized. We surface this through `SherpaOnnxConfig`'s
`model_id` field — see [`03`](03-sherpa-onnx-backend.md). No second
backend module needed.

Fun-ASR-Nano-2512 is worth re-evaluating once it has clean ONNX exports
and a checkpoint in the sherpa-onnx pretrained zoo; if it does, it
becomes a candidate to **replace** the bilingual Zipformer as the v1
default.

### 4. NVIDIA Nemotron Speech Streaming 0.6B — follow-up

[`nvidia/nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
is a cache-aware FastConformer-RNNT released early 2026. Quantized to int4
it hits 8.20% average streaming WER across eight English benchmarks at ~0.56 s
algorithmic latency.

- **Streaming:** ✅ Cache-aware encoder, configurable chunk sizes
  (80 / 160 / 560 / 1120 ms).
- **Multilingual:** ❌ English only.
- **CPU real-time:** ✅ Runs faster than real-time on CPU after int4 quant;
  the published recipe is an ONNX Runtime pipeline.
- **Rust:** ⚠️ No published Rust bindings. Path is: export to ONNX via NeMo
  `model.set_export_config({'cache_support': 'True'})`, then drive it with
  `ort` ourselves — same pattern as `wavekat-tts` did for Qwen3-TTS.
- **Model size:** ⚠️ ~600 MB pre-quant, ~150–200 MB int4. Acceptable.
- **Maintenance:** ✅ Released by NVIDIA in early 2026.
- **License:** ✅ NVIDIA OSS, commercially usable.

**Verdict:** Best raw English quality. Worth landing as a second backend once
the sherpa-onnx path is in production and the trait has shaken out. Skipped
for v1 because (a) English-only doesn't satisfy ZH coverage and (b) wrapping
it ourselves is a larger lift than the existing sherpa bindings.

### 5. SenseVoice (Alibaba/ModelScope) — wrong shape

[SenseVoice](https://github.com/FunAudioLLM/SenseVoice) is multilingual,
fast, and the offline numbers are strong (50 languages, including good
ZH/EN/JA/KO). It's also explicitly *non-streaming* — designed for batched
offline transcription, with a chunked-encoder design that doesn't emit
partials.

**Verdict:** Same answer as Whisper. Right tool for a batch / "transcribe
this recording" surface; wrong tool for the live phone-call use case
`wavekat-asr` is shaped for. Skipped.

### 6. Moonshine v2 (Useful Sensors)

Tiny (27 MB) on-device streaming model with an "Ergodic Streaming Encoder."

- **Streaming:** ✅
- **Multilingual:** ❌ English-focused.
- **CPU real-time:** ✅ Extremely fast — Raspberry Pi class.
- **Rust:** ⚠️ No first-party bindings; would have to wrap.
- **License:** ✅ MIT.

**Verdict:** Skipped for v1, same reason as Nemotron — English-only. Keep in
mind for a future "ultralight" feature on memory-constrained devices.

### 7. Whisper family (whisper.cpp, faster-whisper, distil-whisper, whisper-stream-rs)

Whisper is the most familiar option and has well-known multilingual coverage,
but it isn't natively streaming. "Streaming Whisper" wrappers all do the same
trick: maintain a rolling audio buffer and re-run Whisper on it every N
hundred milliseconds. That produces flicker (partials change shape mid-stream)
and burns CPU on every re-run.

- **Streaming:** ❌ (not natively — wrapper hacks only)
- **Multilingual:** ✅ Best in class.
- **CPU real-time:** ⚠️ With distil-large-v3 + faster-whisper, yes, but
  duty-cycle is high under the rolling-buffer pattern.
- **Rust:** ✅ Lots — `whisper.cpp` bindings, `whisper-stream-rs`, plus
  `vox` (Distil-Whisper).
- **License:** ✅ MIT (Whisper), MIT (distil-whisper).

**Verdict:** Skipped for v1 as the *streaming* backend. Whisper is still the
right answer when we eventually want a *batch* / "transcribe this recording"
backend — that's a separate trait surface, not what `wavekat-asr` exposes
today. Revisit if and when we add `BatchAsr`.

### 8. Commercial streaming (Deepgram, AssemblyAI Universal-Streaming, Google Speech-to-Text v2, Azure Speech, Speechmatics, Qwen3-ASR-Flash on Alibaba Cloud)

All offer WebSocket-based streaming with sub-300 ms partial latency, native
multilingual support, and excellent quality. These are the natural commercial
fallback when a user wants better-than-local accuracy or doesn't want to ship
~100 MB of model files.

- **Streaming:** ✅ All of them.
- **Multilingual:** ✅
- **Rust:** ⚠️ No first-party SDKs; we'd hand-roll the WebSocket protocol
  per vendor. Each protocol is similar but not identical.
- **License:** N/A — paid services.

**Verdict:** Out of scope for v1 — local first. See
[`04-commercial-backends.md`](04-commercial-backends.md) for the strategy.

---

## Summary table

| Backend | Streaming | Multilingual | CPU real-time | Rust ready | Size | v1? |
|---------|-----------|--------------|----------------|------------|------|-----|
| **sherpa-onnx + Zipformer (k2)** | ✅ native | ✅ EN+ZH+ | ✅ | ✅ bindings | ~100 MB | **yes (v1 default)** |
| sherpa-onnx + Paraformer-online (FunASR) | ✅ native | ⚠️ ZH-only today | ✅ | ✅ same crate | ~220 MB | **yes (alt model)** |
| Qwen3-ASR 0.6B / 1.7B (Alibaba, Jan 2026) | ✅ unified | ✅ 52 langs | ⚠️ 0.6B only | ⚠️ wrap ONNX (`ort`) | ~300 MB–1.5 GB | follow-up |
| Fun-ASR-Nano-2512 | ✅ | ✅ 31 langs | ✅ | ⚠️ if ONNX export ships | ~?? | re-evaluate |
| Nemotron Speech Streaming | ✅ cache-aware | ❌ EN only | ✅ (int4) | ⚠️ wrap ONNX | ~150–600 MB | follow-up |
| SenseVoice | ❌ offline | ✅ | ✅ | varies | ~?? | wrong surface |
| Moonshine v2 | ✅ | ❌ EN | ✅ | ⚠️ wrap | ~27 MB | future |
| Whisper-family (streaming) | ❌ hack | ✅ | ⚠️ | ✅ | varies | wrong surface |
| Commercial (Deepgram / AssemblyAI / Google / Azure / Speechmatics / Qwen3-ASR-Flash) | ✅ | ✅ | n/a | ⚠️ hand-roll | n/a | see 04 |

---

## Open trait questions, answered for the first backend

[`01-plan-asr.md`](01-plan-asr.md) listed five open questions. Picking
sherpa-onnx lets us commit to answers for v1:

1. **Channel multiplexing** — **one `OnlineStream` per `Channel`.** sherpa-onnx
   exposes `Recognizer::create_stream()` which returns a per-stream state. We
   hold two streams inside the backend (`local`, `remote`), route
   `push_audio` to the matching one, and decode both. Models are shared
   across streams; only stream state duplicates. Memory cost is minimal
   (KV-like caches are small for Zipformer).
2. **Backpressure** — **none yet.** sherpa-onnx accepts samples
   synchronously and decodes in-line. If we later add a network backend,
   that's when backpressure has to surface on the trait. Don't pre-design.
3. **Reset semantics** — **add `reset()` to the trait** before v1 lands.
   sherpa-onnx has cheap per-stream reset; commercial backends will want it
   to start a new utterance without reconnecting. See `03` for the proposed
   signature.
4. **Configuration shape** — **per-backend constructor**, same as
   wavekat-tts. `SherpaOnnxAsr::with_config(SherpaOnnxConfig)`. No shared
   builder trait until at least two real backends exist.
5. **Confidence reporting** — Zipformer transducer doesn't natively report
   per-segment confidence; emit `1.0` as the trait already allows. Revisit
   when a backend (commercial Deepgram does) actually produces it.

---

## Out of scope for v1

- Diarization (speaker labels beyond `Channel::Local` / `Channel::Remote`).
  The two-channel split *is* our diarization for telephony.
- Word-level timestamps. Channel + segment `ts_ms..end_ms` is enough for the
  voice UI. Add later if a backend exposes it cheaply.
- Punctuation / capitalization restoration as a separate stage. Zipformer
  models already emit cased text + punctuation.
- Translation. Different trait surface.

---

## Sources

- [Gladia — Best open-source speech-to-text models in 2026](https://www.gladia.io/blog/best-open-source-speech-to-text-models)
- [Northflank — Best open source STT model in 2026 (with benchmarks)](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks)
- [AssemblyAI — Top 8 open source STT options for voice applications in 2026](https://www.assemblyai.com/blog/top-open-source-stt-options-for-voice-applications)
- [arXiv 2604.14493 — Pushing the Limits of On-Device Streaming ASR](https://arxiv.org/abs/2604.14493)
- [nvidia/nemotron-speech-streaming-en-0.6b (HF)](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
- [Scaling Real-Time Voice Agents with Cache-Aware Streaming ASR (NVIDIA blog on HF)](https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents)
- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [sherpa-onnx Rust docs](https://docs.rs/sherpa-onnx/)
- [sherpa-transducers crate](https://crates.io/crates/sherpa-transducers)
- [sherpa-onnx Paraformer models](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/paraformer-models.html)
- [modelscope/FunASR](https://github.com/modelscope/FunASR)
- [funasr/paraformer-zh-streaming (HF)](https://huggingface.co/funasr/paraformer-zh-streaming)
- [Fun-ASR Technical Report (arXiv 2509.12508)](https://arxiv.org/html/2509.12508v3)
- [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-ASR & Qwen3-ForcedAligner is Now Open Sourced (Qwen blog)](https://qwen.ai/blog?id=qwen3asr)
- [Qwen3-ASR-Flash review (Sider, 2026)](https://sider.ai/blog/ai-tools/qwen3-asr-flash-review-real-time-accuracy-meets-speed-for-2025)
- [Ruoqi Jin — ASR in 2025–2026 deep dive](https://ruoqijin.com/blog/asr-deep-dive-2025-2026)
