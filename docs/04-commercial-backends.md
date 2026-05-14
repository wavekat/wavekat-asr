# 04 — Commercial backends strategy

**Status:** Sketch — local-first; commercial backends are post-v1.
**Date:** 2026-05-14

---

## Why this doc exists now

The `wavekat-asr` mandate from the consumer ([`wavekat-voice`](https://github.com/wavekat/wavekat-voice))
is "support local model **or** commercial models behind one interface."
[`03`](03-sherpa-onnx-backend.md) commits us to a local backend first; this
doc records the constraints commercial backends will place on the trait so
we don't paint the local backend into a corner.

We do **not** implement any commercial backend in v1. We just make sure the
trait survives one.

---

## Target vendors

The streaming-ASR vendor field in 2026 is small and converged. Any
production backend list will look approximately like:

| Vendor | Transport | Partial latency | Multilingual | Notes |
|--------|-----------|------------------|--------------|-------|
| **Deepgram** | WebSocket (Nova-3+) | ~150–300 ms | yes (100+) | Most common in voice-agent stacks; cheap; per-word confidence; native interim results |
| **AssemblyAI Universal-Streaming** | WebSocket | ~200 ms | yes | Word-level timestamps, formatted finals |
| **Google Speech-to-Text v2** | gRPC streaming | ~300 ms | yes | Heavy SDK; auth via service account; not WebSocket |
| **Azure Speech** | WebSocket / SDK | ~300 ms | yes | Microsoft-flavored auth; SDK-heavy |
| **Speechmatics** | WebSocket | ~250 ms | yes | Strong on accents; enterprise-flavored pricing |
| **OpenAI gpt-4o-transcribe streaming** | WebSocket / Realtime API | ~400 ms | yes | Newer; bundled with Realtime voice; pricing favors voice-agent users |

We don't need to implement all of these. We pick **one** when there's a
paying user reason to. Deepgram is the obvious first target — cheapest,
most-used in voice-agent stacks, WebSocket-clean — but this doc is vendor-
agnostic.

---

## Where commercial backends bend the trait

The local sherpa-onnx backend gets away with a simple sync trait because it
decodes on the same thread that pushed audio. Network backends don't have
that luxury. The places the trait creaks:

### 1. Async transport, sync trait

A WebSocket client wants tokio. `push_audio` is sync. The backend's
constructor spawns its own tokio runtime (same trick `wavekat-tts` uses
for backends that need async networking), and `push_audio` hands frames to
a tokio `mpsc::Sender` that the runtime drains. The receiver of the trait
already lives outside the backend — vendor responses come back to the same
`mpsc::Sender<TranscriptEvent>` that local backends use.

No trait change needed. Document the pattern in the backend module.

### 2. Reconnection on transport drop

Sockets drop. The behavior contract: a commercial backend MAY emit a
`TranscriptEvent::Warning("reconnecting: ...")` and recover transparently.
It MUST NOT silently lose audio. If reconnect ultimately fails, the next
`push_audio` returns `AsrError::Backend(...)`.

This is implementation discipline, not a trait change.

### 3. Backpressure

Real network backends fall behind under load. We have two options:

**Option A — internal bounded queue.** The backend keeps an
`mpsc::Sender<Frame>` with a bound (say 5 s of audio). When it fills,
`push_audio` returns `AsrError::Backpressure`. Consumer's choice what to do
(drop, log, panic).

**Option B — internal unbounded queue.** Buffer until OOM. Bad.

Pick **A** when the first commercial backend lands; add an
`AsrError::Backpressure` variant at that time. The trait surface itself
doesn't need to change — `push_audio` already returns `Result`.

### 4. Vendor authentication

Each vendor authenticates differently (Deepgram API key in URL, Google
service-account JWT, Azure subscription key). All of these are
**construction-time** concerns — they live in the per-vendor `Config`
struct, not on the trait.

```rust
pub struct DeepgramConfig {
    pub api_key: SecretString,
    pub model: String,           // e.g. "nova-3"
    pub language: Option<String>,
    pub interim_results: bool,
    pub punctuate: bool,
    pub endpointing_ms: u32,
    pub keyterms: Vec<String>,   // boosted vocabulary
}
```

Same pattern wavekat-tts already uses. No common builder trait.

### 5. Single bidirectional session vs two sessions

The local backend (sherpa-onnx) is fine running two streams over one
recognizer, because compute is local and CPU is the cost. For commercial
vendors, every concurrent session is metered. Two sessions per call doubles
the bill.

Most vendors support **multichannel input** in a single WebSocket — you
send stereo audio with `Channel::Local` on the left and `Channel::Remote`
on the right, and the response carries a `channel` field. That's the
cheaper option, and answers open question #1 differently per backend:

| Backend | Multiplexing |
|---------|--------------|
| sherpa-onnx (local) | one recognizer, two streams — see [`03`](03-sherpa-onnx-backend.md) |
| Deepgram | one WebSocket, stereo input with `channels=2` |
| AssemblyAI | one WebSocket per channel today (no native multichannel) — pay 2× |
| Google v2 | one StreamingRecognize per channel today |

This is fine — the trait surface (`push_audio(frame, channel)`) doesn't
care which one the backend chose internally. Document it per backend.

### 6. Confidence reporting actually matters here

Deepgram emits per-word `confidence`. AssemblyAI emits per-segment
`confidence`. The local Zipformer emits nothing.

Today the trait says backends without confidence emit `1.0`. Once we have
a commercial backend producing real numbers, downstream UIs may want to
distinguish "known-confident" from "we have no idea." Two paths:

**A.** Keep `confidence: f32`, with `1.0` as the no-data sentinel.
Documented; consumers that care can compare against backend metadata.

**B.** Change to `confidence: Option<f32>`. Cleaner semantically; tiny
ergonomic cost on consumers.

**Defer the decision.** Don't change anything in v1. Revisit when the first
commercial backend lands and we have UI code that actually wants to render
confidence shading.

---

## Trait additions that survive both worlds

From the analysis above, what we believe v1 needs (already in [`03`](03-sherpa-onnx-backend.md)):

```rust
pub trait StreamingAsr: Send {
    fn push_audio(&mut self, frame: &AudioFrame, channel: Channel) -> Result<(), AsrError>;
    fn finish(&mut self) -> Result<(), AsrError>;
    fn reset(&mut self, channel: Channel) -> Result<(), AsrError>;
}
```

What we expect to add *only* when a commercial backend lands:

- `AsrError::Backpressure` variant.
- A `TranscriptEvent::Warning` use convention for transient transport
  blips (already exists; just spec it in the backend doc).
- Possibly `confidence: Option<f32>` if we go path (B) above — breaking,
  but `0.0.x` allows it.

What we deliberately are **not** adding:

- A `Backend` trait that abstracts "construct any vendor from any config."
  Real configs diverge too much; the wavekat-tts experience shows this
  costs more than it pays.
- Per-backend metric hooks. Wait for the daemon to ask for them.
- Async variants. The whole point of the sync trait is consumer
  ergonomics; commercial backends spawn their own runtimes.

---

## When we'd actually build one

Reasonable triggers for landing a commercial backend:

1. **A paying user requests it.** Best signal.
2. **WER on the bilingual local model is unacceptable for a real call.**
   Plausible for noisy phone audio.
3. **A language the local Zipformer doesn't cover well** (Hindi, Arabic,
   etc.) becomes important.

Until one of those, sherpa-onnx is the whole story.

---

## References

- [Deepgram streaming API](https://developers.deepgram.com/docs/live-streaming-audio)
- [AssemblyAI Universal-Streaming](https://www.assemblyai.com/docs/speech-to-text/streaming)
- [Google Cloud Speech-to-Text v2](https://cloud.google.com/speech-to-text/v2/docs/streaming-recognize)
- [Azure Speech real-time STT](https://learn.microsoft.com/azure/ai-services/speech-service/get-started-speech-to-text)
- [Speechmatics Real-Time SaaS](https://docs.speechmatics.com/rt-api-ref)
