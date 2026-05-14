# 05 — wavekat-voice integration

**Status:** Sketch — what `wavekat-voice` has to add to consume this crate.
**Date:** 2026-05-14
**Audience:** people working in `wavekat-voice`, not this crate. Lives here
because the integration contract is shaped by what `wavekat-asr` exposes.

---

## Goal

When a SIP call is connected, transcribe both legs live and render the
transcript in the desktop UI as the call unfolds. User picks up the phone →
each spoken sentence (from either side) appears in the transcript pane
within ~600 ms of being said, with stable finals replacing flickery partials
when the speaker pauses.

---

## Data flow

```
                            wavekat-voice (Rust daemon)
                            ┌──────────────────────────────────────────────┐
   mic capture ──f32───────►│  ┌─────────┐                                 │
                            │  │  mixer  ├──┐                              │
   RTP recv ───G.711 i16───►│  │  /tap   │  │                              │
                            │  └─────────┘  │                              │
                            │   per call    │                              │
                            │               ▼                              │
                            │      ┌─────────────────┐                     │
                            │      │  StreamingAsr   │                     │
                            │      │ (SherpaOnnxAsr) │                     │
                            │      └────────┬────────┘                     │
                            │               │ Receiver<TranscriptEvent>    │
                            │               ▼                              │
                            │      ┌─────────────────┐                     │
                            │      │   asr task      │  re-emits as        │
                            │      │ (tokio spawn)   │  events::Event      │
                            │      └────────┬────────┘                     │
                            │               │                              │
                            │               ▼                              │
                            │      ┌─────────────────┐    SSE              │
                            │      │   EventBus      │═════════════════╗   │
                            │      └─────────────────┘                 ║   │
                            └──────────────────────────────────────────╫───┘
                                                                       ║
                            ┌──────────────────────────────────────────╫───┐
                            │   Electron renderer                      ║   │
                            │   ┌─────────────────────┐                ║   │
                            │   │ /eventStream client │◀══════════════╝    │
                            │   └──────────┬──────────┘                    │
                            │              ▼                               │
                            │   ┌─────────────────────┐                    │
                            │   │ Transcript pane     │                    │
                            │   │ (per call, two cols)│                    │
                            │   └─────────────────────┘                    │
                            └──────────────────────────────────────────────┘
```

The pieces that need to exist on the voice side:

1. **A tap on the RTP receive path.** Today's RTP receive in `wavekat-sip`
   decodes G.711 and hands i16 samples to the playback path. We add a
   second consumer that gets the same samples — see "Audio plumbing"
   below.

2. **A tap on the mic capture path.** Mirror.

3. **One `StreamingAsr` per active call.** Constructed when the call goes
   `Active`, dropped when it goes `Ended`. Mic frames push with
   `Channel::Local`, RTP frames push with `Channel::Remote`.

4. **An asr task** that drains the `Receiver<TranscriptEvent>` and republishes
   each event onto the daemon's existing `EventBus`, wrapped in a new
   `Event::Transcript*` variant.

5. **A renderer surface** — new component in `apps/desktop/src/` that
   subscribes to the SSE stream, filters for transcript events for the
   active call, and renders two columns of partials + finals.

---

## Audio plumbing

The trait wants `AudioFrame` (from `wavekat-core`). Both sides of a call
already have a buffer we can borrow from.

### Local (mic)

`wavekat-voice` doesn't have a mic capture path landed yet (M2). When it
lands, the capture loop produces 16 kHz mono f32 frames at some chunk size
(typically 10–20 ms). Fan them out to:

1. The RTP send path (encode + transmit), as today's plan.
2. The ASR task — wrap the same buffer as an `AudioFrame::new(samples, 16_000)`.

`AudioFrame` borrows; no allocation on the hot path.

### Remote (RTP receive)

`wavekat-sip`'s RTP receive will decode G.711 µ-law/A-law to 8 kHz i16 mono.
The decoded frame is consumed by the playback path. We add a tee:

1. Playback (today's plan).
2. ASR — same i16 samples wrapped as `AudioFrame::new(samples, 8_000)`.

`SherpaOnnxAsr` resamples 8 kHz → 16 kHz internally, so the voice daemon
does **not** need to know the model's native rate.

### Frame size

sherpa-onnx accepts arbitrary chunks; internally it accumulates to the
Zipformer's feature window. The natural chunk size matches whatever the
audio path emits — 20 ms G.711 frames (160 samples at 8 kHz) work fine.

---

## Event mapping

`wavekat-asr` emits `TranscriptEvent`. `wavekat-voice` emits `Event`
(see [`events.rs`](../../wavekat-voice/crates/wavekat-voice/src/events.rs)).
Add three variants:

```rust
// wavekat-voice/crates/wavekat-voice/src/events.rs

pub enum Event {
    // ... existing variants ...

    TranscriptPartial {
        call_id: Uuid,
        channel: TranscriptChannel,  // local | remote
        ts_ms: u64,
        text: String,
        #[serde(with = "time::serde::rfc3339")]
        at: OffsetDateTime,
    },
    TranscriptFinal {
        call_id: Uuid,
        channel: TranscriptChannel,
        ts_ms: u64,
        end_ms: u64,
        text: String,
        confidence: f32,
        #[serde(with = "time::serde::rfc3339")]
        at: OffsetDateTime,
    },
    TranscriptWarning {
        call_id: Uuid,
        message: String,
        #[serde(with = "time::serde::rfc3339")]
        at: OffsetDateTime,
    },
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptChannel { Local, Remote }
```

The asr task maps 1:1:

| `TranscriptEvent` | `Event` |
|-------------------|---------|
| `Partial { channel, ts_ms, text }` | `TranscriptPartial { call_id, channel, ts_ms, text, at: now }` |
| `Final { channel, ts_ms, end_ms, text, confidence }` | `TranscriptFinal { call_id, channel, ts_ms, end_ms, text, confidence, at: now }` |
| `Warning(s)` | `TranscriptWarning { call_id, message: s, at: now }` |
| `SpeechStarted` / `SpeechEnded` | dropped (UI doesn't need them yet) |

`call_id` is the UUID the daemon already mints for the call, threaded into
the asr task at construction.

---

## Persistence

The existing event store (`db.rs`) persists everything published on the
`EventBus`. Transcript events get the same treatment by default. Two
considerations:

- **Volume.** Partials can fire many times per second. Persisting them all
  bloats the log. Recommend: persist only `TranscriptFinal` and
  `TranscriptWarning`. Partials stream to SSE for live rendering but
  bypass the store. This requires a small change to the event publisher
  to take a `persist: bool` flag, or for the asr task to call a different
  publish method.
- **PII.** Transcripts are sensitive. Storage location is already
  user-local (SQLite under the app data dir), so no exfiltration risk
  beyond what already exists for SIP From/To. If we ever ship an export
  feature, transcripts need a separate opt-in.

---

## Renderer

The Electron desktop already has a transcript-shaped slot in mind (see
`wavekat-voice/docs/06-event-log.md` for the existing log table). For the
live transcript UI:

- New component, `apps/desktop/src/components/TranscriptPane.tsx`.
- Subscribes to the existing `/eventStream` SSE endpoint, filters for
  `kind === "transcript_partial" | "transcript_final"` matching the
  active call id.
- Two columns: "You" (`channel = local`) on the right, the remote party's
  display name (or AOR) on the left. Mirror the way iMessage renders
  inbound vs outbound.
- Partials render in a muted color and replace any prior partial on the
  same channel since the last final. Finals render in the normal text
  color and append.
- When a `TranscriptFinal` arrives with `ts_ms` overlapping a still-on-
  screen partial on the same channel, drop the partial and render the
  final in its place.

UI copy rules from `wavekat-voice/CLAUDE.md` apply — no "ASR", no
"transcription engine", no "endpoint." User-facing labels: "Transcript",
"Live transcript", "You said", "<contact> said." Settings exposure for
selecting a backend lives in `/settings/audio` or a new
`/settings/transcript` page; keep it labeled "Transcript" not "ASR."

---

## Settings

A future `/settings/transcript` page exposes:

- **Enable live transcript** — boolean. Default off in v1 (large model
  download; opt-in to first run).
- **Language / model** — dropdown: "Bilingual (English + Chinese)",
  "English only", "Chinese only", "Off."
- **Quality** — radio: "Local (private, ~100 MB)", "Cloud (Deepgram)" once
  a commercial backend lands. Hidden until then.

Settings shape lives in `wavekat-voice/src/settings.rs`; not designing it
here.

---

## What lands in this PR vs follow-ups

This is just the contract. Actual integration is a `wavekat-voice` change,
which depends on:

1. The first `wavekat-asr` release that ships a real backend (`0.0.2` or
   so, gated on landing [`03`](03-sherpa-onnx-backend.md)).
2. The mic capture path in `wavekat-voice` (M2 in the roadmap).
3. The RTP receive tap in `wavekat-sip`.

Sequence we recommend:

1. **`wavekat-asr` 0.0.2** — sherpa-onnx backend, single channel, file-
   driven example.
2. **`wavekat-asr` 0.0.3** — dual channel + resampling.
3. **`wavekat-voice` M2 mic capture** — independent, blocks #4.
4. **`wavekat-voice` `Event::Transcript*` + asr task** — depends on #2 and #3.
5. **`wavekat-voice` renderer `TranscriptPane`** — depends on #4.
6. **`wavekat-voice` `/settings/transcript`** — last.

---

## Open questions

- **Does the renderer want word-level timestamps?** Today the contract
  carries only segment-level `ts_ms`/`end_ms`. If the UI wants karaoke-
  style word highlighting, we add per-word timestamps to
  `TranscriptEvent::Final` — sherpa-onnx exposes them on `OnlineRecognizerResult`.
  Cheap to add later.
- **Speaker labels beyond local/remote.** Conference calls with three
  remote participants would need real diarization. Not v1.
- **Should partials persist for debugging?** Argument for: ASR debugging
  benefits from seeing the partial flicker. Argument against: log
  bloat. Punt to "off by default, behind a debug flag."
