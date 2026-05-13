//! # wavekat-asr
//!
//! Streaming ASR trait surface, intended to wrap one or more speech-to-text
//! backends behind a common Rust API. Modeled on the same pattern as
//! [`wavekat-vad`] and [`wavekat-turn`].
//!
//! [`wavekat-vad`]: https://crates.io/crates/wavekat-vad
//! [`wavekat-turn`]: https://crates.io/crates/wavekat-turn
//!
//! # Status
//!
//! This release ships only the trait shape and a `mock` backend
//! ([`backends::mock`], behind the `mock` Cargo feature) so downstream
//! consumers can wire integration tests against the contract. No real
//! ASR backends are bundled yet — the trait may iterate before the
//! first one lands. Pin to an exact patch version.

pub mod backends;
pub mod error;

pub use error::AsrError;
pub use wavekat_core::AudioFrame;

/// Which side of a two-channel call the audio (or transcript) belongs to.
///
/// The daemon tees both RTP directions through one ASR instance, so every
/// event needs to carry the channel it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    /// Audio captured from the local mic — what the user said.
    Local,
    /// Audio received over RTP — what the remote party said.
    Remote,
}

/// One transcript event emitted by a [`StreamingAsr`] backend.
#[derive(Debug, Clone)]
pub enum TranscriptEvent {
    /// Backend has begun receiving speech on this channel.
    ///
    /// Optional — not every backend emits this; consumers must not gate
    /// finals on having seen a `SpeechStarted` first.
    SpeechStarted { channel: Channel, ts_ms: u64 },

    /// Backend has detected end of speech on this channel.
    ///
    /// Optional, same caveat as [`SpeechStarted`](TranscriptEvent::SpeechStarted).
    SpeechEnded { channel: Channel, ts_ms: u64 },

    /// In-flight transcript that may be revised before becoming
    /// [`Final`](TranscriptEvent::Final). Render but do not persist.
    Partial {
        channel: Channel,
        /// Stream start time in ms.
        ts_ms: u64,
        text: String,
    },

    /// Stable transcript for a segment. Persist this; drop any partials
    /// that share the same channel and overlapping `ts_ms..end_ms`.
    Final {
        channel: Channel,
        ts_ms: u64,
        end_ms: u64,
        text: String,
        /// Backend-reported confidence in `[0.0, 1.0]`. Backends that
        /// don't report confidence should emit `1.0`.
        confidence: f32,
    },

    /// Backend hit a non-fatal error and continues. Fatal errors come
    /// back from [`StreamingAsr::push_audio`] / [`StreamingAsr::finish`]
    /// as `Err`.
    Warning(String),
}

/// A streaming ASR session.
///
/// Implementations are expected to:
///
/// - Accept any [`AudioFrame`] sample rate; resample internally.
/// - Be `Send` so the daemon can move them between tasks.
/// - Emit [`TranscriptEvent`]s via the receiver returned at construction
///   time (see backend docs for the constructor shape).
///
/// The trait is intentionally tiny in `0.0.1`. Expect additions
/// (per-utterance reset, hot-swappable config, metric hooks) as real
/// backends land in later releases.
pub trait StreamingAsr: Send {
    /// Push audio into the stream.
    ///
    /// Returns synchronously; transcript events are delivered on the
    /// backend's receiver, not as a return value here.
    fn push_audio(&mut self, frame: &AudioFrame, channel: Channel) -> Result<(), AsrError>;

    /// Signal end-of-stream. The backend should flush any remaining
    /// audio and emit a terminal [`Final`](TranscriptEvent::Final) per
    /// channel where applicable.
    fn finish(&mut self) -> Result<(), AsrError>;
}
