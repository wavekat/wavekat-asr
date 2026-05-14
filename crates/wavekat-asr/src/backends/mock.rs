//! Mock ASR backend.
//!
//! Doesn't actually transcribe — emits a scripted sequence of
//! [`TranscriptEvent`]s when [`MockAsr::push_audio`] is called. The goal
//! is to let downstream consumers (`wavekat-voice` first) wire the SSE
//! plumbing and UI against a real `Receiver<TranscriptEvent>` before any
//! cloud / local backend lands.

use std::collections::VecDeque;
use std::sync::mpsc::{channel, Receiver, Sender};

use crate::{AsrError, AudioFrame, Channel, StreamingAsr, TranscriptEvent};

/// Mock streaming ASR session.
///
/// Emits a fixed sequence of partials → final per `push_audio` call. The
/// sequence is configurable via [`MockAsr::with_script`]; the default
/// script is `["hello", "hello there"]` followed by a final.
pub struct MockAsr {
    tx: Sender<TranscriptEvent>,
    script: VecDeque<TranscriptEvent>,
    finished: bool,
    audio_pushes: u64,
}

impl MockAsr {
    /// Build a new mock session paired with its event receiver.
    ///
    /// The default script emits two partials then a final on the first
    /// three `push_audio` calls.
    pub fn new() -> (Self, Receiver<TranscriptEvent>) {
        Self::with_script(default_script())
    }

    /// Build a mock session that will replay the given script. Each entry
    /// is emitted on the next `push_audio` call (round-robin); once the
    /// script is exhausted further pushes emit nothing.
    pub fn with_script(script: Vec<TranscriptEvent>) -> (Self, Receiver<TranscriptEvent>) {
        let (tx, rx) = channel();
        (
            Self {
                tx,
                script: script.into(),
                finished: false,
                audio_pushes: 0,
            },
            rx,
        )
    }

    /// Total number of `push_audio` calls received. Surfaced for tests.
    pub fn audio_pushes(&self) -> u64 {
        self.audio_pushes
    }
}

impl StreamingAsr for MockAsr {
    fn push_audio(&mut self, _frame: &AudioFrame, _channel: Channel) -> Result<(), AsrError> {
        if self.finished {
            return Err(AsrError::AlreadyFinished);
        }
        self.audio_pushes += 1;
        if let Some(evt) = self.script.pop_front() {
            // Best-effort: if the receiver is gone the consumer doesn't
            // care anymore. Swallow rather than fail the push path.
            let _ = self.tx.send(evt);
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), AsrError> {
        if self.finished {
            return Err(AsrError::AlreadyFinished);
        }
        self.finished = true;
        let _ = self.tx.send(TranscriptEvent::SpeechEnded {
            channel: Channel::Local,
            ts_ms: 0,
        });
        Ok(())
    }

    fn reset(&mut self, _channel: Channel) -> Result<(), AsrError> {
        Ok(())
    }
}

fn default_script() -> Vec<TranscriptEvent> {
    vec![
        TranscriptEvent::SpeechStarted {
            channel: Channel::Local,
            ts_ms: 0,
        },
        TranscriptEvent::Partial {
            channel: Channel::Local,
            ts_ms: 0,
            text: "hello".into(),
        },
        TranscriptEvent::Partial {
            channel: Channel::Local,
            ts_ms: 0,
            text: "hello there".into(),
        },
        TranscriptEvent::Final {
            channel: Channel::Local,
            ts_ms: 0,
            end_ms: 1500,
            text: "hello there".into(),
            confidence: 1.0,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_script_emits_partials_then_final() {
        let (mut asr, rx) = MockAsr::new();
        let samples = vec![0i16; 160];
        let frame = AudioFrame::new(&samples, 16_000);

        for _ in 0..4 {
            asr.push_audio(&frame, Channel::Local).unwrap();
        }

        let events: Vec<_> = rx.try_iter().collect();
        assert_eq!(events.len(), 4);
        assert!(matches!(events[3], TranscriptEvent::Final { .. }));
    }

    #[test]
    fn finish_twice_errors() {
        let (mut asr, _rx) = MockAsr::new();
        asr.finish().unwrap();
        assert!(matches!(asr.finish(), Err(AsrError::AlreadyFinished)));
    }
}
