//! The stable-prefix streaming loop, independent of the inference engine.
//!
//! Port of `R2T2ASRModel.streaming_transcribe` /
//! `finish_streaming_transcribe` and the per-chunk scheduling in
//! `ws_server.py` from the Confucius4-R2T2 repository. On every chunk:
//!
//! 1. Append the chunk to the utterance audio and re-feed **all** of it.
//! 2. Prompt with the previously decoded text minus its last
//!    `unfixed_token_num` tokens (the rollback window), so the model
//!    continues from a prefix it has already committed to.
//! 3. Cut the continuation at the `|` stable-prefix marker, roll back the
//!    same window again, and commit whatever extends the committed text.
//!
//! Committed text is append-only: once emitted it never changes, which
//! is the property R2T2 is trained for.

use crate::AsrError;

use super::text::{
    before_marker, ends_with_cjk_word, normalize_punct_by_context, parse_asr_output,
    parse_language, remove_spaces_between_cjk, ASR_TEXT_TAG,
};

/// Samples per second; R2T2 consumes 16 kHz mono f32.
pub(crate) const SAMPLE_RATE: usize = 16_000;

/// The reference scales its per-step token budget by audio length at
/// one token per 80 ms (1280 samples).
const SAMPLES_PER_TOKEN: usize = 1280;

/// The inference surface the loop needs. Implemented by the llama.cpp
/// engine, and by a scripted fake in the tests.
pub(crate) trait Decoder {
    /// Greedy-decode up to `max_tokens` tokens continuing `prompt` over
    /// `audio`, returning only the generated continuation.
    fn generate(
        &mut self,
        audio: &[f32],
        prompt: &str,
        max_tokens: usize,
    ) -> Result<String, AsrError>;
    /// Tokenize with special-token parsing (so `<asr_text>` is one token).
    fn tokenize(&self, text: &str) -> Vec<i32>;
    /// Inverse of [`tokenize`](Self::tokenize), rendering special tokens.
    /// Invalid UTF-8 from a split multi-byte character comes back as
    /// U+FFFD, which the rollback uses to widen its window.
    fn detokenize(&self, tokens: &[i32]) -> String;
}

/// The chat prompt R2T2 was trained on, up to the start of the
/// assistant turn. Same layout as `build_asr_prompt` in the reference.
pub(crate) fn build_prompt(context: &str, language: Option<&str>) -> String {
    let mut prompt = format!(
        "<|im_start|>system\n{context}<|im_end|>\n\
         <|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n\
         <|im_start|>assistant\n"
    );
    if let Some(lang) = language {
        prompt.push_str(&format!("language {lang}{ASR_TEXT_TAG}"));
    }
    prompt
}

/// Chunking parameters, all in samples at 16 kHz.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Schedule {
    /// Audio added per decode step.
    pub chunk: usize,
    /// Extra audio required before the very first decode.
    pub lookahead: usize,
    /// Upper bound on chunks merged into one step when audio is queued
    /// faster than it can be decoded. `1` reproduces the reference
    /// exactly; larger values let a slow machine keep up.
    pub max_chunks_per_step: usize,
    /// Tokens held back from the end of every hypothesis.
    pub unfixed_token_num: usize,
}

/// Outcome of one decode step, for metrics and events.
#[derive(Debug, Clone, Default)]
pub(crate) struct StepOutcome {
    /// Audio consumed by this step, in samples.
    pub samples: usize,
    /// Number of `chunk`-sized pieces merged into the step.
    pub chunks: usize,
    /// Text newly appended to the committed transcript (may be empty).
    pub appended: String,
}

/// Decoder state for one utterance.
pub(crate) struct Utterance {
    schedule: Schedule,
    prompt_raw: String,
    forced_language: Option<String>,
    /// Audio fed to the model so far (re-fed in full every step).
    audio: Vec<f32>,
    /// Audio received but not yet consumed by a step.
    pending: Vec<f32>,
    /// Raw decoded text including any `language X<asr_text>` header.
    raw: String,
    /// Latest full hypothesis, including the unfixed tail.
    text: String,
    /// Latest detected (or forced) language.
    language: String,
    /// Append-only committed transcript.
    committed: String,
    started: bool,
    /// Per-chunk token budget carried between steps (fractional, as in
    /// the reference).
    max_new_tokens: f32,
}

impl Utterance {
    pub fn new(schedule: Schedule, context: &str, language: Option<&str>) -> Self {
        Self {
            schedule,
            prompt_raw: build_prompt(context, language),
            forced_language: language.map(str::to_string),
            audio: Vec::new(),
            pending: Vec::new(),
            raw: String::new(),
            text: String::new(),
            language: String::new(),
            committed: String::new(),
            started: false,
            max_new_tokens: 0.0,
        }
    }

    pub fn push(&mut self, samples: &[f32]) {
        self.pending.extend_from_slice(samples);
    }

    /// Committed (append-only) transcript so far.
    pub fn committed(&self) -> &str {
        &self.committed
    }

    /// Latest detected language, empty until the model reports one.
    pub fn language(&self) -> &str {
        &self.language
    }

    /// Audio consumed by decode steps so far, in samples.
    pub fn decoded_samples(&self) -> usize {
        self.audio.len()
    }

    /// Audio received but not yet decoded, in samples.
    pub fn pending_samples(&self) -> usize {
        self.pending.len()
    }

    pub fn total_samples(&self) -> usize {
        self.audio.len() + self.pending.len()
    }

    /// Run one decode step if enough audio is pending. Returns `None`
    /// when waiting for more audio.
    pub fn step<D: Decoder>(&mut self, dec: &mut D) -> Result<Option<StepOutcome>, AsrError> {
        let s = self.schedule;
        let (take, chunks) = if !self.started {
            let first = s.chunk + s.lookahead;
            if self.pending.len() < first {
                return Ok(None);
            }
            (first, 1)
        } else {
            let available = self.pending.len() / s.chunk;
            if available == 0 {
                return Ok(None);
            }
            let n = available.min(s.max_chunks_per_step.max(1));
            (n * s.chunk, n)
        };

        let budget = if !self.started {
            self.started = true;
            let first = ((s.chunk + s.lookahead) / SAMPLES_PER_TOKEN).max(1) as f32;
            self.max_new_tokens = first;
            first as usize
        } else {
            // The carried budget is per chunk; scale it when chunks merge.
            ((self.max_new_tokens * chunks as f32) as usize).max(1)
        };

        self.audio.extend(self.pending.drain(..take));
        let fixed = self.decode(dec, budget)?;
        let appended = self.commit(&fixed);
        self.update_budget(!appended.is_empty());
        Ok(Some(StepOutcome {
            samples: take,
            chunks,
            appended,
        }))
    }

    /// One `streaming_transcribe` iteration over the current audio.
    /// Returns the fixed (committable) text for this step.
    fn decode<D: Decoder>(&mut self, dec: &mut D, budget: usize) -> Result<String, AsrError> {
        let k = self.schedule.unfixed_token_num;
        self.raw = before_marker(&self.raw).to_string();
        let prefix = rollback(dec, &self.raw, k, 0);
        let prefix = before_marker(&prefix).to_string();
        let prompt = format!("{}{prefix}", self.prompt_raw);

        let generated = dec.generate(&self.audio, &prompt, budget)?;
        let generated = normalize_punct_by_context(&generated).replace('\u{FFFD}', "");
        let mut raw = format!("{prefix}{generated}");

        let detected = match &self.forced_language {
            Some(lang) => lang.clone(),
            None => parse_language(&raw),
        };
        if detected == "Chinese" {
            raw = remove_spaces_between_cjk(&raw);
        }
        let (lang, txt) = parse_asr_output(&raw, self.forced_language.as_deref());

        let has_tag = raw.contains(ASR_TEXT_TAG);
        raw = if has_tag {
            let meta = raw.split(ASR_TEXT_TAG).next().unwrap_or("");
            format!("{meta}{ASR_TEXT_TAG}{txt}")
        } else {
            txt.clone()
        };
        self.raw = before_marker(&raw).to_string();

        // `language None` with no text means "no speech yet". Carrying
        // that header forward as the prompt prefix would pin the
        // utterance to "no language" for good, so drop it and let the
        // next step detect the language afresh.
        if self.forced_language.is_none() {
            if let Some((meta, rest)) = self.raw.split_once(ASR_TEXT_TAG) {
                if rest.is_empty() && meta.to_lowercase().contains("language none") {
                    self.raw.clear();
                    self.text.clear();
                    return Ok(String::new());
                }
            }
        }

        // Nothing past the header yet: nothing to roll back.
        let header_only = self
            .raw
            .split_once(ASR_TEXT_TAG)
            .is_some_and(|(_, t)| t.is_empty());
        let k = if header_only { 0 } else { k };
        let mut fixed = rollback(dec, &self.raw, k, 0);
        if let Some((_, after)) = fixed.split_once(ASR_TEXT_TAG) {
            fixed = after.to_string();
        }
        let fixed = before_marker(&fixed).to_string();
        tracing::trace!(
            audio_samples = self.audio.len(),
            budget,
            %generated,
            raw = %self.raw,
            %fixed,
            "r2t2 step"
        );

        if !self.raw.contains(ASR_TEXT_TAG) && self.forced_language.is_none() {
            // Auto-language mode and the header isn't complete yet.
            self.text.clear();
            return Ok(String::new());
        }
        self.language = lang;
        self.text = before_marker(&txt).to_string();
        Ok(fixed)
    }

    /// Append the part of `fixed` that extends the committed text, the
    /// way the reference server forwards only the new suffix.
    fn commit(&mut self, fixed: &str) -> String {
        let have = self.committed.chars().count();
        if fixed.chars().count() <= have {
            return String::new();
        }
        let appended: String = fixed.chars().skip(have).collect();
        self.committed.push_str(&appended);
        appended
    }

    /// Adaptive token budget from the reference server: reset to the
    /// per-chunk base after progress, creep up by half a token while
    /// stalled on English, and double after a Chinese word.
    fn update_budget(&mut self, progressed: bool) {
        let base = (self.schedule.chunk / SAMPLES_PER_TOKEN).max(1) as f32;
        let cap = (2.0 * base).clamp(4.0, 32.0);
        let cjk = ends_with_cjk_word(&self.committed);
        let mut next = if progressed || cjk {
            base
        } else {
            self.max_new_tokens + 0.5
        };
        if cjk {
            next *= 2.0;
        }
        self.max_new_tokens = next.min(cap);
    }

    /// Decode any remaining audio (including a partial chunk) and return
    /// the final transcript for the utterance. Mirrors
    /// `finish_streaming_transcribe` plus the server's suffix stitching.
    pub fn finish<D: Decoder>(&mut self, dec: &mut D) -> Result<FinishOutcome, AsrError> {
        let mut steps = Vec::new();
        while let Some(step) = self.step(dec)? {
            steps.push(step);
        }
        let tail = self.pending.len();
        if tail > 0 {
            self.audio.append(&mut self.pending);
            let k = self.schedule.unfixed_token_num;
            let prefix = if self.raw.is_empty() {
                String::new()
            } else {
                rollback(dec, &self.raw, k, 1)
            };
            let prefix = before_marker(&prefix).to_string();
            let prompt = format!("{}{prefix}", self.prompt_raw);
            let budget =
                ((self.schedule.chunk + self.schedule.lookahead) / SAMPLES_PER_TOKEN).max(1);
            let generated = dec.generate(&self.audio, &prompt, budget)?;
            let generated = normalize_punct_by_context(&generated).replace('\u{FFFD}', "");
            self.raw = before_marker(&format!("{prefix}{generated}")).to_string();
            let (lang, txt) = parse_asr_output(&self.raw, self.forced_language.as_deref());
            self.language = lang;
            self.text = before_marker(&txt).to_string();
        }

        let mut text = self.committed.clone();
        let have = text.chars().count();
        if self.text.chars().count() > have {
            text.extend(self.text.chars().skip(have));
        }
        Ok(FinishOutcome {
            steps,
            tail_samples: tail,
            text,
        })
    }
}

/// Result of [`Utterance::finish`].
#[derive(Debug, Clone)]
pub(crate) struct FinishOutcome {
    /// Regular steps run while draining full chunks.
    pub steps: Vec<StepOutcome>,
    /// Samples decoded in the final partial-chunk pass (0 if none).
    pub tail_samples: usize,
    /// Final transcript: committed text plus whatever the last
    /// hypothesis adds beyond it.
    pub text: String,
}

/// Drop the last `k` tokens of `text`, widening the window while the
/// cut lands inside a multi-byte character. `min_keep` keeps at least
/// that many tokens (the reference's finish path keeps one).
fn rollback<D: Decoder>(dec: &D, text: &str, mut k: usize, min_keep: usize) -> String {
    if text.is_empty() {
        return String::new();
    }
    let ids = dec.tokenize(text);
    loop {
        let end = ids.len().saturating_sub(k).max(min_keep.min(ids.len()));
        let s = if end > 0 {
            dec.detokenize(&ids[..end])
        } else {
            String::new()
        };
        if !s.contains('\u{FFFD}') {
            return s;
        }
        if end <= min_keep.min(ids.len()) {
            return if min_keep > 0 {
                s.replace('\u{FFFD}', "")
            } else {
                String::new()
            };
        }
        k += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Character-level fake: one token per char (`<asr_text>` is one
    /// token), and `generate` replays a scripted continuation per call.
    struct Fake {
        script: Vec<&'static str>,
        calls: Vec<(usize, String, usize)>,
    }

    const TAG_ID: i32 = -1;

    impl Decoder for Fake {
        fn generate(
            &mut self,
            audio: &[f32],
            prompt: &str,
            max: usize,
        ) -> Result<String, AsrError> {
            self.calls.push((audio.len(), prompt.to_string(), max));
            Ok(self
                .script
                .get(self.calls.len() - 1)
                .copied()
                .unwrap_or("")
                .to_string())
        }
        fn tokenize(&self, text: &str) -> Vec<i32> {
            let mut out = Vec::new();
            let mut rest = text;
            while !rest.is_empty() {
                if let Some(r) = rest.strip_prefix(ASR_TEXT_TAG) {
                    out.push(TAG_ID);
                    rest = r;
                } else {
                    let c = rest.chars().next().unwrap();
                    out.push(c as i32);
                    rest = &rest[c.len_utf8()..];
                }
            }
            out
        }
        fn detokenize(&self, tokens: &[i32]) -> String {
            tokens
                .iter()
                .map(|&t| {
                    if t == TAG_ID {
                        ASR_TEXT_TAG.to_string()
                    } else {
                        char::from_u32(t as u32).unwrap().to_string()
                    }
                })
                .collect()
        }
    }

    fn schedule() -> Schedule {
        Schedule {
            chunk: 2560,
            lookahead: 2560,
            max_chunks_per_step: 1,
            unfixed_token_num: 1,
        }
    }

    #[test]
    fn waits_for_chunk_plus_lookahead_before_first_decode() {
        let mut fake = Fake {
            script: vec!["hel"],
            calls: vec![],
        };
        let mut utt = Utterance::new(schedule(), "", Some("English"));
        utt.push(&vec![0.0; 5119]);
        assert!(utt.step(&mut fake).unwrap().is_none());
        utt.push(&[0.0]);
        let out = utt.step(&mut fake).unwrap().unwrap();
        assert_eq!(out.samples, 5120);
        // First budget: (chunk + lookahead) / 1280 tokens.
        assert_eq!(fake.calls[0].2, 4);
        assert!(fake.calls[0].1.ends_with("language English<asr_text>"));
        // One token is held back.
        assert_eq!(utt.committed(), "he");
    }

    #[test]
    fn commits_append_only_and_prompts_with_rolled_back_prefix() {
        let mut fake = Fake {
            script: vec!["hel", "lo w", "world|xx"],
            calls: vec![],
        };
        let mut utt = Utterance::new(schedule(), "", Some("English"));
        utt.push(&vec![0.0; 5120 + 2 * 2560]);
        let mut appended = Vec::new();
        while let Some(out) = utt.step(&mut fake).unwrap() {
            appended.push(out.appended);
        }
        // Step 2 is prompted with "he" (the rolled-back "hel") and
        // re-fed all audio so far.
        assert!(fake.calls[1].1.ends_with("<asr_text>he"));
        assert_eq!(fake.calls[1].0, 5120 + 2560);
        // Third step continues after "helo " and stops at the marker.
        assert!(fake.calls[2].1.ends_with("<asr_text>helo "));
        assert_eq!(appended, vec!["he", "lo ", "worl"]);
        assert_eq!(utt.committed(), "helo worl");
        let fin = utt.finish(&mut fake).unwrap();
        assert_eq!(fin.tail_samples, 0);
        // Final adds the held-back token from the last hypothesis.
        assert_eq!(fin.text, "helo world");
    }

    #[test]
    fn auto_language_waits_for_header() {
        let mut fake = Fake {
            // Each continuation restarts at the rolled-back token, as
            // the real model does.
            script: vec!["language", "e Chinese<asr_text>你", "你好"],
            calls: vec![],
        };
        let mut utt = Utterance::new(schedule(), "", None);
        utt.push(&vec![0.0; 5120 + 2 * 2560]);
        let first = utt.step(&mut fake).unwrap().unwrap();
        assert_eq!(first.appended, "");
        let second = utt.step(&mut fake).unwrap().unwrap();
        // "你" is the last token, so it is held back.
        assert_eq!(second.appended, "");
        assert_eq!(utt.language(), "Chinese");
        let third = utt.step(&mut fake).unwrap().unwrap();
        assert_eq!(third.appended, "你");
    }

    #[test]
    fn merges_queued_chunks_when_allowed() {
        let mut fake = Fake {
            script: vec!["a", "b"],
            calls: vec![],
        };
        let mut s = schedule();
        s.max_chunks_per_step = 4;
        let mut utt = Utterance::new(s, "", Some("English"));
        utt.push(&vec![0.0; 5120 + 3 * 2560]);
        utt.step(&mut fake).unwrap().unwrap();
        let merged = utt.step(&mut fake).unwrap().unwrap();
        assert_eq!(merged.chunks, 3);
        assert_eq!(merged.samples, 3 * 2560);
        assert_eq!(utt.pending_samples(), 0);
    }

    #[test]
    fn finish_decodes_partial_tail() {
        let mut fake = Fake {
            script: vec!["hi"],
            calls: vec![],
        };
        let mut utt = Utterance::new(schedule(), "", Some("English"));
        utt.push(&vec![0.0; 3000]);
        let fin = utt.finish(&mut fake).unwrap();
        assert_eq!(fin.tail_samples, 3000);
        assert_eq!(fin.text, "hi");
        assert_eq!(fake.calls.len(), 1);
    }

    #[test]
    fn rollback_widens_past_broken_characters() {
        struct Bytes;
        impl Decoder for Bytes {
            fn generate(&mut self, _: &[f32], _: &str, _: usize) -> Result<String, AsrError> {
                unreachable!()
            }
            fn tokenize(&self, text: &str) -> Vec<i32> {
                text.bytes().map(i32::from).collect()
            }
            fn detokenize(&self, tokens: &[i32]) -> String {
                let bytes: Vec<u8> = tokens.iter().map(|&t| t as u8).collect();
                String::from_utf8_lossy(&bytes).into_owned()
            }
        }
        // "a你" is 4 bytes; dropping 1 byte splits "你", so the window
        // widens until only "a" remains.
        assert_eq!(rollback(&Bytes, "a你", 1, 0), "a");
        assert_eq!(rollback(&Bytes, "ab", 1, 0), "a");
        assert_eq!(rollback(&Bytes, "ab", 5, 0), "");
    }

    #[test]
    fn prompt_matches_reference_layout() {
        let p = build_prompt("hotword", Some("Chinese"));
        assert_eq!(
            p,
            "<|im_start|>system\nhotword<|im_end|>\n<|im_start|>user\n\
             <|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n\
             <|im_start|>assistant\nlanguage Chinese<asr_text>"
        );
    }
}
