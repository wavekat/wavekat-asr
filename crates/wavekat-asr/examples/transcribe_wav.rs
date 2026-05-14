//! Transcribe a 16 kHz mono WAV file with the sherpa-onnx backend.
//!
//! ```sh
//! cargo run --release --example transcribe_wav --features sherpa-onnx -- path/to/audio.wav
//! ```
//!
//! On first run with no `WAVEKAT_ASR_MODEL_DIR` set, the default
//! bilingual EN+ZH streaming Zipformer is downloaded from HuggingFace
//! into hf-hub's cache (~/.cache/huggingface). Subsequent runs reuse it.
//!
//! Pick a different model with `WAVEKAT_ASR_PRESET=<name>`:
//!   `bilingual` (default) | `en` | `zh` | `paraformer-zh-en`

use std::path::PathBuf;

use wavekat_asr::backends::sherpa_onnx::{
    ModelPreset, SherpaOnnxAsr, BILINGUAL_ZH_EN, PARAFORMER_BILINGUAL_ZH_EN, PARAFORMER_ZH,
    ZIPFORMER_EN,
};
use wavekat_asr::{AudioFrame, Channel, StreamingAsr, TranscriptEvent};
use wavekat_core::AudioFrame as CoreAudioFrame;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let wav_path: PathBuf = std::env::args()
        .nth(1)
        .ok_or("usage: transcribe_wav <path-to-16k-wav>")?
        .into();

    let frame = CoreAudioFrame::from_wav(&wav_path)?;
    if frame.sample_rate() != 16_000 {
        return Err(format!(
            "this example accepts 16 kHz WAVs only (got {} Hz)",
            frame.sample_rate()
        )
        .into());
    }
    eprintln!(
        "loaded {} ({:.2}s @ {} Hz)",
        wav_path.display(),
        frame.duration_secs(),
        frame.sample_rate()
    );

    let preset = pick_preset();
    eprintln!(
        "constructing sherpa-onnx backend with preset `{}` (may download model on first run)…",
        preset_label()
    );
    let (mut asr, rx) = SherpaOnnxAsr::with_preset(preset)?;

    // Push in 100 ms chunks so we get partials during streaming, not
    // just one block at the end.
    let chunk = 1_600;
    for window in frame.samples().chunks(chunk) {
        let sub: AudioFrame = AudioFrame::new(window, 16_000);
        asr.push_audio(&sub, Channel::Local)?;
        drain(&rx);
    }
    asr.finish()?;
    drain(&rx);

    Ok(())
}

fn preset_label() -> String {
    std::env::var("WAVEKAT_ASR_PRESET").unwrap_or_else(|_| "bilingual".into())
}

fn pick_preset() -> ModelPreset {
    match std::env::var("WAVEKAT_ASR_PRESET")
        .unwrap_or_default()
        .as_str()
    {
        "" | "bilingual" | "bilingual-zh-en" => BILINGUAL_ZH_EN,
        "en" | "english" | "zipformer-en" => ZIPFORMER_EN,
        "zh" | "chinese" | "paraformer-zh" => PARAFORMER_ZH,
        "paraformer-zh-en" | "paraformer-bilingual" => PARAFORMER_BILINGUAL_ZH_EN,
        other => {
            eprintln!(
                "unknown WAVEKAT_ASR_PRESET `{other}`; falling back to `bilingual`"
            );
            BILINGUAL_ZH_EN
        }
    }
}

fn drain(rx: &std::sync::mpsc::Receiver<TranscriptEvent>) {
    for event in rx.try_iter() {
        match event {
            TranscriptEvent::Partial { ts_ms, text, .. } => {
                println!("[{ts_ms:>6} ms] partial: {text}");
            }
            TranscriptEvent::Final {
                ts_ms,
                end_ms,
                text,
                ..
            } => {
                println!("[{ts_ms:>6}-{end_ms:<6} ms] final  : {text}");
            }
            TranscriptEvent::Warning(msg) => eprintln!("warning: {msg}"),
            TranscriptEvent::SpeechStarted { .. } | TranscriptEvent::SpeechEnded { .. } => {}
        }
    }
}
