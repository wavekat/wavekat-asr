//! Live microphone transcription with the sherpa-onnx backend.
//!
//! Captures audio from the default input device via [`cpal`], downmixes
//! to mono, resamples to 16 kHz, and streams it into [`SherpaOnnxAsr`].
//! Partials and finals are printed as they arrive. Hit `Ctrl-C` to stop.
//!
//! ```sh
//! cargo run --release --example transcribe_mic --features sherpa-onnx
//! ```
//!
//! First run downloads the bilingual EN+ZH Zipformer (~75 MB) into
//! hf-hub's cache.
//!
//! Pick a different model with `WAVEKAT_ASR_PRESET=<name>`:
//!   `bilingual` (default) | `en` | `zh` | `paraformer-zh-en`

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};

use wavekat_asr::backends::sherpa_onnx::{
    ModelPreset, SherpaOnnxAsr, BILINGUAL_ZH_EN, PARAFORMER_BILINGUAL_ZH_EN, PARAFORMER_ZH,
    ZIPFORMER_EN,
};
use wavekat_asr::{AudioFrame, Channel, StreamingAsr, TranscriptEvent};
use wavekat_core::AudioFrame as CoreAudioFrame;

const TARGET_RATE: u32 = 16_000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let running = Arc::new(AtomicBool::new(true));
    {
        let r = running.clone();
        ctrlc::set_handler(move || r.store(false, Ordering::SeqCst))?;
    }

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("no default input device found")?;
    #[allow(deprecated)]
    let name = device.name().unwrap_or_else(|_| "<unknown>".into());
    let supported = device.default_input_config()?;
    let device_rate: u32 = supported.sample_rate();
    let channels = supported.channels() as usize;
    let sample_format = supported.sample_format();

    eprintln!(
        "input device : {name} ({device_rate} Hz, {channels} ch, {:?})",
        sample_format
    );
    let preset = pick_preset();
    eprintln!(
        "model preset : {} (override with WAVEKAT_ASR_PRESET)",
        preset_label()
    );
    eprintln!("constructing sherpa-onnx backend (may download model on first run)…");
    let (mut asr, asr_rx) = SherpaOnnxAsr::with_preset(preset)?;
    eprintln!("listening — speak into the mic, Ctrl-C to stop.\n");

    let (audio_tx, audio_rx) = channel::<Vec<f32>>();
    let config: StreamConfig = supported.into();
    let err_cb = |e| eprintln!("stream error: {e}");

    let stream = match sample_format {
        SampleFormat::F32 => device.build_input_stream(
            &config,
            make_callback::<f32>(audio_tx.clone(), channels),
            err_cb,
            None,
        )?,
        SampleFormat::I16 => device.build_input_stream(
            &config,
            make_callback::<i16>(audio_tx.clone(), channels),
            err_cb,
            None,
        )?,
        SampleFormat::U16 => device.build_input_stream(
            &config,
            make_callback::<u16>(audio_tx.clone(), channels),
            err_cb,
            None,
        )?,
        other => return Err(format!("unsupported sample format: {other:?}").into()),
    };
    drop(audio_tx);
    stream.play()?;

    while running.load(Ordering::SeqCst) {
        match audio_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(samples) => {
                let frame_at_device = CoreAudioFrame::from_vec(samples, device_rate);
                let frame_16k: CoreAudioFrame<'static> = if device_rate == TARGET_RATE {
                    frame_at_device
                } else {
                    frame_at_device.resample(TARGET_RATE)?
                };
                let f = AudioFrame::new(frame_16k.samples(), TARGET_RATE);
                asr.push_audio(&f, Channel::Local)?;
                drain(&asr_rx);
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    eprintln!("\nstopping…");
    drop(stream);
    asr.finish()?;
    drain(&asr_rx);
    Ok(())
}

trait ToF32: Copy {
    fn to_f32(self) -> f32;
}
impl ToF32 for f32 {
    fn to_f32(self) -> f32 {
        self
    }
}
impl ToF32 for i16 {
    fn to_f32(self) -> f32 {
        self as f32 / i16::MAX as f32
    }
}
impl ToF32 for u16 {
    fn to_f32(self) -> f32 {
        (self as f32 - 32768.0) / 32768.0
    }
}

fn make_callback<T: ToF32 + Send + 'static>(
    tx: Sender<Vec<f32>>,
    channels: usize,
) -> impl FnMut(&[T], &cpal::InputCallbackInfo) + Send + 'static {
    move |data: &[T], _| {
        let mono: Vec<f32> = if channels == 1 {
            data.iter().map(|s| s.to_f32()).collect()
        } else {
            data.chunks_exact(channels)
                .map(|frame| {
                    let sum: f32 = frame.iter().map(|s| s.to_f32()).sum();
                    sum / channels as f32
                })
                .collect()
        };
        if !mono.is_empty() {
            let _ = tx.send(mono);
        }
    }
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

fn drain(rx: &Receiver<TranscriptEvent>) {
    loop {
        match rx.try_recv() {
            Ok(TranscriptEvent::Partial { text, .. }) => {
                print!("\r\x1b[2Kpartial: {text}");
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
            Ok(TranscriptEvent::Final {
                ts_ms,
                end_ms,
                text,
                ..
            }) => {
                println!("\r\x1b[2K[{ts_ms:>6}-{end_ms:<6} ms] {text}");
            }
            Ok(TranscriptEvent::Warning(msg)) => eprintln!("warning: {msg}"),
            Ok(TranscriptEvent::SpeechStarted { .. })
            | Ok(TranscriptEvent::SpeechEnded { .. }) => {}
            Err(TryRecvError::Empty) => return,
            Err(TryRecvError::Disconnected) => return,
        }
    }
}
