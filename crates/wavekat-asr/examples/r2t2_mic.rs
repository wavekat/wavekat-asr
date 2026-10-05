//! Live microphone transcription with the Confucius4-R2T2 backend.
//!
//! Captures the default input device via [`cpal`], downmixes to mono,
//! resamples to 16 kHz and streams into [`R2t2Asr`]. Committed text is
//! printed as it grows (it never changes once shown), finals on their
//! own line, and a status line with decode timing. `Ctrl-C` to stop.
//!
//! ```sh
//! cargo run --release --example r2t2_mic --features r2t2
//! ```
//!
//! First run downloads the Q8_0 GGUF weights (~2.5 GB). Same
//! environment overrides as the `r2t2_wav` example:
//!   `WAVEKAT_R2T2_WEIGHTS` (`q8` | `q4` | `f16`), `WAVEKAT_R2T2_LANGUAGE`,
//!   `WAVEKAT_R2T2_CHUNK_MS`, `WAVEKAT_R2T2_CONTEXT`, `WAVEKAT_ASR_R2T2_DIR`.

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};

use wavekat_asr::backends::r2t2::{
    R2t2Asr, R2t2Config, R2t2Stats, R2T2_F16, R2T2_Q4_K_M, R2T2_Q8_0,
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
    eprintln!("input device : {name} ({device_rate} Hz, {channels} ch, {sample_format:?})");

    let config = config_from_env();
    eprintln!(
        "model        : {} (chunk {} ms, language {})",
        config.weights.model.name,
        config.chunk_ms,
        config.language.as_deref().unwrap_or("auto"),
    );
    eprintln!("loading (downloads ~2.5 GB on first run)…");
    let t_load = Instant::now();
    let (mut asr, asr_rx) = R2t2Asr::with_config(config)?;
    eprintln!(
        "ready in {:.1}s — speak into the mic, Ctrl-C to stop.\n",
        t_load.elapsed().as_secs_f64()
    );

    let (audio_tx, audio_rx) = channel::<Vec<f32>>();
    let stream_config: StreamConfig = supported.into();
    let err_cb = |e| eprintln!("stream error: {e}");
    let stream = match sample_format {
        SampleFormat::F32 => device.build_input_stream(
            &stream_config,
            make_callback::<f32>(audio_tx.clone(), channels),
            err_cb,
            None,
        )?,
        SampleFormat::I16 => device.build_input_stream(
            &stream_config,
            make_callback::<i16>(audio_tx.clone(), channels),
            err_cb,
            None,
        )?,
        SampleFormat::U16 => device.build_input_stream(
            &stream_config,
            make_callback::<u16>(audio_tx.clone(), channels),
            err_cb,
            None,
        )?,
        other => return Err(format!("unsupported sample format: {other:?}").into()),
    };
    drop(audio_tx);
    stream.play()?;

    let mut view = View::default();
    while running.load(Ordering::SeqCst) {
        match audio_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(samples) => {
                let at_device = CoreAudioFrame::from_vec(samples, device_rate);
                let at_16k: CoreAudioFrame<'static> = if device_rate == TARGET_RATE {
                    at_device
                } else {
                    at_device.resample(TARGET_RATE)?
                };
                asr.push_audio(
                    &AudioFrame::new(at_16k.samples(), TARGET_RATE),
                    Channel::Local,
                )?;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
        view.update(&asr_rx, &asr.stats());
    }

    eprintln!("\nstopping…");
    drop(stream);
    asr.finish()?;
    view.update(&asr_rx, &asr.stats());
    let s = asr.stats();
    eprintln!(
        "\n{} utterances, {} steps, mean step {:.1} ms, max {:.1} ms, RTF {:.3}",
        s.utterances, s.decode_steps, s.mean_step_ms, s.max_step_ms, s.rtf
    );
    Ok(())
}

/// Terminal rendering: the in-progress utterance on one rewritable
/// line, with a timing suffix.
#[derive(Default)]
struct View {
    partial: String,
    last_draw: Option<Instant>,
}

impl View {
    fn update(&mut self, rx: &Receiver<TranscriptEvent>, stats: &R2t2Stats) {
        let mut dirty = false;
        for event in rx.try_iter() {
            match event {
                TranscriptEvent::Partial { text, .. } => {
                    self.partial = text;
                    dirty = true;
                }
                TranscriptEvent::Final {
                    ts_ms,
                    end_ms,
                    text,
                    ..
                } => {
                    print!("\r\x1b[2K");
                    println!(
                        "[{:>7.1}s–{:<7.1}s] {text}",
                        ts_ms as f64 / 1e3,
                        end_ms as f64 / 1e3
                    );
                    self.partial.clear();
                    dirty = true;
                }
                TranscriptEvent::Warning(msg) => eprintln!("\nwarning: {msg}"),
                TranscriptEvent::SpeechStarted { .. } | TranscriptEvent::SpeechEnded { .. } => {}
            }
        }
        let stale = self
            .last_draw
            .is_none_or(|t| t.elapsed() > Duration::from_millis(250));
        if dirty || stale {
            print!(
                "\r\x1b[2K{}\x1b[2m  [step {:.0} ms · lag {:.0} ms · RTF {:.2} · {}]\x1b[0m",
                self.partial,
                stats.last_step_ms,
                stats.last_lag_ms,
                stats.rtf,
                if stats.language.is_empty() {
                    "…"
                } else {
                    &stats.language
                },
            );
            let _ = std::io::stdout().flush();
            self.last_draw = Some(Instant::now());
        }
    }
}

fn config_from_env() -> R2t2Config {
    let mut config = R2t2Config::default();
    match std::env::var("WAVEKAT_R2T2_WEIGHTS")
        .unwrap_or_default()
        .as_str()
    {
        "q4" => config.weights = R2T2_Q4_K_M,
        "f16" => config.weights = R2T2_F16,
        "" | "q8" => config.weights = R2T2_Q8_0,
        other => eprintln!("unknown WAVEKAT_R2T2_WEIGHTS `{other}`; using q8"),
    }
    if let Ok(lang) = std::env::var("WAVEKAT_R2T2_LANGUAGE") {
        config.language = Some(lang).filter(|l| !l.is_empty());
    }
    if let Some(ms) = std::env::var("WAVEKAT_R2T2_CHUNK_MS")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        config.chunk_ms = ms;
    }
    if let Ok(context) = std::env::var("WAVEKAT_R2T2_CONTEXT") {
        config.context = context;
    }
    config
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
                .map(|frame| frame.iter().map(|s| s.to_f32()).sum::<f32>() / channels as f32)
                .collect()
        };
        if !mono.is_empty() {
            let _ = tx.send(mono);
        }
    }
}
