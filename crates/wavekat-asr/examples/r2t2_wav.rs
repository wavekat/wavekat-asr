//! Stream a WAV file through the Confucius4-R2T2 backend and report
//! latency / real-time factor.
//!
//! ```sh
//! cargo run --release --example r2t2_wav --features r2t2 -- path/to/audio.wav
//! cargo run --release --example r2t2_wav --features r2t2 -- path/to/audio.wav --fast
//! ```
//!
//! By default audio is pushed at real-time pace (like a microphone), so
//! the printed lag is what a live caller would see. `--fast` pushes as
//! fast as possible with the reference's fixed chunk schedule (no chunk
//! merging), which measures raw throughput.
//!
//! First run downloads the Q8_0 GGUF weights (~2.5 GB) into hf-hub's
//! cache. Environment overrides:
//!   `WAVEKAT_R2T2_WEIGHTS`  `q8` (default) | `q4` | `f16`
//!   `WAVEKAT_R2T2_LANGUAGE` e.g. `Chinese`, `English` (default: auto-detect)
//!   `WAVEKAT_R2T2_CHUNK_MS` decode chunk, 80–2000 (default 160)
//!   `WAVEKAT_R2T2_CONTEXT`  hotword / context hint
//!   `WAVEKAT_ASR_R2T2_DIR`  local directory holding the GGUF files

use std::path::PathBuf;
use std::time::{Duration, Instant};

use wavekat_asr::backends::r2t2::{R2t2Asr, R2t2Config, R2T2_F16, R2T2_Q4_K_M, R2T2_Q8_0};
use wavekat_asr::{AudioFrame, Channel, StreamingAsr, TranscriptEvent};
use wavekat_core::AudioFrame as CoreAudioFrame;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut wav_path: Option<PathBuf> = None;
    let mut fast = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--fast" => fast = true,
            other => wav_path = Some(other.into()),
        }
    }
    let wav_path = wav_path.ok_or("usage: r2t2_wav <path.wav> [--fast]")?;

    let loaded = CoreAudioFrame::from_wav(&wav_path)?;
    let frame = if loaded.sample_rate() == 16_000 {
        loaded
    } else {
        loaded.resample(16_000)?
    };
    let audio_secs = frame.duration_secs();
    eprintln!("loaded {} ({audio_secs:.2}s)", wav_path.display());

    let mut config = config_from_env();
    if fast {
        config.catch_up = false;
    }
    eprintln!(
        "loading {} (chunk {} ms, language {}, {})…",
        config.weights.model.name,
        config.chunk_ms,
        config.language.as_deref().unwrap_or("auto"),
        if fast { "fast" } else { "real-time pacing" },
    );
    let t_load = Instant::now();
    let (mut asr, rx) = R2t2Asr::with_config(config)?;
    eprintln!("model ready in {:.1}s\n", t_load.elapsed().as_secs_f64());

    // Push 20 ms frames, like a mic callback would.
    let frame_len = 320;
    let start = Instant::now();
    for (i, window) in frame.samples().chunks(frame_len).enumerate() {
        if !fast {
            let due = Duration::from_millis(i as u64 * 20);
            if let Some(wait) = due.checked_sub(start.elapsed()) {
                std::thread::sleep(wait);
            }
        }
        asr.push_audio(&AudioFrame::new(window, 16_000), Channel::Local)?;
        drain(&rx, start);
    }
    asr.finish()?;
    drain(&rx, start);
    let wall = start.elapsed().as_secs_f64();

    let s = asr.stats();
    println!("\n--- stats ---");
    println!("audio            : {audio_secs:.2} s, wall {wall:.2} s");
    println!(
        "decode steps     : {} ({} merged to catch up)",
        s.decode_steps, s.merged_steps
    );
    println!(
        "step time        : mean {:.1} ms, max {:.1} ms",
        s.mean_step_ms, s.max_step_ms
    );
    println!(
        "real-time factor : {:.3} (decode time / audio decoded)",
        s.rtf
    );
    println!(
        "lag              : mean {:.0} ms, max {:.0} ms (audio arrival -> step done)",
        s.mean_lag_ms, s.max_lag_ms
    );
    println!("utterances       : {}", s.utterances);
    println!("language         : {}", s.language);
    Ok(())
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

fn drain(rx: &std::sync::mpsc::Receiver<TranscriptEvent>, start: Instant) {
    for event in rx.try_iter() {
        let wall = start.elapsed().as_millis();
        match event {
            TranscriptEvent::Partial { ts_ms, text, .. } => {
                println!("[wall {wall:>6} ms | audio {ts_ms:>6} ms] partial: {text}");
            }
            TranscriptEvent::Final {
                ts_ms,
                end_ms,
                text,
                ..
            } => {
                println!("[wall {wall:>6} ms | audio {ts_ms:>6}-{end_ms:<6} ms] final  : {text}");
            }
            TranscriptEvent::Warning(msg) => eprintln!("warning: {msg}"),
            TranscriptEvent::SpeechStarted { .. } | TranscriptEvent::SpeechEnded { .. } => {}
        }
    }
}
