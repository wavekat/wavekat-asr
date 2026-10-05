//! llama.cpp inference for Confucius4-R2T2 via `llama-cpp-2`'s `mtmd`
//! (multimodal) API.
//!
//! The call sequence mirrors the reference `r2t2_llama` native extension
//! in the Confucius4-R2T2 repository: clear the KV cache, swap the
//! Qwen3-ASR audio placeholder for mtmd's media marker, tokenize the
//! prompt together with the audio, evaluate every chunk (the audio
//! encoder runs here), then decode greedily until end-of-generation.

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::mtmd::{
    mtmd_default_marker, MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText,
};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

use crate::AsrError;

use super::stream::Decoder;

/// The Qwen3-ASR audio placeholder used in the chat prompt.
const AUDIO_PLACEHOLDER: &str = "<|audio_start|><|audio_pad|><|audio_end|>";

/// Process-wide llama.cpp backend. `LlamaBackend::init` may only run
/// once per process.
fn backend() -> Result<&'static LlamaBackend, AsrError> {
    static BACKEND: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();
    BACKEND
        .get_or_init(|| {
            // Route llama.cpp / ggml logs through `tracing` (they're very
            // chatty at load time), and keep mtmd's separate logger to
            // warnings and errors. Installed before init so the GPU
            // backend's startup lines are captured too.
            llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default());
            unsafe {
                llama_cpp_sys_2::mtmd_helper_log_set(Some(mtmd_log), std::ptr::null_mut());
            }
            LlamaBackend::init().map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|e| AsrError::Backend(format!("llama.cpp backend init failed: {e}")))
}

unsafe extern "C" fn mtmd_log(
    level: llama_cpp_sys_2::ggml_log_level,
    text: *const std::os::raw::c_char,
    _user_data: *mut std::os::raw::c_void,
) {
    if text.is_null() || level < llama_cpp_sys_2::GGML_LOG_LEVEL_WARN {
        return;
    }
    let text = unsafe { std::ffi::CStr::from_ptr(text) }.to_string_lossy();
    let text = text.trim_end();
    if text.is_empty() {
        return;
    }
    if level >= llama_cpp_sys_2::GGML_LOG_LEVEL_ERROR {
        tracing::error!(target: "wavekat_asr::r2t2::mtmd", "{text}");
    } else {
        tracing::warn!(target: "wavekat_asr::r2t2::mtmd", "{text}");
    }
}

/// Where the weights live and how to run them.
#[derive(Debug, Clone)]
pub(crate) struct EngineConfig {
    pub model_path: PathBuf,
    pub mmproj_path: PathBuf,
    pub use_gpu: bool,
    pub n_threads: i32,
    pub n_ctx: u32,
}

/// Loaded language-model weights, shareable across sessions.
///
/// Each session still owns its own llama context and audio encoder,
/// because neither may be used from two threads at once.
pub struct R2t2Model {
    model: LlamaModel,
    mmproj_path: PathBuf,
    use_gpu: bool,
    n_threads: i32,
    n_ctx: u32,
}

impl std::fmt::Debug for R2t2Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("R2t2Model")
            .field("mmproj_path", &self.mmproj_path)
            .field("use_gpu", &self.use_gpu)
            .finish_non_exhaustive()
    }
}

impl R2t2Model {
    pub(crate) fn load(cfg: &EngineConfig) -> Result<Arc<Self>, AsrError> {
        for path in [&cfg.model_path, &cfg.mmproj_path] {
            if !path.exists() {
                return Err(AsrError::Backend(format!(
                    "model file not found: {}",
                    path.display()
                )));
            }
        }
        let backend = backend()?;
        let params =
            LlamaModelParams::default().with_n_gpu_layers(if cfg.use_gpu { 999 } else { 0 });
        let model = LlamaModel::load_from_file(backend, &cfg.model_path, &params)
            .map_err(|e| AsrError::Backend(format!("loading {}: {e}", cfg.model_path.display())))?;
        Ok(Arc::new(Self {
            model,
            mmproj_path: cfg.mmproj_path.clone(),
            use_gpu: cfg.use_gpu,
            n_threads: cfg.n_threads,
            n_ctx: cfg.n_ctx,
        }))
    }
}

/// One inference session: a llama context plus an audio encoder bound
/// to a shared [`R2t2Model`].
pub(crate) struct Engine {
    // Field order matters: both contexts borrow `model` and must drop
    // before the `Arc` that keeps it alive.
    ctx: LlamaContext<'static>,
    mtmd: MtmdContext,
    n_batch: i32,
    model: Arc<R2t2Model>,
}

// SAFETY: llama.cpp contexts are plain heap handles with no thread
// affinity; they only require that calls are not concurrent, which
// `&mut self` on every method that touches them guarantees.
unsafe impl Send for Engine {}

impl Engine {
    pub(crate) fn new(model: Arc<R2t2Model>) -> Result<Self, AsrError> {
        let backend = backend()?;
        // SAFETY: `ctx` borrows the model inside the `Arc`, which this
        // struct holds for its whole lifetime and drops last (see the
        // field order above). The heap allocation never moves.
        let llama_model: &'static LlamaModel = unsafe { &*std::ptr::addr_of!(model.model) };

        let n_batch = model.n_ctx.min(2048);
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(model.n_ctx))
            .with_n_batch(n_batch)
            .with_n_ubatch(n_batch.min(512))
            .with_n_threads(model.n_threads)
            .with_n_threads_batch(model.n_threads);
        let ctx = llama_model
            .new_context(backend, ctx_params)
            .map_err(|e| AsrError::Backend(format!("creating llama context: {e}")))?;

        let mtmd_params = MtmdContextParams {
            use_gpu: model.use_gpu,
            print_timings: false,
            n_threads: model.n_threads,
            ..MtmdContextParams::default()
        };
        let mmproj = path_str(&model.mmproj_path)?;
        let mtmd = MtmdContext::init_from_file(mmproj, llama_model, &mtmd_params).map_err(|e| {
            AsrError::Backend(format!("loading {}: {e}", model.mmproj_path.display()))
        })?;
        if !mtmd.support_audio() {
            return Err(AsrError::Backend(format!(
                "{} is not an audio projector",
                model.mmproj_path.display()
            )));
        }
        Ok(Self {
            ctx,
            mtmd,
            n_batch: n_batch as i32,
            model,
        })
    }

    fn eval_prompt(&mut self, audio: &[f32], prompt: &str) -> Result<(), AsrError> {
        self.ctx.clear_kv_cache();
        let text = prompt.replacen(AUDIO_PLACEHOLDER, mtmd_default_marker(), 1);
        if !text.contains(mtmd_default_marker()) {
            return Err(AsrError::Backend("prompt has no audio placeholder".into()));
        }
        let bitmap = MtmdBitmap::from_audio_data(audio)
            .map_err(|e| AsrError::Backend(format!("audio bitmap: {e}")))?;
        let chunks = self
            .mtmd
            .tokenize(
                MtmdInputText {
                    text,
                    add_special: true,
                    parse_special: true,
                },
                &[&bitmap],
            )
            .map_err(|e| AsrError::Backend(format!("mtmd tokenize: {e}")))?;
        chunks
            .eval_chunks(&self.mtmd, &self.ctx, 0, 0, self.n_batch, true)
            .map_err(|e| AsrError::Backend(format!("mtmd eval: {e}")))?;
        Ok(())
    }
}

impl Decoder for Engine {
    fn generate(
        &mut self,
        audio: &[f32],
        prompt: &str,
        max_tokens: usize,
    ) -> Result<String, AsrError> {
        if audio.is_empty() || max_tokens == 0 {
            return Ok(String::new());
        }
        self.eval_prompt(audio, prompt)?;

        let vocab = self.model.model.vocab();
        let mut sampler = LlamaSampler::greedy();
        let mut generated: Vec<LlamaToken> = Vec::with_capacity(max_tokens);
        for i in 0..max_tokens {
            let token = sampler.sample(&self.ctx, -1);
            sampler.accept(token);
            if vocab.is_eog(token) {
                break;
            }
            generated.push(token);
            if i + 1 == max_tokens {
                break;
            }
            let tokens = [token];
            let mut batch = LlamaBatch::get_one(&tokens)
                .map_err(|e| AsrError::Backend(format!("llama batch: {e}")))?;
            self.ctx
                .decode(&mut batch)
                .map_err(|e| AsrError::Backend(format!("llama decode: {e}")))?;
        }

        // Piece by piece without control tokens, like the reference.
        let mut bytes = Vec::with_capacity(generated.len() * 4);
        for token in generated {
            bytes.extend(vocab.token_to_piece(token, false, None));
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn tokenize(&self, text: &str) -> Vec<i32> {
        self.model
            .model
            .vocab()
            .tokenize(text.as_bytes(), false, true)
            .into_iter()
            .map(|t| t.0)
            .collect()
    }

    fn detokenize(&self, tokens: &[i32]) -> String {
        let tokens: Vec<LlamaToken> = tokens.iter().copied().map(LlamaToken).collect();
        let bytes = self.model.model.vocab().detokenize(&tokens, false, true);
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

fn path_str(path: &Path) -> Result<&str, AsrError> {
    path.to_str()
        .ok_or_else(|| AsrError::Backend(format!("non-UTF-8 path: {}", path.display())))
}
