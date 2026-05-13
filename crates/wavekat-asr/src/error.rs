use thiserror::Error;

/// Errors that can occur during streaming ASR.
#[derive(Debug, Error)]
pub enum AsrError {
    #[error("backend error: {0}")]
    Backend(String),

    #[error("invalid audio frame: {0}")]
    InvalidFrame(String),

    #[error("session already finished")]
    AlreadyFinished,

    #[error("backend not compiled in — enable the matching Cargo feature")]
    BackendNotEnabled,

    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}
