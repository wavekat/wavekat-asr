//! Backend implementations of [`crate::StreamingAsr`].
//!
//! Each backend lives behind its own Cargo feature so dependants can
//! pull in only what they ship.

#[cfg(feature = "sherpa-onnx")]
pub mod sherpa_onnx;
