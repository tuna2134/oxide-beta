//! Reference model implementations built on top of `oxide-torch`.

pub mod models;

#[cfg(feature = "cuda")]
pub mod gemma4_cuda;

pub use models::{bert, gemma4, mobilenet_v4};
