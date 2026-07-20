//! A compact Torch-like tensor API with a lazy graph and a traced JIT.
//!
//! The default backend is portable CPU Rust. Enable `cuda` and build with
//! `cargo oxide` to compile the same crate's kernels to PTX with `NVLabs`
//! cuda-oxide.

mod error;
pub mod jit;
pub mod models;
pub mod nn;
mod tensor;

#[cfg(feature = "cuda")]
mod cuda;

pub use error::{Error, Result};
pub use tensor::{Device, Tensor};

/// Creates a tensor from row-major `f32` data.
///
/// # Errors
///
/// Returns [`Error::InvalidShape`] when the shape does not match the data.
pub fn tensor(data: impl Into<Vec<f32>>, shape: impl Into<Vec<usize>>) -> Result<Tensor> {
    Tensor::from_vec(data.into(), shape.into())
}
