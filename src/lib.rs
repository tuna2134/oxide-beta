#![cfg_attr(feature = "cuda", feature(core_intrinsics))]
#![cfg_attr(feature = "cuda", allow(internal_features))]

//! A compact Torch-like tensor API with a lazy graph and a traced JIT.
//!
//! The default backend is portable CPU Rust. Enable `cuda` and build with
//! `cargo oxide` to compile the same crate's kernels to PTX with `NVLabs`
//! cuda-oxide.

pub mod data;
mod error;
pub mod jit;
pub mod loss;
pub mod models;
pub mod nn;
pub mod optim;
pub mod safetensors;
mod tensor;

#[cfg(feature = "cuda")]
mod cublas;
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
mod cuda_graph;
#[cfg(feature = "cudnn")]
mod cudnn;
#[cfg(feature = "cuda")]
pub mod gemma4_cuda;

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
