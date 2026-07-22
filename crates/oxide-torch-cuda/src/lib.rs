#![cfg_attr(feature = "cuda", feature(core_intrinsics))]

//! CUDA kernels, graph execution, cuBLAS, and cuDNN primitives for oxide-torch.

#[cfg(feature = "cuda")]
pub mod cublas;
#[cfg(feature = "cuda")]
pub mod cuda_graph;
#[cfg(feature = "cuda")]
pub mod cudnn;
mod error;
#[cfg(feature = "cuda")]
mod kernel_module;

pub use error::{Error, Result};
#[cfg(feature = "cuda")]
pub use kernel_module::module as kernels;
