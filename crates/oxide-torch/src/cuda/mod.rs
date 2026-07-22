// Kernel launch methods are unsafe because cuda-oxide cannot prove device
// buffer lengths and aliasing. This module validates those invariants at each
// call site and documents them with a SAFETY comment.
#![allow(static_mut_refs, unsafe_code)]

mod jit;
mod runtime;

pub(crate) use jit::CudaJitPlan;
pub use oxide_torch_cuda::kernels;
pub(crate) use runtime::*;
