//! Minimal dynamically loaded cuBLAS BF16 GEMM backend.
#![allow(unsafe_code)]

use crate::{Error, Result};
use cuda_core::{CudaStream, DeviceBuffer};
use libloading::Library;
use std::ffi::{c_int, c_void};

type Handle = *mut c_void;
type Status = c_int;
type Create = unsafe extern "C" fn(*mut Handle) -> Status;
type Destroy = unsafe extern "C" fn(Handle) -> Status;
type SetStream = unsafe extern "C" fn(Handle, *mut c_void) -> Status;
type GemmEx = unsafe extern "C" fn(
    Handle,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const c_void,
    *const c_void,
    c_int,
    c_int,
    *const c_void,
    c_int,
    c_int,
    *const c_void,
    *mut c_void,
    c_int,
    c_int,
    c_int,
    c_int,
) -> Status;

const OP_N: c_int = 0;
const OP_T: c_int = 1;
const CUDA_R_32F: c_int = 0;
const CUDA_R_16BF: c_int = 14;
const COMPUTE_32F: c_int = 68;
const GEMM_DEFAULT_TENSOR_OP: c_int = 99;

pub(crate) struct Cublas {
    handle: Handle,
    destroy: Destroy,
    set_stream: SetStream,
    gemm_ex: GemmEx,
    _library: Library,
}

impl Cublas {
    pub(crate) fn new() -> Result<Self> {
        let library = ["libcublas.so.13", "libcublas.so.12", "libcublas.so"]
            .into_iter()
            .find_map(|name| unsafe { Library::new(name).ok() })
            .ok_or_else(|| Error::Execution("cuBLAS shared library was not found".into()))?;
        // SAFETY: symbol names and signatures match the stable cuBLAS C ABI.
        unsafe {
            let create: Create = *library.get(b"cublasCreate_v2\0").map_err(ffi_error)?;
            let destroy: Destroy = *library.get(b"cublasDestroy_v2\0").map_err(ffi_error)?;
            let set_stream: SetStream = *library.get(b"cublasSetStream_v2\0").map_err(ffi_error)?;
            let gemm_ex: GemmEx = *library.get(b"cublasGemmEx\0").map_err(ffi_error)?;
            let mut handle = std::ptr::null_mut();
            check(create(&mut handle), "cublasCreate_v2")?;
            Ok(Self {
                handle,
                destroy,
                set_stream,
                gemm_ex,
                _library: library,
            })
        }
    }

    /// Computes row-major `output[m,n] = input[m,k] * weight[n,k]^T`.
    pub(crate) fn linear_bf16_f32(
        &self,
        stream: &CudaStream,
        m: usize,
        n: usize,
        k: usize,
        input: &DeviceBuffer<u16>,
        weight: &DeviceBuffer<u16>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if input.len() != m * k || weight.len() != n * k || output.len() != m * n {
            return Err(Error::InvalidShape(
                "cuBLAS linear buffer length mismatch".into(),
            ));
        }
        let (m, n, k) = (to_i32(m)?, to_i32(n)?, to_i32(k)?);
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        // SAFETY: buffers have the validated extents; row-major output is
        // represented as its column-major transpose without copying.
        unsafe {
            check(
                (self.set_stream)(self.handle, stream.cu_stream().cast()),
                "cublasSetStream_v2",
            )?;
            check(
                (self.gemm_ex)(
                    self.handle,
                    OP_T,
                    OP_N,
                    n,
                    m,
                    k,
                    (&raw const alpha).cast(),
                    weight.cu_deviceptr() as usize as *const c_void,
                    CUDA_R_16BF,
                    k,
                    input.cu_deviceptr() as usize as *const c_void,
                    CUDA_R_16BF,
                    k,
                    (&raw const beta).cast(),
                    output.cu_deviceptr() as usize as *mut c_void,
                    CUDA_R_32F,
                    n,
                    COMPUTE_32F,
                    GEMM_DEFAULT_TENSOR_OP,
                ),
                "cublasGemmEx",
            )
        }
    }
}

impl Drop for Cublas {
    fn drop(&mut self) {
        // SAFETY: handle was created once and is destroyed once here.
        let _ = unsafe { (self.destroy)(self.handle) };
    }
}

fn check(status: Status, operation: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::Execution(format!(
            "{operation} failed with cuBLAS status {status}"
        )))
    }
}

fn to_i32(value: usize) -> Result<c_int> {
    c_int::try_from(value).map_err(|_| Error::Execution("cuBLAS dimension exceeds i32".into()))
}

fn ffi_error(error: libloading::Error) -> Error {
    Error::Execution(format!("failed to load cuBLAS symbol: {error}"))
}
