//! Gemma 4 persistent BF16 CUDA state.
#![allow(unsafe_code)]

use crate::cublas::Cublas;
use crate::models::gemma4::Gemma4ForCausalLM;
use crate::{Error, Result};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use safetensors::Dtype;
use std::collections::HashMap;
use std::sync::Arc;

/// Persistent CUDA resources for Gemma 4 inference.
///
/// All BF16 language-model weights are uploaded once during construction and
/// remain resident until this value is dropped.
struct CudaWeight {
    shape: Vec<usize>,
    buffer: DeviceBuffer<u16>,
}

pub struct Gemma4CudaState {
    pub(crate) _context: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) cublas: Cublas,
    module: crate::cuda::kernels::LoadedModule,
    weights: HashMap<String, CudaWeight>,
}

impl Gemma4CudaState {
    pub(crate) fn load(model: &Gemma4ForCausalLM, device: usize) -> Result<Self> {
        let context = CudaContext::new(device).map_err(cuda_error)?;
        let stream = context.default_stream();
        let cublas = Cublas::new()?;
        let module = crate::cuda::kernels::load(&context).map_err(cuda_error)?;
        let names: Vec<_> = model
            .checkpoint_weight_names()
            .filter(|name| {
                name.starts_with("model.language_model.")
                    || name == &"model.multi_modal_projector.linear.weight"
            })
            .map(str::to_owned)
            .collect();
        let mut weights = HashMap::with_capacity(names.len());
        for name in names {
            let weight = model.with_checkpoint_view(&name, |dtype, shape, bytes| {
                if dtype != Dtype::BF16 {
                    return Err(Error::Execution(format!(
                        "Gemma 4 CUDA weight `{name}` is {dtype:?}, expected BF16"
                    )));
                }
                let elements = shape.iter().try_fold(1usize, |size, dimension| {
                    size.checked_mul(*dimension)
                        .ok_or_else(|| Error::InvalidShape("Gemma 4 weight shape overflow".into()))
                })?;
                if bytes.len() != elements * 2 || bytes.as_ptr().align_offset(2) != 0 {
                    return Err(Error::Execution(format!(
                        "Gemma 4 weight `{name}` has an invalid BF16 payload"
                    )));
                }
                // SAFETY: alignment and exact 2*elements byte extent were
                // checked; every u16 bit pattern is a valid BF16 carrier. The
                // synchronous copy completes before the mmap borrow ends.
                let host =
                    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u16>(), elements) };
                let buffer = DeviceBuffer::from_host(&stream, host).map_err(cuda_error)?;
                Ok(CudaWeight {
                    shape: shape.to_vec(),
                    buffer,
                })
            })?;
            weights.insert(name, weight);
        }
        stream.synchronize().map_err(cuda_error)?;
        Ok(Self {
            _context: context,
            stream,
            cublas,
            module,
            weights,
        })
    }

    /// Number of persistent checkpoint tensors stored on the GPU.
    #[must_use]
    pub fn weight_count(&self) -> usize {
        self.weights.len()
    }

    /// Total bytes occupied by persistent BF16 checkpoint tensors.
    #[must_use]
    pub fn weight_bytes(&self) -> usize {
        self.weights
            .values()
            .map(|weight| weight.buffer.len() * std::mem::size_of::<u16>())
            .sum()
    }

    fn weight(&self, suffix: &str) -> Result<&CudaWeight> {
        let aliases = [
            format!("model.{suffix}"),
            format!("model.language_model.{suffix}"),
            format!("language_model.{suffix}"),
            suffix.to_owned(),
        ];
        aliases
            .iter()
            .find_map(|name| self.weights.get(name))
            .ok_or_else(|| Error::Execution(format!("CUDA weight `{suffix}` is missing")))
    }

    fn output_f32(&self, len: usize) -> Result<DeviceBuffer<f32>> {
        // SAFETY: every caller launches a kernel/cuBLAS operation that writes
        // all elements on `self.stream` before any read or synchronization.
        unsafe { DeviceBuffer::uninitialized_async(&self.stream, len) }.map_err(cuda_error)
    }

    fn output_bf16(&self, len: usize) -> Result<DeviceBuffer<u16>> {
        // SAFETY: same invariant as `output_f32`.
        unsafe { DeviceBuffer::uninitialized_async(&self.stream, len) }.map_err(cuda_error)
    }

    fn launch_config(len: usize) -> Result<LaunchConfig> {
        if len == 0 {
            return Err(Error::InvalidShape("zero-sized Gemma CUDA launch".into()));
        }
        Ok(LaunchConfig::for_num_elems(len as u32))
    }

    /// Executes tied-embedding LM-head projection for one token.
    ///
    /// This is also a runtime validation of the persistent BF16 store and the
    /// dynamically loaded cuBLAS BF16 GEMM ABI.
    pub fn embedding_logits(&self, token: u32, hidden_size: usize) -> Result<Vec<f32>> {
        let embedding = self.weight("embed_tokens.weight")?;
        if embedding.shape.len() != 2 || embedding.shape[1] != hidden_size {
            return Err(Error::InvalidShape(format!(
                "embed_tokens.weight has unexpected shape {:?}",
                embedding.shape
            )));
        }
        let token = token as usize;
        if token >= embedding.shape[0] {
            return Err(Error::Execution(
                "token id exceeds CUDA embedding table".into(),
            ));
        }
        let mut hidden = self.output_f32(hidden_size)?;
        #[allow(clippy::cast_precision_loss)]
        let scale = (hidden_size as f32).sqrt();
        // SAFETY: token bounds and `[vocab, hidden]` shape were checked and
        // output has exactly `hidden_size` elements.
        unsafe {
            self.module.gemma_bf16_to_f32_scaled(
                &self.stream,
                Self::launch_config(hidden_size)?,
                token * hidden_size,
                scale,
                &embedding.buffer,
                &mut hidden,
            )
        }
        .map_err(cuda_error)?;
        let mut hidden_bf16 = self.output_bf16(hidden_size)?;
        // SAFETY: input/output lengths are equal and allocations are disjoint.
        unsafe {
            self.module.gemma_f32_to_bf16(
                &self.stream,
                Self::launch_config(hidden_size)?,
                &hidden,
                &mut hidden_bf16,
            )
        }
        .map_err(cuda_error)?;
        let mut logits = self.output_f32(embedding.shape[0])?;
        self.cublas.linear_bf16_f32(
            &self.stream,
            1,
            embedding.shape[0],
            hidden_size,
            &hidden_bf16,
            &embedding.buffer,
            &mut logits,
        )?;
        logits.to_host_vec(&self.stream).map_err(cuda_error)
    }
}

fn cuda_error(error: impl std::fmt::Display) -> Error {
    Error::Execution(format!("Gemma 4 CUDA error: {error}"))
}
