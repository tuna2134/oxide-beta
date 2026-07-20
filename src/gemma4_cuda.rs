//! Gemma 4 persistent BF16 CUDA state.
#![allow(unsafe_code)]

use crate::cublas::Cublas;
use crate::models::gemma4::Gemma4ForCausalLM;
use crate::{Error, Result};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use safetensors::Dtype;
use std::collections::HashMap;
use std::sync::Arc;

pub(crate) struct Gemma4CudaState {
    pub(crate) _context: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) cublas: Cublas,
    pub(crate) weights: HashMap<String, DeviceBuffer<u16>>,
}

impl Gemma4CudaState {
    pub(crate) fn load(model: &Gemma4ForCausalLM, device: usize) -> Result<Self> {
        let context = CudaContext::new(device).map_err(cuda_error)?;
        let stream = context.default_stream();
        let cublas = Cublas::new()?;
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
            let buffer = model.with_checkpoint_view(&name, |dtype, shape, bytes| {
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
                DeviceBuffer::from_host(&stream, host).map_err(cuda_error)
            })?;
            weights.insert(name, buffer);
        }
        Ok(Self {
            _context: context,
            stream,
            cublas,
            weights,
        })
    }
}

fn cuda_error(error: impl std::fmt::Display) -> Error {
    Error::Execution(format!("Gemma 4 CUDA error: {error}"))
}
