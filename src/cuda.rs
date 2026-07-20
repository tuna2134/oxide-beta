use crate::tensor::{Op, Tensor};
use crate::{Error, Result};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use std::collections::HashMap;
use std::sync::Arc;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn add(a: &[f32], b: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(value) = output.get_mut(index) {
            *value = a[raw] + b[raw];
        }
    }

    #[kernel]
    pub fn mul(a: &[f32], b: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(value) = output.get_mut(index) {
            *value = a[raw] * b[raw];
        }
    }

    #[kernel]
    pub fn relu(input: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(value) = output.get_mut(index) {
            let input_value = input[raw];
            *value = if input_value > 0.0 { input_value } else { 0.0 };
        }
    }

    #[kernel]
    pub fn matmul(
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(value) = output.get_mut(index) {
            let row = raw / n;
            let col = raw % n;
            if row < m {
                let mut sum = 0.0;
                let mut inner = 0;
                while inner < k {
                    sum += a[row * k + inner] * b[inner * n + col];
                    inner += 1;
                }
                *value = sum;
            }
        }
    }
}

pub(crate) fn eval(tensor: &Tensor, device: usize) -> Result<Vec<f32>> {
    let context = CudaContext::new(device).map_err(cuda_error)?;
    let stream = context.default_stream();
    let module = kernels::load(&context).map_err(cuda_error)?;
    let output = eval_node(tensor, &stream, &module, &mut HashMap::new())?;
    output.to_host_vec(&stream).map_err(cuda_error)
}

fn eval_node(
    tensor: &Tensor,
    stream: &Arc<cuda_core::CudaStream>,
    module: &kernels::LoadedModule,
    cache: &mut HashMap<u64, Arc<DeviceBuffer<f32>>>,
) -> Result<Arc<DeviceBuffer<f32>>> {
    if let Some(value) = cache.get(&tensor.node.id) {
        return Ok(value.clone());
    }
    let output = match &tensor.node.op {
        Op::Data(data) => DeviceBuffer::from_host(stream, data).map_err(cuda_error)?,
        Op::Placeholder(slot) => {
            return Err(Error::Execution(format!(
                "unbound CUDA JIT placeholder {slot}"
            )));
        }
        Op::Add(a, b) | Op::Mul(a, b) => {
            let a = eval_node(a, stream, module, cache)?;
            let b = eval_node(b, stream, module, cache)?;
            let mut output =
                DeviceBuffer::<f32>::zeroed(stream, tensor.numel()).map_err(cuda_error)?;
            let config = LaunchConfig::for_num_elems(
                u32::try_from(tensor.numel())
                    .map_err(|_| Error::Execution("tensor is too large for a CUDA grid".into()))?,
            );
            // SAFETY: all buffers have the same validated element count; the
            // kernel guards its 1D index with the output slice length.
            unsafe {
                match &tensor.node.op {
                    Op::Add(_, _) => module.add(stream, config, &a, &b, &mut output),
                    Op::Mul(_, _) => module.mul(stream, config, &a, &b, &mut output),
                    _ => unreachable!(),
                }
            }
            .map_err(cuda_error)?;
            output
        }
        Op::Relu(input) => {
            let input = eval_node(input, stream, module, cache)?;
            let mut output =
                DeviceBuffer::<f32>::zeroed(stream, tensor.numel()).map_err(cuda_error)?;
            let config = LaunchConfig::for_num_elems(
                u32::try_from(tensor.numel())
                    .map_err(|_| Error::Execution("tensor is too large for a CUDA grid".into()))?,
            );
            // SAFETY: input and output lengths are identical and the kernel
            // guards every write through DisjointSlice.
            unsafe { module.relu(stream, config, &input, &mut output) }.map_err(cuda_error)?;
            output
        }
        Op::MatMul(a, b) => {
            let a_buffer = eval_node(a, stream, module, cache)?;
            let b_buffer = eval_node(b, stream, module, cache)?;
            let mut output =
                DeviceBuffer::<f32>::zeroed(stream, tensor.numel()).map_err(cuda_error)?;
            let config = LaunchConfig::for_num_elems(
                u32::try_from(tensor.numel())
                    .map_err(|_| Error::Execution("tensor is too large for a CUDA grid".into()))?,
            );
            // SAFETY: Tensor::matmul validates rank, aligned inner dimensions,
            // and output size; the 1D kernel guards every output access.
            unsafe {
                module.matmul(
                    stream,
                    config,
                    a.shape()[0],
                    b.shape()[1],
                    a.shape()[1],
                    &a_buffer,
                    &b_buffer,
                    &mut output,
                )
            }
            .map_err(cuda_error)?;
            output
        }
    };
    let output = Arc::new(output);
    cache.insert(tensor.node.id, output.clone());
    Ok(output)
}

fn cuda_error(error: impl std::fmt::Display) -> Error {
    Error::Execution(error.to_string())
}
