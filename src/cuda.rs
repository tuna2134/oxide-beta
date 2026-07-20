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

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d(
        batch: usize,
        in_channels: usize,
        height: usize,
        width: usize,
        out_channels: usize,
        out_height: usize,
        out_width: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        groups: usize,
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(value) = output.get_mut(index) {
            let out_x = raw % out_width;
            let position = raw / out_width;
            let out_y = position % out_height;
            let position = position / out_height;
            let out_channel = position % out_channels;
            let batch_index = position / out_channels;
            if batch_index >= batch {
                return;
            }
            let in_per_group = in_channels / groups;
            let out_per_group = out_channels / groups;
            let group = out_channel / out_per_group;
            let mut sum = bias[out_channel];
            let mut local_channel = 0;
            while local_channel < in_per_group {
                let in_channel = group * in_per_group + local_channel;
                let mut kernel_y = 0;
                while kernel_y < kernel_size {
                    let padded_y = out_y * stride + kernel_y;
                    if padded_y >= padding && padded_y - padding < height {
                        let in_y = padded_y - padding;
                        let mut kernel_x = 0;
                        while kernel_x < kernel_size {
                            let padded_x = out_x * stride + kernel_x;
                            if padded_x >= padding && padded_x - padding < width {
                                let in_x = padded_x - padding;
                                let input_index =
                                    ((batch_index * in_channels + in_channel) * height + in_y)
                                        * width
                                        + in_x;
                                let weight_index = ((out_channel * in_per_group + local_channel)
                                    * kernel_size
                                    + kernel_y)
                                    * kernel_size
                                    + kernel_x;
                                sum += input[input_index] * weight[weight_index];
                            }
                            kernel_x += 1;
                        }
                    }
                    kernel_y += 1;
                }
                local_channel += 1;
            }
            *value = sum;
        }
    }

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn avg_pool2d(
        channels: usize,
        height: usize,
        width: usize,
        out_height: usize,
        out_width: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        input: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(value) = output.get_mut(index) {
            let out_x = raw % out_width;
            let position = raw / out_width;
            let out_y = position % out_height;
            let position = position / out_height;
            let channel = position % channels;
            let batch_index = position / channels;
            let mut sum = 0.0;
            let mut kernel_y = 0;
            while kernel_y < kernel_h {
                let mut kernel_x = 0;
                while kernel_x < kernel_w {
                    let in_y = out_y * stride_h + kernel_y;
                    let in_x = out_x * stride_w + kernel_x;
                    let input_index =
                        ((batch_index * channels + channel) * height + in_y) * width + in_x;
                    sum += input[input_index];
                    kernel_x += 1;
                }
                kernel_y += 1;
            }
            *value = sum / (kernel_h * kernel_w) as f32;
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
        Op::Conv2d {
            input,
            weight,
            bias,
            stride,
            padding,
            groups,
        } => {
            let input_buffer = eval_node(input, stream, module, cache)?;
            let weight_buffer = eval_node(weight, stream, module, cache)?;
            let bias_buffer = eval_node(bias, stream, module, cache)?;
            let mut output =
                DeviceBuffer::<f32>::zeroed(stream, tensor.numel()).map_err(cuda_error)?;
            let config = launch_config(tensor.numel())?;
            // SAFETY: Tensor::conv2d validates all NCHW/OIHW dimensions,
            // grouping, output size, and the kernel guards the output slice.
            unsafe {
                module.conv2d(
                    stream,
                    config,
                    input.shape()[0],
                    input.shape()[1],
                    input.shape()[2],
                    input.shape()[3],
                    weight.shape()[0],
                    tensor.shape()[2],
                    tensor.shape()[3],
                    weight.shape()[2],
                    *stride,
                    *padding,
                    *groups,
                    &input_buffer,
                    &weight_buffer,
                    &bias_buffer,
                    &mut output,
                )
            }
            .map_err(cuda_error)?;
            output
        }
        Op::AvgPool2d {
            input,
            kernel,
            stride,
        } => {
            let input_buffer = eval_node(input, stream, module, cache)?;
            let mut output =
                DeviceBuffer::<f32>::zeroed(stream, tensor.numel()).map_err(cuda_error)?;
            let config = launch_config(tensor.numel())?;
            // SAFETY: avg_pool2d validates the kernel, stride, and output
            // extent; the kernel guards every output write.
            unsafe {
                module.avg_pool2d(
                    stream,
                    config,
                    input.shape()[1],
                    input.shape()[2],
                    input.shape()[3],
                    tensor.shape()[2],
                    tensor.shape()[3],
                    kernel[0],
                    kernel[1],
                    stride[0],
                    stride[1],
                    &input_buffer,
                    &mut output,
                )
            }
            .map_err(cuda_error)?;
            output
        }
        Op::Reshape(input) => return eval_node(input, stream, module, cache),
        Op::CrossEntropy { .. } => {
            let host_value = crate::tensor::eval_cpu(tensor, &mut HashMap::new(), None)?;
            DeviceBuffer::from_host(stream, &host_value).map_err(cuda_error)?
        }
    };
    let output = Arc::new(output);
    cache.insert(tensor.node.id, output.clone());
    Ok(output)
}

fn launch_config(numel: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig::for_num_elems(u32::try_from(numel).map_err(
        |_| Error::Execution("tensor is too large for a CUDA grid".into()),
    )?))
}

fn cuda_error(error: impl std::fmt::Display) -> Error {
    Error::Execution(error.to_string())
}
