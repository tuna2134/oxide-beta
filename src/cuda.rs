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

    #[kernel]
    pub fn mul_backward(
        gradient: &[f32],
        other: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(value) = output.get_mut(index) {
            *value = gradient[raw] * other[raw];
        }
    }

    #[kernel]
    pub fn relu_backward(
        gradient: &[f32],
        input: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(value) = output.get_mut(index) {
            *value = if input[raw] > 0.0 { gradient[raw] } else { 0.0 };
        }
    }

    #[kernel]
    pub fn matmul_left_backward(
        rows: usize,
        inner: usize,
        columns: usize,
        gradient: &[f32],
        right: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(value) = output.get_mut(index) {
            let row = raw / inner;
            let inner_index = raw % inner;
            let mut sum = 0.0;
            let mut column = 0;
            while column < columns && row < rows {
                sum += gradient[row * columns + column]
                    * right[inner_index * columns + column];
                column += 1;
            }
            *value = sum;
        }
    }

    #[kernel]
    pub fn matmul_right_backward(
        rows: usize,
        inner: usize,
        columns: usize,
        gradient: &[f32],
        left: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(value) = output.get_mut(index) {
            let inner_index = raw / columns;
            let column = raw % columns;
            let mut sum = 0.0;
            let mut row = 0;
            while row < rows && inner_index < inner {
                sum += left[row * inner + inner_index] * gradient[row * columns + column];
                row += 1;
            }
            *value = sum;
        }
    }

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_input_backward(
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
        gradient: &[f32],
        weight: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(value) = output.get_mut(index) {
            let in_x = raw % width;
            let position = raw / width;
            let in_y = position % height;
            let position = position / height;
            let in_channel = position % in_channels;
            let batch_index = position / in_channels;
            let in_per_group = in_channels / groups;
            let out_per_group = out_channels / groups;
            let group = in_channel / in_per_group;
            let local_channel = in_channel % in_per_group;
            let mut sum = 0.0;
            let mut local_out = 0;
            while local_out < out_per_group && batch_index < batch {
                let out_channel = group * out_per_group + local_out;
                let mut out_y = 0;
                while out_y < out_height {
                    let kernel_y_signed = in_y + padding;
                    let base_y = out_y * stride;
                    if kernel_y_signed >= base_y {
                        let kernel_y = kernel_y_signed - base_y;
                        if kernel_y < kernel_size {
                            let mut out_x = 0;
                            while out_x < out_width {
                                let kernel_x_signed = in_x + padding;
                                let base_x = out_x * stride;
                                if kernel_x_signed >= base_x {
                                    let kernel_x = kernel_x_signed - base_x;
                                    if kernel_x < kernel_size {
                                        let grad_index = ((batch_index * out_channels
                                            + out_channel)
                                            * out_height
                                            + out_y)
                                            * out_width
                                            + out_x;
                                        let weight_index = ((out_channel * in_per_group
                                            + local_channel)
                                            * kernel_size
                                            + kernel_y)
                                            * kernel_size
                                            + kernel_x;
                                        sum += gradient[grad_index] * weight[weight_index];
                                    }
                                }
                                out_x += 1;
                            }
                        }
                    }
                    out_y += 1;
                }
                local_out += 1;
            }
            *value = sum;
        }
    }

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_weight_backward(
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
        gradient: &[f32],
        input: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(value) = output.get_mut(index) {
            let kernel_x = raw % kernel_size;
            let position = raw / kernel_size;
            let kernel_y = position % kernel_size;
            let position = position / kernel_size;
            let in_per_group = in_channels / groups;
            let local_channel = position % in_per_group;
            let out_channel = position / in_per_group;
            let out_per_group = out_channels / groups;
            let group = out_channel / out_per_group;
            let in_channel = group * in_per_group + local_channel;
            let mut sum = 0.0;
            let mut batch_index = 0;
            while batch_index < batch && out_channel < out_channels {
                let mut out_y = 0;
                while out_y < out_height {
                    let padded_y = out_y * stride + kernel_y;
                    if padded_y >= padding && padded_y - padding < height {
                        let in_y = padded_y - padding;
                        let mut out_x = 0;
                        while out_x < out_width {
                            let padded_x = out_x * stride + kernel_x;
                            if padded_x >= padding && padded_x - padding < width {
                                let in_x = padded_x - padding;
                                let input_index = ((batch_index * in_channels + in_channel)
                                    * height
                                    + in_y)
                                    * width
                                    + in_x;
                                let grad_index = ((batch_index * out_channels + out_channel)
                                    * out_height
                                    + out_y)
                                    * out_width
                                    + out_x;
                                sum += input[input_index] * gradient[grad_index];
                            }
                            out_x += 1;
                        }
                    }
                    out_y += 1;
                }
                batch_index += 1;
            }
            *value = sum;
        }
    }

    #[kernel]
    pub fn conv2d_bias_backward(
        batch: usize,
        out_channels: usize,
        spatial: usize,
        gradient: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let channel_index = thread::index_1d();
        let channel = channel_index.get();
        if let Some(value) = output.get_mut(channel_index) {
            let mut sum = 0.0;
            let mut batch_index = 0;
            while batch_index < batch && channel < out_channels {
                let start = (batch_index * out_channels + channel) * spatial;
                let mut position = 0;
                while position < spatial {
                    sum += gradient[start + position];
                    position += 1;
                }
                batch_index += 1;
            }
            *value = sum;
        }
    }

    #[kernel]
    pub fn avg_pool2d_backward(
        channels: usize,
        height: usize,
        width: usize,
        out_height: usize,
        out_width: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        gradient: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(value) = output.get_mut(index) {
            let in_x = raw % width;
            let position = raw / width;
            let in_y = position % height;
            let position = position / height;
            let channel = position % channels;
            let batch_index = position / channels;
            let mut sum = 0.0;
            let mut out_y = 0;
            while out_y < out_height {
                let start_y = out_y * stride_h;
                if in_y >= start_y && in_y < start_y + kernel_h {
                    let mut out_x = 0;
                    while out_x < out_width {
                        let start_x = out_x * stride_w;
                        if in_x >= start_x && in_x < start_x + kernel_w {
                            let grad_index = ((batch_index * channels + channel) * out_height
                                + out_y)
                                * out_width
                                + out_x;
                            sum += gradient[grad_index] / (kernel_h * kernel_w) as f32;
                        }
                        out_x += 1;
                    }
                }
                out_y += 1;
            }
            *value = sum;
        }
    }

    #[kernel]
    pub fn batch_norm_statistics(
        batch: usize,
        channels: usize,
        spatial: usize,
        epsilon: f32,
        input: &[f32],
        mut mean: DisjointSlice<f32>,
        mut inverse_std: DisjointSlice<f32>,
        mut unbiased_variance: DisjointSlice<f32>,
    ) {
        let channel_index = thread::index_1d();
        let channel = channel_index.get();
        if let Some(mean_value) = mean.get_mut(channel_index) {
            let samples = batch * spatial;
            let mut sum = 0.0;
            let mut batch_index = 0;
            while batch_index < batch && channel < channels {
                let start = (batch_index * channels + channel) * spatial;
                let mut position = 0;
                while position < spatial {
                    sum += input[start + position];
                    position += 1;
                }
                batch_index += 1;
            }
            let channel_mean = sum / samples as f32;
            let mut squared_sum = 0.0;
            batch_index = 0;
            while batch_index < batch && channel < channels {
                let start = (batch_index * channels + channel) * spatial;
                let mut position = 0;
                while position < spatial {
                    let centered = input[start + position] - channel_mean;
                    squared_sum += centered * centered;
                    position += 1;
                }
                batch_index += 1;
            }
            let variance = squared_sum / samples as f32;
            *mean_value = channel_mean;
            if let Some(value) = inverse_std.get_mut(channel_index) {
                *value = (variance + epsilon).sqrt().recip();
            }
            if let Some(value) = unbiased_variance.get_mut(channel_index) {
                *value = squared_sum / (samples - 1) as f32;
            }
        }
    }

    #[kernel]
    pub fn batch_norm_apply(
        channels: usize,
        spatial: usize,
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        mean: &[f32],
        inverse_std: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(value) = output.get_mut(index) {
            let channel = (raw / spatial) % channels;
            *value = (input[raw] - mean[channel]) * inverse_std[channel] * weight[channel]
                + bias[channel];
        }
    }

    #[kernel]
    pub fn batch_norm_input_backward(
        batch: usize,
        channels: usize,
        spatial: usize,
        training: bool,
        gradient: &[f32],
        input: &[f32],
        weight: &[f32],
        mean: &[f32],
        inverse_std: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(value) = output.get_mut(index) {
            let channel = (raw / spatial) % channels;
            if !training {
                *value = gradient[raw] * weight[channel] * inverse_std[channel];
                return;
            }
            let samples = batch * spatial;
            let normalized = (input[raw] - mean[channel]) * inverse_std[channel];
            let mut gradient_sum = 0.0;
            let mut normalized_gradient_sum = 0.0;
            let mut batch_index = 0;
            while batch_index < batch {
                let start = (batch_index * channels + channel) * spatial;
                let mut position = 0;
                while position < spatial {
                    let item = start + position;
                    let item_normalized = (input[item] - mean[channel]) * inverse_std[channel];
                    gradient_sum += gradient[item];
                    normalized_gradient_sum += gradient[item] * item_normalized;
                    position += 1;
                }
                batch_index += 1;
            }
            let samples_f32 = samples as f32;
            *value = weight[channel] * inverse_std[channel] / samples_f32
                * (samples_f32 * gradient[raw]
                    - gradient_sum
                    - normalized * normalized_gradient_sum);
        }
    }

    #[kernel]
    pub fn batch_norm_affine_backward(
        batch: usize,
        channels: usize,
        spatial: usize,
        gradient: &[f32],
        input: &[f32],
        mean: &[f32],
        inverse_std: &[f32],
        mut weight_gradient: DisjointSlice<f32>,
        mut bias_gradient: DisjointSlice<f32>,
    ) {
        let channel_index = thread::index_1d();
        let channel = channel_index.get();
        if let Some(weight_value) = weight_gradient.get_mut(channel_index) {
            let mut weight_sum = 0.0;
            let mut bias_sum = 0.0;
            let mut batch_index = 0;
            while batch_index < batch && channel < channels {
                let start = (batch_index * channels + channel) * spatial;
                let mut position = 0;
                while position < spatial {
                    let item = start + position;
                    let normalized = (input[item] - mean[channel]) * inverse_std[channel];
                    weight_sum += gradient[item] * normalized;
                    bias_sum += gradient[item];
                    position += 1;
                }
                batch_index += 1;
            }
            *weight_value = weight_sum;
            if let Some(bias_value) = bias_gradient.get_mut(channel_index) {
                *bias_value = bias_sum;
            }
        }
    }

    #[kernel]
    pub fn cross_entropy_forward(
        batch: usize,
        classes: usize,
        logits: &[f32],
        targets: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        if let Some(value) = output.get_mut(index) {
            if index.get() != 0 {
                return;
            }
            let mut loss = 0.0;
            let mut batch_index = 0;
            while batch_index < batch {
                let start = batch_index * classes;
                let mut maximum = f32::NEG_INFINITY;
                let mut class = 0;
                while class < classes {
                    maximum = maximum.max(logits[start + class]);
                    class += 1;
                }
                let mut normalizer = 0.0;
                class = 0;
                while class < classes {
                    normalizer += (logits[start + class] - maximum).exp();
                    class += 1;
                }
                let target = targets[batch_index] as usize;
                if target >= classes {
                    *value = f32::NAN;
                    return;
                }
                loss += normalizer.ln() + maximum - logits[start + target];
                batch_index += 1;
            }
            *value = loss / batch as f32;
        }
    }

    #[kernel]
    pub fn cross_entropy_backward(
        batch: usize,
        classes: usize,
        logits: &[f32],
        targets: &[f32],
        outer_gradient: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(value) = output.get_mut(index) {
            let batch_index = raw / classes;
            let class_index = raw % classes;
            let start = batch_index * classes;
            let mut maximum = f32::NEG_INFINITY;
            let mut class = 0;
            while class < classes && batch_index < batch {
                maximum = maximum.max(logits[start + class]);
                class += 1;
            }
            let mut normalizer = 0.0;
            class = 0;
            while class < classes && batch_index < batch {
                normalizer += (logits[start + class] - maximum).exp();
                class += 1;
            }
            let target = targets[batch_index] as usize;
            if target >= classes {
                *value = f32::NAN;
                return;
            }
            let probability = (logits[raw] - maximum).exp() / normalizer;
            *value = outer_gradient[0] / batch as f32
                * (probability - if class_index == target { 1.0 } else { 0.0 });
        }
    }

    #[kernel]
    pub fn accumulate(a: &[f32], b: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(value) = output.get_mut(index) {
            *value = a[raw] + b[raw];
        }
    }

    #[kernel]
    pub fn batch_norm_running_inverse_std(
        epsilon: f32,
        variance: &[f32],
        mut inverse_std: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(value) = inverse_std.get_mut(index) {
            *value = (variance[raw] + epsilon).sqrt().recip();
        }
    }

    #[kernel]
    pub fn batch_norm_update_running(
        momentum: f32,
        batch_mean: &[f32],
        batch_variance: &[f32],
        old_mean: &[f32],
        old_variance: &[f32],
        mut new_mean: DisjointSlice<f32>,
        mut new_variance: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(mean) = new_mean.get_mut(index) {
            *mean = (1.0 - momentum) * old_mean[raw] + momentum * batch_mean[raw];
            if let Some(variance) = new_variance.get_mut(index) {
                *variance = (1.0 - momentum) * old_variance[raw]
                    + momentum * batch_variance[raw];
            }
        }
    }

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn adamw(
        learning_rate: f32,
        weight_decay: f32,
        beta1: f32,
        beta2: f32,
        first_correction: f32,
        second_correction: f32,
        epsilon: f32,
        parameter: &[f32],
        gradient: &[f32],
        first_moment: &[f32],
        second_moment: &[f32],
        mut new_parameter: DisjointSlice<f32>,
        mut new_first_moment: DisjointSlice<f32>,
        mut new_second_moment: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(value) = new_parameter.get_mut(index) {
            let first = beta1 * first_moment[raw] + (1.0 - beta1) * gradient[raw];
            let second = beta2 * second_moment[raw]
                + (1.0 - beta2) * gradient[raw] * gradient[raw];
            let normalized =
                (first / first_correction) / ((second / second_correction).sqrt() + epsilon);
            *value = parameter[raw]
                - learning_rate * (normalized + weight_decay * parameter[raw]);
            if let Some(moment) = new_first_moment.get_mut(index) {
                *moment = first;
            }
            if let Some(moment) = new_second_moment.get_mut(index) {
                *moment = second;
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
        Op::CrossEntropy { .. } | Op::BatchNorm2d { .. } => {
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
