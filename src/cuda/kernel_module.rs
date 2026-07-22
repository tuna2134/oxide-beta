use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub mod module {
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
        static mut WEIGHT_TILE: SharedArray<f32, 256> = SharedArray::UNINIT;

        let thread_index = thread::threadIdx_x() as usize;
        let block_index = thread::blockIdx_x() as usize;
        let spatial = out_height * out_width;
        let spatial_tiles = (spatial + 255) / 256;
        let plane = block_index / spatial_tiles;
        let output_position = (block_index % spatial_tiles) * 256 + thread_index;
        let out_channel = plane % out_channels;
        let batch_index = plane / out_channels;
        let active = batch_index < batch && output_position < spatial;
        let out_y = output_position / out_width;
        let out_x = output_position % out_width;
        let in_per_group = in_channels / groups;
        let out_per_group = out_channels / groups;
        let group = out_channel / out_per_group;
        let reduction = in_per_group * kernel_size * kernel_size;
        let mut sum = if active { bias[out_channel] } else { 0.0 };
        let mut tile_start = 0;
        while tile_start < reduction {
            let reduction_index = tile_start + thread_index;
            // SAFETY: threadIdx.x is unique in [0, 256), matching the fixed
            // launch shape. Every slot is initialized before the barrier.
            unsafe {
                WEIGHT_TILE[thread_index] = if reduction_index < reduction {
                    weight[out_channel * reduction + reduction_index]
                } else {
                    0.0
                };
            }
            thread::sync_threads();
            if active {
                let tile_len = (reduction - tile_start).min(256);
                let mut tile_index = 0;
                while tile_index < tile_len {
                    let reduction_index = tile_start + tile_index;
                    let kernel_x = reduction_index % kernel_size;
                    let position = reduction_index / kernel_size;
                    let kernel_y = position % kernel_size;
                    let local_channel = position / kernel_size;
                    let padded_y = out_y * stride + kernel_y;
                    let padded_x = out_x * stride + kernel_x;
                    if padded_y >= padding
                        && padded_y - padding < height
                        && padded_x >= padding
                        && padded_x - padding < width
                    {
                        let in_channel = group * in_per_group + local_channel;
                        let input_index = ((batch_index * in_channels + in_channel) * height
                            + padded_y
                            - padding)
                            * width
                            + padded_x
                            - padding;
                        // SAFETY: the preceding barrier initialized the full
                        // tile and tile_index is below its fixed capacity.
                        sum += input[input_index] * unsafe { WEIGHT_TILE[tile_index] };
                    }
                    tile_index += 1;
                }
            }
            thread::sync_threads();
            tile_start += 256;
        }
        if active {
            let output_index = plane * spatial + output_position;
            // SAFETY: each block owns one plane/tile and each thread owns one
            // spatial position, so output_index is in-bounds and unique.
            unsafe { *output.get_unchecked_mut(output_index) = sum };
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
    pub fn mul_backward(gradient: &[f32], other: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(value) = output.get_mut(index) {
            *value = gradient[raw] * other[raw];
        }
    }

    #[kernel]
    pub fn relu_backward(gradient: &[f32], input: &[f32], mut output: DisjointSlice<f32>) {
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
                sum += gradient[row * columns + column] * right[inner_index * columns + column];
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
        static mut WEIGHT_TILE: SharedArray<f32, 256> = SharedArray::UNINIT;

        let thread_index = thread::threadIdx_x() as usize;
        let block_index = thread::blockIdx_x() as usize;
        let spatial = height * width;
        let spatial_tiles = (spatial + 255) / 256;
        let plane = block_index / spatial_tiles;
        let input_position = (block_index % spatial_tiles) * 256 + thread_index;
        let in_channel = plane % in_channels;
        let batch_index = plane / in_channels;
        let active = batch_index < batch && input_position < spatial;
        let in_y = input_position / width;
        let in_x = input_position % width;
        let in_per_group = in_channels / groups;
        let out_per_group = out_channels / groups;
        let group = in_channel / in_per_group;
        let local_channel = in_channel % in_per_group;
        let reduction = out_per_group * kernel_size * kernel_size;
        let mut sum = 0.0;
        let mut tile_start = 0;
        while tile_start < reduction {
            let reduction_index = tile_start + thread_index;
            // SAFETY: the 256-thread launch gives each thread a unique shared
            // slot, and all slots are initialized before synchronization.
            unsafe {
                WEIGHT_TILE[thread_index] = if reduction_index < reduction {
                    let kernel_x = reduction_index % kernel_size;
                    let position = reduction_index / kernel_size;
                    let kernel_y = position % kernel_size;
                    let local_out = position / kernel_size;
                    let out_channel = group * out_per_group + local_out;
                    weight[((out_channel * in_per_group + local_channel) * kernel_size + kernel_y)
                        * kernel_size
                        + kernel_x]
                } else {
                    0.0
                };
            }
            thread::sync_threads();
            if active {
                let tile_len = (reduction - tile_start).min(256);
                let mut tile_index = 0;
                while tile_index < tile_len {
                    let reduction_index = tile_start + tile_index;
                    let kernel_x = reduction_index % kernel_size;
                    let position = reduction_index / kernel_size;
                    let kernel_y = position % kernel_size;
                    let local_out = position / kernel_size;
                    let padded_y = in_y + padding;
                    let padded_x = in_x + padding;
                    if padded_y >= kernel_y && padded_x >= kernel_x {
                        let output_y_numerator = padded_y - kernel_y;
                        let output_x_numerator = padded_x - kernel_x;
                        if output_y_numerator % stride == 0 && output_x_numerator % stride == 0 {
                            let out_y = output_y_numerator / stride;
                            let out_x = output_x_numerator / stride;
                            if out_y < out_height && out_x < out_width {
                                let out_channel = group * out_per_group + local_out;
                                let gradient_index = ((batch_index * out_channels + out_channel)
                                    * out_height
                                    + out_y)
                                    * out_width
                                    + out_x;
                                // SAFETY: the tile was initialized before the
                                // barrier and tile_index is below 256.
                                sum +=
                                    gradient[gradient_index] * unsafe { WEIGHT_TILE[tile_index] };
                            }
                        }
                    }
                    tile_index += 1;
                }
            }
            thread::sync_threads();
            tile_start += 256;
        }
        if active {
            let output_index = plane * spatial + input_position;
            // SAFETY: plane/tile/thread mapping is one-to-one and host shape
            // validation guarantees output_index is within the input tensor.
            unsafe { *output.get_unchecked_mut(output_index) = sum };
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
        static mut GRADIENT_TILE: SharedArray<f32, 32> = SharedArray::UNINIT;

        let thread_index = thread::threadIdx_x() as usize;
        let block_index = thread::blockIdx_x() as usize;
        let in_per_group = in_channels / groups;
        let weights_per_output = in_per_group * kernel_size * kernel_size;
        let weight_tiles = (weights_per_output + 31) / 32;
        let out_channel = block_index / weight_tiles;
        let weight_position = (block_index % weight_tiles) * 32 + thread_index;
        let active = out_channel < out_channels && weight_position < weights_per_output;
        let kernel_x = weight_position % kernel_size;
        let position = weight_position / kernel_size;
        let kernel_y = position % kernel_size;
        let local_channel = position / kernel_size;
        let out_per_group = out_channels / groups;
        let group = out_channel / out_per_group;
        let in_channel = group * in_per_group + local_channel;
        let output_spatial = out_height * out_width;
        let reduction = batch * output_spatial;
        let mut sum = 0.0;
        let mut tile_start = 0;
        while tile_start < reduction {
            let reduction_index = tile_start + thread_index;
            // SAFETY: each of the 32 threads initializes its unique slot;
            // out-of-range reduction elements are explicitly zero-filled.
            unsafe {
                GRADIENT_TILE[thread_index] = if reduction_index < reduction {
                    let batch_index = reduction_index / output_spatial;
                    let output_position = reduction_index % output_spatial;
                    gradient[(batch_index * out_channels + out_channel) * output_spatial
                        + output_position]
                } else {
                    0.0
                };
            }
            thread::sync_threads();
            if active {
                let tile_len = (reduction - tile_start).min(32);
                let mut tile_index = 0;
                while tile_index < tile_len {
                    let reduction_index = tile_start + tile_index;
                    let batch_index = reduction_index / output_spatial;
                    let output_position = reduction_index % output_spatial;
                    let out_y = output_position / out_width;
                    let out_x = output_position % out_width;
                    let padded_y = out_y * stride + kernel_y;
                    let padded_x = out_x * stride + kernel_x;
                    if padded_y >= padding
                        && padded_y - padding < height
                        && padded_x >= padding
                        && padded_x - padding < width
                    {
                        let input_index = ((batch_index * in_channels + in_channel) * height
                            + padded_y
                            - padding)
                            * width
                            + padded_x
                            - padding;
                        // SAFETY: all gradient tile slots were initialized and
                        // made visible by the preceding block-wide barrier.
                        sum += input[input_index] * unsafe { GRADIENT_TILE[tile_index] };
                    }
                    tile_index += 1;
                }
            }
            thread::sync_threads();
            tile_start += 32;
        }
        if active {
            let output_index = out_channel * weights_per_output + weight_position;
            // SAFETY: each block owns one out_channel/tile and each thread one
            // weight, making output_index unique and in bounds.
            unsafe { *output.get_unchecked_mut(output_index) = sum };
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
            if let Some(value) = inverse_std.get_mut(thread::index_1d()) {
                *value = 1.0 / core::intrinsics::sqrtf32(variance + epsilon);
            }
            if let Some(value) = unbiased_variance.get_mut(thread::index_1d()) {
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
        channels: usize,
        spatial: usize,
        samples: usize,
        training: bool,
        gradient: &[f32],
        input: &[f32],
        weight: &[f32],
        mean: &[f32],
        inverse_std: &[f32],
        weight_gradient: &[f32],
        bias_gradient: &[f32],
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
            let normalized = (input[raw] - mean[channel]) * inverse_std[channel];
            let samples_f32 = samples as f32;
            *value = weight[channel] * inverse_std[channel] / samples_f32
                * (samples_f32 * gradient[raw]
                    - bias_gradient[channel]
                    - normalized * weight_gradient[channel]);
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
            if let Some(bias_value) = bias_gradient.get_mut(thread::index_1d()) {
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
        let raw = index.get();
        if let Some(value) = output.get_mut(index) {
            if raw != 0 {
                return;
            }
            let mut loss = 0.0;
            let mut batch_index = 0;
            while batch_index < batch {
                let start = batch_index * classes;
                let mut maximum = f32::NEG_INFINITY;
                let mut class = 0;
                while class < classes {
                    let candidate = logits[start + class];
                    if candidate > maximum {
                        maximum = candidate;
                    }
                    class += 1;
                }
                let mut normalizer = 0.0;
                class = 0;
                while class < classes {
                    normalizer += core::intrinsics::expf32(logits[start + class] - maximum);
                    class += 1;
                }
                let target = targets[batch_index] as usize;
                if target >= classes {
                    *value = f32::NAN;
                    return;
                }
                loss += core::intrinsics::logf32(normalizer) + maximum - logits[start + target];
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
                let candidate = logits[start + class];
                if candidate > maximum {
                    maximum = candidate;
                }
                class += 1;
            }
            let mut normalizer = 0.0;
            class = 0;
            while class < classes && batch_index < batch {
                normalizer += core::intrinsics::expf32(logits[start + class] - maximum);
                class += 1;
            }
            let target = targets[batch_index] as usize;
            if target >= classes {
                *value = f32::NAN;
                return;
            }
            let probability = core::intrinsics::expf32(logits[raw] - maximum) / normalizer;
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
            *value = 1.0 / core::intrinsics::sqrtf32(variance[raw] + epsilon);
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
            if let Some(variance) = new_variance.get_mut(thread::index_1d()) {
                *variance = (1.0 - momentum) * old_variance[raw] + momentum * batch_variance[raw];
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
            let second = beta2 * second_moment[raw] + (1.0 - beta2) * gradient[raw] * gradient[raw];
            let normalized = (first / first_correction)
                / (core::intrinsics::sqrtf32(second / second_correction) + epsilon);
            *value = parameter[raw] - learning_rate * (normalized + weight_decay * parameter[raw]);
            if let Some(moment) = new_first_moment.get_mut(thread::index_1d()) {
                *moment = first;
            }
            if let Some(moment) = new_second_moment.get_mut(thread::index_1d()) {
                *moment = second;
            }
        }
    }

    /// Model-independent kernels used by autoregressive inference backends.
    pub mod inference {
        use super::*;

        #[kernel]
        pub fn rms_norm(
            hidden: usize,
            epsilon: f32,
            input: &[f32],
            weight_bf16: &[u16],
            mut output: DisjointSlice<f32>,
        ) {
            static mut SCALE: SharedArray<f32, 1> = SharedArray::UNINIT;
            let row = thread::blockIdx_x() as usize;
            let lane = thread::threadIdx_x() as usize;
            let row_offset = row * hidden;
            if lane == 0 {
                let mut square_sum = 0.0;
                let mut offset = 0;
                while offset < hidden {
                    let x = input[row_offset + offset];
                    square_sum += x * x;
                    offset += 1;
                }
                // SAFETY: lane zero is the sole writer and the block barrier below
                // publishes the value to every lane.
                unsafe {
                    SCALE[0] = 1.0 / core::intrinsics::sqrtf32(square_sum / hidden as f32 + epsilon)
                };
            }
            thread::sync_threads();
            let scale = unsafe { SCALE[0] };
            let mut column = lane;
            while column < hidden {
                let weight = f32::from_bits((weight_bf16[column] as u32) << 16);
                // SAFETY: each lane owns columns congruent to its lane index and
                // the host launches one block for every complete row.
                unsafe {
                    *output.get_unchecked_mut(row_offset + column) =
                        input[row_offset + column] * scale * weight
                };
                column += 256;
            }
        }

        #[kernel]
        pub fn rms_norm_bf16(
            hidden: usize,
            epsilon: f32,
            input: &[f32],
            weight_bf16: &[u16],
            mut output: DisjointSlice<u16>,
        ) {
            static mut SCALE: SharedArray<f32, 1> = SharedArray::UNINIT;
            let row = thread::blockIdx_x() as usize;
            let lane = thread::threadIdx_x() as usize;
            let row_offset = row * hidden;
            if lane == 0 {
                let mut square_sum = 0.0;
                let mut offset = 0;
                while offset < hidden {
                    let x = input[row_offset + offset];
                    square_sum += x * x;
                    offset += 1;
                }
                unsafe {
                    SCALE[0] = 1.0 / core::intrinsics::sqrtf32(square_sum / hidden as f32 + epsilon)
                };
            }
            thread::sync_threads();
            let scale = unsafe { SCALE[0] };
            let mut column = lane;
            while column < hidden {
                let weight = f32::from_bits((weight_bf16[column] as u32) << 16);
                let bits = (input[row_offset + column] * scale * weight).to_bits();
                let rounding_bias = 0x7fff + ((bits >> 16) & 1);
                unsafe {
                    *output.get_unchecked_mut(row_offset + column) =
                        bits.wrapping_add(rounding_bias).wrapping_shr(16) as u16
                };
                column += 256;
            }
        }

        #[kernel]
        pub fn rms_norm_unit(
            hidden: usize,
            epsilon: f32,
            input: &[f32],
            mut output: DisjointSlice<f32>,
        ) {
            static mut SCALE: SharedArray<f32, 1> = SharedArray::UNINIT;
            let row = thread::blockIdx_x() as usize;
            let lane = thread::threadIdx_x() as usize;
            let row_offset = row * hidden;
            if lane == 0 {
                let mut square_sum = 0.0;
                let mut offset = 0;
                while offset < hidden {
                    let x = input[row_offset + offset];
                    square_sum += x * x;
                    offset += 1;
                }
                unsafe {
                    SCALE[0] = 1.0 / core::intrinsics::sqrtf32(square_sum / hidden as f32 + epsilon)
                };
            }
            thread::sync_threads();
            let scale = unsafe { SCALE[0] };
            let mut column = lane;
            while column < hidden {
                unsafe {
                    *output.get_unchecked_mut(row_offset + column) =
                        input[row_offset + column] * scale
                };
                column += 256;
            }
        }

        #[kernel]
        pub fn mul_bf16_scalar(input: &[f32], scalar_bf16: &[u16], mut output: DisjointSlice<f32>) {
            let index = thread::index_1d();
            let raw = index.get();
            if let Some(value) = output.get_mut(index) {
                let scalar = f32::from_bits((scalar_bf16[0] as u32) << 16);
                *value = input[raw] * scalar;
            }
        }

        #[kernel]
        pub fn bf16_to_f32_scaled(
            offset: usize,
            scale: f32,
            input: &[u16],
            mut output: DisjointSlice<f32>,
        ) {
            let index = thread::index_1d();
            let raw = index.get();
            if let Some(value) = output.get_mut(index) {
                *value = f32::from_bits((input[offset + raw] as u32) << 16) * scale;
            }
        }

        #[kernel]
        pub fn bf16_row_scaled_state(
            row_width: usize,
            scale: f32,
            state: &[usize],
            input: &[u16],
            mut output: DisjointSlice<f32>,
        ) {
            let index = thread::index_1d();
            let raw = index.get();
            if let Some(value) = output.get_mut(index) {
                *value = f32::from_bits((input[state[0] * row_width + raw] as u32) << 16) * scale;
            }
        }

        #[kernel]
        pub fn decode_state_update(
            token: usize,
            position: usize,
            mark_seen: bool,
            mut state: DisjointSlice<usize>,
            mut seen: DisjointSlice<u8>,
        ) {
            if thread::index_1d().get() == 0 {
                // SAFETY: host launches one thread, state has two elements, and
                // validated token ids are within the seen-token bitmap.
                unsafe {
                    *state.get_unchecked_mut(0) = token;
                    *state.get_unchecked_mut(1) = position;
                    if mark_seen {
                        *seen.get_unchecked_mut(token) = 1;
                    }
                }
            }
        }

        #[kernel]
        pub fn decode_state_set(token: usize, position: usize, mut state: DisjointSlice<usize>) {
            if thread::index_1d().get() == 0 {
                // SAFETY: host always supplies a two-element state allocation.
                unsafe {
                    *state.get_unchecked_mut(0) = token;
                    *state.get_unchecked_mut(1) = position;
                }
            }
        }

        #[kernel]
        pub fn embedding_rows_bf16(
            hidden: usize,
            scale: f32,
            tokens: &[u32],
            embedding_bf16: &[u16],
            mut output: DisjointSlice<f32>,
        ) {
            let index = thread::index_1d();
            let raw = index.get();
            if let Some(value) = output.get_mut(index) {
                let row = raw / hidden;
                let column = raw % hidden;
                let token = tokens[row] as usize;
                *value =
                    f32::from_bits((embedding_bf16[token * hidden + column] as u32) << 16) * scale;
            }
        }

        #[kernel]
        pub fn f32_to_bf16(input: &[f32], mut output: DisjointSlice<u16>) {
            let index = thread::index_1d();
            let raw = index.get();
            if let Some(value) = output.get_mut(index) {
                // Round-to-nearest-even before discarding the low 16 bits.
                let bits = input[raw].to_bits();
                let rounding_bias = 0x7fff + ((bits >> 16) & 1);
                *value = bits.wrapping_add(rounding_bias).wrapping_shr(16) as u16;
            }
        }

        #[kernel]
        pub fn copy_bf16(output_offset: usize, input: &[u16], mut output: DisjointSlice<u16>) {
            let index = thread::index_1d();
            let raw = index.get();
            if raw < input.len() {
                unsafe { *output.get_unchecked_mut(output_offset + raw) = input[raw] };
            }
        }

        #[kernel]
        pub fn scale_f32(scale: f32, input: &[f32], mut output: DisjointSlice<f32>) {
            let index = thread::index_1d();
            let raw = index.get();
            if let Some(value) = output.get_mut(index) {
                *value = input[raw] * scale;
            }
        }

        #[kernel]
        pub fn slice_f32(offset: usize, input: &[f32], mut output: DisjointSlice<f32>) {
            let index = thread::index_1d();
            let raw = index.get();
            if let Some(value) = output.get_mut(index) {
                *value = input[offset + raw];
            }
        }

        #[kernel]
        pub fn slice_rows_f32(
            input_width: usize,
            output_width: usize,
            column_offset: usize,
            input: &[f32],
            mut output: DisjointSlice<f32>,
        ) {
            let index = thread::index_1d();
            let raw = index.get();
            if let Some(value) = output.get_mut(index) {
                let row = raw / output_width;
                let column = raw % output_width;
                *value = input[row * input_width + column_offset + column];
            }
        }

        #[kernel]
        pub fn gelu_f32(input: &[f32], mut output: DisjointSlice<f32>) {
            let index = thread::index_1d();
            let raw = index.get();
            if let Some(value) = output.get_mut(index) {
                let x = input[raw];
                let inner = 0.797_884_6 * (x + 0.044_715 * x * x * x);
                let tanh = if inner >= 0.0 {
                    let exponential = core::intrinsics::expf32(-2.0 * inner);
                    (1.0 - exponential) / (1.0 + exponential)
                } else {
                    let exponential = core::intrinsics::expf32(2.0 * inner);
                    (exponential - 1.0) / (exponential + 1.0)
                };
                *value = 0.5 * x * (1.0 + tanh);
            }
        }

        #[kernel]
        pub fn cache_write_f32(offset: usize, input: &[f32], mut cache: DisjointSlice<f32>) {
            let index = thread::index_1d();
            let raw = index.get();
            if raw < input.len() {
                // SAFETY: the host checks the destination extent and every
                // thread writes one unique cache element.
                unsafe { *cache.get_unchecked_mut(offset + raw) = input[raw] };
            }
        }

        #[kernel]
        pub fn cache_write_f32_state(
            width: usize,
            cache_capacity: usize,
            state: &[usize],
            input: &[f32],
            mut cache: DisjointSlice<f32>,
        ) {
            let index = thread::index_1d();
            let raw = index.get();
            if raw < width {
                let destination = (state[1] % cache_capacity) * width + raw;
                // SAFETY: cache has `cache_capacity * width` elements and each
                // launched thread writes one distinct `raw` destination.
                unsafe { *cache.get_unchecked_mut(destination) = input[raw] };
            }
        }

        #[kernel]
        pub fn gelu_mul_f32(gate: &[f32], up: &[f32], mut output: DisjointSlice<f32>) {
            let index = thread::index_1d();
            let raw = index.get();
            if let Some(value) = output.get_mut(index) {
                let x = gate[raw];
                let inner = 0.797_884_6 * (x + 0.044_715 * x * x * x);
                let tanh = if inner >= 0.0 {
                    let exponential = core::intrinsics::expf32(-2.0 * inner);
                    (1.0 - exponential) / (1.0 + exponential)
                } else {
                    let exponential = core::intrinsics::expf32(2.0 * inner);
                    (exponential - 1.0) / (exponential + 1.0)
                };
                let gelu = 0.5 * x * (1.0 + tanh);
                *value = gelu * up[raw];
            }
        }
    }

    /// Kernels whose math or memory layout follows the Gemma 4 architecture.
    pub mod gemma4 {
        use super::*;

        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn rope(
            heads: usize,
            head_dim: usize,
            rotary_dim: usize,
            position_offset: usize,
            theta: f32,
            factor: f32,
            input: &[f32],
            mut output: DisjointSlice<f32>,
        ) {
            let index = thread::index_1d();
            let raw = index.get();
            if let Some(value) = output.get_mut(index) {
                let dimension = raw % head_dim;
                let half = head_dim / 2;
                let frequency_index = dimension % half;
                if frequency_index >= rotary_dim / 2 {
                    *value = input[raw];
                    return;
                }
                let token = raw / (heads * head_dim);
                let head_base = raw - dimension;
                let pair = if dimension < half {
                    dimension + half
                } else {
                    dimension - half
                };
                // Proportional RoPE may rotate only a prefix of the frequency
                // lanes, but Transformers still divides the exponent by the full
                // head dimension (e.g. 512, not the 128 rotated dimensions).
                let exponent = -((2 * frequency_index) as f32) / head_dim as f32;
                let frequency = core::intrinsics::powf32(theta, exponent) / factor;
                let angle = (position_offset + token) as f32 * frequency;
                let rotated = if dimension < half {
                    -input[head_base + pair]
                } else {
                    input[head_base + pair]
                };
                *value = input[raw] * core::intrinsics::cosf32(angle)
                    + rotated * core::intrinsics::sinf32(angle);
            }
        }

        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn rope_state(
            heads: usize,
            head_dim: usize,
            rotary_dim: usize,
            theta: f32,
            factor: f32,
            state: &[usize],
            input: &[f32],
            mut output: DisjointSlice<f32>,
        ) {
            let index = thread::index_1d();
            let raw = index.get();
            if let Some(value) = output.get_mut(index) {
                let dimension = raw % head_dim;
                let half = head_dim / 2;
                let frequency_index = dimension % half;
                if frequency_index >= rotary_dim / 2 {
                    *value = input[raw];
                    return;
                }
                let token = raw / (heads * head_dim);
                let head_base = raw - dimension;
                let pair = if dimension < half {
                    dimension + half
                } else {
                    dimension - half
                };
                let exponent = -((2 * frequency_index) as f32) / head_dim as f32;
                let frequency = core::intrinsics::powf32(theta, exponent) / factor;
                let angle = (state[1] + token) as f32 * frequency;
                let rotated = if dimension < half {
                    -input[head_base + pair]
                } else {
                    input[head_base + pair]
                };
                *value = input[raw] * core::intrinsics::cosf32(angle)
                    + rotated * core::intrinsics::sinf32(angle);
            }
        }

        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn gqa_decode(
            heads: usize,
            kv_heads: usize,
            head_dim: usize,
            sequence: usize,
            window: usize,
            cache_start: usize,
            cache_capacity: usize,
            query: &[f32],
            key_cache: &[f32],
            value_cache: &[f32],
            mut output: DisjointSlice<f32>,
        ) {
            let index = thread::index_1d();
            let raw = index.get();
            if let Some(value) = output.get_mut(index) {
                let head = raw / head_dim;
                let dimension = raw % head_dim;
                let kv_head = head / (heads / kv_heads);
                let start = if window == 0 || sequence <= window {
                    0
                } else {
                    sequence - window
                };
                let mut maximum = f32::NEG_INFINITY;
                let mut position = start;
                while position < sequence {
                    let mut score = 0.0;
                    let mut inner = 0;
                    while inner < head_dim {
                        let physical = (cache_start + position) % cache_capacity;
                        score += query[head * head_dim + inner]
                            * key_cache[(physical * kv_heads + kv_head) * head_dim + inner];
                        inner += 1;
                    }
                    if score > maximum {
                        maximum = score;
                    }
                    position += 1;
                }
                let mut normalizer = 0.0;
                let mut weighted = 0.0;
                position = start;
                while position < sequence {
                    let mut score = 0.0;
                    let mut inner = 0;
                    while inner < head_dim {
                        let physical = (cache_start + position) % cache_capacity;
                        score += query[head * head_dim + inner]
                            * key_cache[(physical * kv_heads + kv_head) * head_dim + inner];
                        inner += 1;
                    }
                    let probability = core::intrinsics::expf32(score - maximum);
                    normalizer += probability;
                    let physical = (cache_start + position) % cache_capacity;
                    weighted += probability
                        * value_cache[(physical * kv_heads + kv_head) * head_dim + dimension];
                    position += 1;
                }
                *value = weighted / normalizer;
            }
        }

        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn gqa_decode_state(
            heads: usize,
            kv_heads: usize,
            head_dim: usize,
            window: usize,
            cache_capacity: usize,
            state: &[usize],
            query: &[f32],
            key_cache: &[f32],
            value_cache: &[f32],
            mut output: DisjointSlice<f32>,
        ) {
            let index = thread::index_1d();
            let raw = index.get();
            if let Some(value) = output.get_mut(index) {
                let total_seen = state[1] + 1;
                let sequence = if total_seen < cache_capacity {
                    total_seen
                } else {
                    cache_capacity
                };
                let cache_start = if total_seen > cache_capacity {
                    total_seen % cache_capacity
                } else {
                    0
                };
                let head = raw / head_dim;
                let dimension = raw % head_dim;
                let kv_head = head / (heads / kv_heads);
                let start = if window == 0 || sequence <= window {
                    0
                } else {
                    sequence - window
                };
                let mut maximum = f32::NEG_INFINITY;
                let mut position = start;
                while position < sequence {
                    let physical = (cache_start + position) % cache_capacity;
                    let mut score = 0.0;
                    let mut inner = 0;
                    while inner < head_dim {
                        score += query[head * head_dim + inner]
                            * key_cache[(physical * kv_heads + kv_head) * head_dim + inner];
                        inner += 1;
                    }
                    if score > maximum {
                        maximum = score;
                    }
                    position += 1;
                }
                let mut normalizer = 0.0;
                let mut weighted = 0.0;
                position = start;
                while position < sequence {
                    let physical = (cache_start + position) % cache_capacity;
                    let mut score = 0.0;
                    let mut inner = 0;
                    while inner < head_dim {
                        score += query[head * head_dim + inner]
                            * key_cache[(physical * kv_heads + kv_head) * head_dim + inner];
                        inner += 1;
                    }
                    let probability = core::intrinsics::expf32(score - maximum);
                    normalizer += probability;
                    weighted += probability
                        * value_cache[(physical * kv_heads + kv_head) * head_dim + dimension];
                    position += 1;
                }
                *value = weighted / normalizer;
            }
        }

        /// Single-query GQA specialized for decode. Attention scores are computed
        /// once per head and shared by all head dimensions, instead of recomputing
        /// the same dot product twice for every output element.
        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn gqa_decode_block(
            heads: usize,
            kv_heads: usize,
            head_dim: usize,
            sequence: usize,
            window: usize,
            cache_start: usize,
            cache_capacity: usize,
            query: &[f32],
            key_cache: &[f32],
            value_cache: &[f32],
            mut output: DisjointSlice<f32>,
        ) {
            static mut PROBABILITIES: SharedArray<f32, 4096> = SharedArray::UNINIT;
            static mut NORMALIZER: SharedArray<f32, 1> = SharedArray::UNINIT;
            let head = thread::blockIdx_x() as usize;
            let lane = thread::threadIdx_x() as usize;
            if head >= heads {
                return;
            }
            let kv_head = head / (heads / kv_heads);
            let start = if window == 0 || sequence <= window {
                0
            } else {
                sequence - window
            };
            if lane == 0 {
                let mut maximum = f32::NEG_INFINITY;
                let mut position = start;
                while position < sequence {
                    let physical = (cache_start + position) % cache_capacity;
                    let mut score = 0.0;
                    let mut inner = 0;
                    while inner < head_dim {
                        score += query[head * head_dim + inner]
                            * key_cache[(physical * kv_heads + kv_head) * head_dim + inner];
                        inner += 1;
                    }
                    unsafe { PROBABILITIES[position - start] = score };
                    if score > maximum {
                        maximum = score;
                    }
                    position += 1;
                }
                let mut normalizer = 0.0;
                position = start;
                while position < sequence {
                    let probability = core::intrinsics::expf32(
                        unsafe { PROBABILITIES[position - start] } - maximum,
                    );
                    unsafe { PROBABILITIES[position - start] = probability };
                    normalizer += probability;
                    position += 1;
                }
                unsafe { NORMALIZER[0] = normalizer };
            }
            thread::sync_threads();
            let normalizer = unsafe { NORMALIZER[0] };
            let mut dimension = lane;
            while dimension < head_dim {
                let mut weighted = 0.0;
                let mut position = start;
                while position < sequence {
                    let physical = (cache_start + position) % cache_capacity;
                    weighted += unsafe { PROBABILITIES[position - start] }
                        * value_cache[(physical * kv_heads + kv_head) * head_dim + dimension];
                    position += 1;
                }
                unsafe {
                    *output.get_unchecked_mut(head * head_dim + dimension) = weighted / normalizer
                };
                dimension += 256;
            }
        }

        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn gqa_decode_block_state(
            heads: usize,
            kv_heads: usize,
            head_dim: usize,
            window: usize,
            cache_capacity: usize,
            state: &[usize],
            query: &[f32],
            key_cache: &[f32],
            value_cache: &[f32],
            mut output: DisjointSlice<f32>,
        ) {
            static mut PROBABILITIES: SharedArray<f32, 4096> = SharedArray::UNINIT;
            static mut NORMALIZER: SharedArray<f32, 1> = SharedArray::UNINIT;
            let head = thread::blockIdx_x() as usize;
            let lane = thread::threadIdx_x() as usize;
            if head >= heads {
                return;
            }
            let total_seen = state[1] + 1;
            let sequence = if total_seen < cache_capacity {
                total_seen
            } else {
                cache_capacity
            };
            let cache_start = if total_seen > cache_capacity {
                total_seen % cache_capacity
            } else {
                0
            };
            let kv_head = head / (heads / kv_heads);
            let start = if window == 0 || sequence <= window {
                0
            } else {
                sequence - window
            };
            if lane == 0 {
                let mut maximum = f32::NEG_INFINITY;
                let mut position = start;
                while position < sequence {
                    let physical = (cache_start + position) % cache_capacity;
                    let mut score = 0.0;
                    let mut inner = 0;
                    while inner < head_dim {
                        score += query[head * head_dim + inner]
                            * key_cache[(physical * kv_heads + kv_head) * head_dim + inner];
                        inner += 1;
                    }
                    unsafe { PROBABILITIES[position - start] = score };
                    if score > maximum {
                        maximum = score;
                    }
                    position += 1;
                }
                let mut normalizer = 0.0;
                position = start;
                while position < sequence {
                    let probability = core::intrinsics::expf32(
                        unsafe { PROBABILITIES[position - start] } - maximum,
                    );
                    unsafe { PROBABILITIES[position - start] = probability };
                    normalizer += probability;
                    position += 1;
                }
                unsafe { NORMALIZER[0] = normalizer };
            }
            thread::sync_threads();
            let normalizer = unsafe { NORMALIZER[0] };
            let mut dimension = lane;
            while dimension < head_dim {
                let mut weighted = 0.0;
                let mut position = start;
                while position < sequence {
                    let physical = (cache_start + position) % cache_capacity;
                    weighted += unsafe { PROBABILITIES[position - start] }
                        * value_cache[(physical * kv_heads + kv_head) * head_dim + dimension];
                    position += 1;
                }
                unsafe {
                    *output.get_unchecked_mut(head * head_dim + dimension) = weighted / normalizer
                };
                dimension += 256;
            }
        }

        /// Causal GQA for a complete prompt resident in a contiguous KV cache.
        /// Each block owns one `(token, query_head)` pair and shares its attention
        /// probabilities across all dimensions of that head.
        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn gqa_prefill_block(
            rows: usize,
            heads: usize,
            kv_heads: usize,
            head_dim: usize,
            window: usize,
            query: &[f32],
            key_cache: &[f32],
            value_cache: &[f32],
            mut output: DisjointSlice<f32>,
        ) {
            static mut PROBABILITIES: SharedArray<f32, 4096> = SharedArray::UNINIT;
            static mut NORMALIZER: SharedArray<f32, 1> = SharedArray::UNINIT;
            let block = thread::blockIdx_x() as usize;
            let lane = thread::threadIdx_x() as usize;
            let row = block / heads;
            let head = block % heads;
            if row >= rows {
                return;
            }
            let visible = row + 1;
            let start = if window == 0 || visible <= window {
                0
            } else {
                visible - window
            };
            let kv_head = head / (heads / kv_heads);
            if lane == 0 {
                let mut maximum = f32::NEG_INFINITY;
                let mut position = start;
                while position < visible {
                    let mut score = 0.0;
                    let mut inner = 0;
                    while inner < head_dim {
                        score += query[(row * heads + head) * head_dim + inner]
                            * key_cache[(position * kv_heads + kv_head) * head_dim + inner];
                        inner += 1;
                    }
                    unsafe { PROBABILITIES[position - start] = score };
                    if score > maximum {
                        maximum = score;
                    }
                    position += 1;
                }
                let mut normalizer = 0.0;
                position = start;
                while position < visible {
                    let probability = core::intrinsics::expf32(
                        unsafe { PROBABILITIES[position - start] } - maximum,
                    );
                    unsafe { PROBABILITIES[position - start] = probability };
                    normalizer += probability;
                    position += 1;
                }
                unsafe { NORMALIZER[0] = normalizer };
            }
            thread::sync_threads();
            let normalizer = unsafe { NORMALIZER[0] };
            let mut dimension = lane;
            while dimension < head_dim {
                let mut weighted = 0.0;
                let mut position = start;
                while position < visible {
                    weighted += unsafe { PROBABILITIES[position - start] }
                        * value_cache[(position * kv_heads + kv_head) * head_dim + dimension];
                    position += 1;
                }
                unsafe {
                    *output.get_unchecked_mut((row * heads + head) * head_dim + dimension) =
                        weighted / normalizer
                };
                dimension += 256;
            }
        }
    }

    /// Model-independent token sampling kernels.
    pub mod sampling {
        use super::*;

        #[kernel]
        pub fn mark_seen(token: usize, mut seen: DisjointSlice<u8>) {
            if thread::index_1d().get() == 0 && token < seen.len() {
                unsafe { *seen.get_unchecked_mut(token) = 1 };
            }
        }

        /// Produces sorted local top-k lists, one list per 256-logit block.
        #[kernel]
        pub fn topk_stage1(
            top_k: usize,
            repetition_penalty: f32,
            logits: &[f32],
            seen: &[u8],
            mut scores: DisjointSlice<f32>,
            mut ids: DisjointSlice<f32>,
        ) {
            if thread::threadIdx_x() != 0 {
                return;
            }
            let block = thread::blockIdx_x() as usize;
            let mut best_scores = [f32::NEG_INFINITY; 64];
            let mut best_ids = [0_u32; 64];
            let start = block * 256;
            let end = (start + 256).min(logits.len());
            let mut index = start;
            while index < end {
                let mut score = logits[index];
                if seen[index] != 0 && repetition_penalty > 1.0 {
                    score = if score >= 0.0 {
                        score / repetition_penalty
                    } else {
                        score * repetition_penalty
                    };
                }
                if score == score && score > best_scores[top_k - 1] {
                    let mut position = top_k - 1;
                    while position > 0 && score > best_scores[position - 1] {
                        best_scores[position] = best_scores[position - 1];
                        best_ids[position] = best_ids[position - 1];
                        position -= 1;
                    }
                    best_scores[position] = score;
                    best_ids[position] = index as u32;
                }
                index += 1;
            }
            let mut rank = 0;
            while rank < top_k {
                let output_index = block * top_k + rank;
                unsafe {
                    *scores.get_unchecked_mut(output_index) = best_scores[rank];
                    *ids.get_unchecked_mut(output_index) = best_ids[rank] as f32;
                }
                rank += 1;
            }
        }

        /// Merges the block-local lists into the final global top-k list.
        #[kernel]
        pub fn topk_stage2(
            top_k: usize,
            input_scores: &[f32],
            input_ids: &[f32],
            mut scores: DisjointSlice<f32>,
            mut ids: DisjointSlice<f32>,
        ) {
            if thread::index_1d().get() != 0 {
                return;
            }
            let mut best_scores = [f32::NEG_INFINITY; 64];
            let mut best_ids = [0_u32; 64];
            let mut index = 0;
            while index < input_scores.len() {
                let score = input_scores[index];
                if score > best_scores[top_k - 1] {
                    let mut position = top_k - 1;
                    while position > 0 && score > best_scores[position - 1] {
                        best_scores[position] = best_scores[position - 1];
                        best_ids[position] = best_ids[position - 1];
                        position -= 1;
                    }
                    best_scores[position] = score;
                    best_ids[position] = input_ids[index] as u32;
                }
                index += 1;
            }
            let mut rank = 0;
            while rank < top_k {
                unsafe {
                    *scores.get_unchecked_mut(rank) = best_scores[rank];
                    *ids.get_unchecked_mut(rank) = best_ids[rank] as f32;
                }
                rank += 1;
            }
        }
    }
}
