// Kernel launch methods are unsafe because cuda-oxide cannot prove device
// buffer lengths and aliasing. This module validates those invariants at each
// call site and documents them with a SAFETY comment.
#![allow(static_mut_refs, unsafe_code)]

use crate::nn::Parameter;
use crate::tensor::{BatchNormState, Op, Tensor};
use crate::{Device, Error, Result};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};

#[cuda_module]
pub(crate) mod kernels {
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

    #[kernel]
    pub fn gemma_rms_norm(
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
    pub fn gemma_rms_norm_bf16(
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
    pub fn gemma_rms_norm_unit(
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
                *output.get_unchecked_mut(row_offset + column) = input[row_offset + column] * scale
            };
            column += 256;
        }
    }

    #[kernel]
    pub fn gemma_mul_bf16_scalar(
        input: &[f32],
        scalar_bf16: &[u16],
        mut output: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(value) = output.get_mut(index) {
            let scalar = f32::from_bits((scalar_bf16[0] as u32) << 16);
            *value = input[raw] * scalar;
        }
    }

    #[kernel]
    pub fn gemma_bf16_to_f32_scaled(
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
    pub fn gemma_embedding_rows(
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
            *value = f32::from_bits((embedding_bf16[token * hidden + column] as u32) << 16) * scale;
        }
    }

    #[kernel]
    pub fn gemma_f32_to_bf16(input: &[f32], mut output: DisjointSlice<u16>) {
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
    pub fn gemma_add(left: &[f32], right: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(value) = output.get_mut(index) {
            *value = left[raw] + right[raw];
        }
    }

    #[kernel]
    pub fn gemma_mul(left: &[f32], right: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(value) = output.get_mut(index) {
            *value = left[raw] * right[raw];
        }
    }

    #[kernel]
    pub fn gemma_scale(scale: f32, input: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(value) = output.get_mut(index) {
            *value = input[raw] * scale;
        }
    }

    #[kernel]
    pub fn gemma_slice(offset: usize, input: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(value) = output.get_mut(index) {
            *value = input[offset + raw];
        }
    }

    #[kernel]
    pub fn gemma_slice_rows(
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
    pub fn gemma_gelu(input: &[f32], mut output: DisjointSlice<f32>) {
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
    pub fn gemma_cache_write(offset: usize, input: &[f32], mut cache: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let raw = index.get();
        if raw < input.len() {
            // SAFETY: the host checks the destination extent and every
            // thread writes one unique cache element.
            unsafe { *cache.get_unchecked_mut(offset + raw) = input[raw] };
        }
    }

    #[kernel]
    pub fn gemma_gelu_mul(gate: &[f32], up: &[f32], mut output: DisjointSlice<f32>) {
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

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn gemma_rope(
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
    pub fn gemma_gqa_decode(
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

    /// Single-query GQA specialized for decode. Attention scores are computed
    /// once per head and shared by all head dimensions, instead of recomputing
    /// the same dot product twice for every output element.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn gemma_gqa_decode_block(
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
                let probability =
                    core::intrinsics::expf32(unsafe { PROBABILITIES[position - start] } - maximum);
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
    pub fn gemma_gqa_prefill_block(
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
                let probability =
                    core::intrinsics::expf32(unsafe { PROBABILITIES[position - start] } - maximum);
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

    #[kernel]
    pub fn gemma_mark_seen(token: usize, mut seen: DisjointSlice<u8>) {
        if thread::index_1d().get() == 0 && token < seen.len() {
            unsafe { *seen.get_unchecked_mut(token) = 1 };
        }
    }

    /// Produces sorted local top-k lists, one list per 256-logit block.
    #[kernel]
    pub fn gemma_topk_stage1(
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
    pub fn gemma_topk_stage2(
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

type Buffer = Arc<DeviceBuffer<f32>>;

struct BatchNormDeviceState {
    running_mean: Buffer,
    running_variance: Buffer,
}

#[derive(Clone)]
struct BatchNormDeviceStatistics {
    mean: Buffer,
    inverse_standard_deviation: Buffer,
}

struct AdamMoments {
    first: Buffer,
    second: Buffer,
}

struct Executor {
    _context: Arc<CudaContext>,
    stream: Arc<cuda_core::CudaStream>,
    module: kernels::LoadedModule,
    values: HashMap<u64, Buffer>,
    gradients: HashMap<u64, Buffer>,
    batch_norm_states: HashMap<usize, BatchNormDeviceState>,
    batch_norm_statistics: HashMap<u64, BatchNormDeviceStatistics>,
    adam_moments: HashMap<(u64, u64), AdamMoments>,
    root_gradient: Buffer,
    pending_host_copies: Vec<Arc<[f32]>>,
    pending_host_bytes: usize,
    #[cfg(feature = "cudnn")]
    cudnn: Option<crate::cudnn::Cudnn>,
}

impl Executor {
    fn new(device: usize) -> Result<Self> {
        let context = CudaContext::new(device).map_err(cuda_error)?;
        let stream = context.default_stream();
        let module = kernels::load(&context).map_err(cuda_error)?;
        let root_gradient = Arc::new(DeviceBuffer::from_host(&stream, &[1.0]).map_err(cuda_error)?);
        #[cfg(feature = "cudnn")]
        let cudnn = crate::cudnn::Cudnn::try_new();
        #[cfg(feature = "cudnn")]
        eprintln!(
            "CUDA device {device}: cuDNN {}",
            if cudnn.is_some() {
                "enabled"
            } else {
                "unavailable; using cuda-oxide kernels"
            }
        );
        Ok(Self {
            _context: context,
            stream,
            module,
            values: HashMap::new(),
            gradients: HashMap::new(),
            batch_norm_states: HashMap::new(),
            batch_norm_statistics: HashMap::new(),
            adam_moments: HashMap::new(),
            root_gradient,
            pending_host_copies: Vec::new(),
            pending_host_bytes: 0,
            #[cfg(feature = "cudnn")]
            cudnn,
        })
    }

    fn zero_buffer(&self, len: usize) -> Result<DeviceBuffer<f32>> {
        DeviceBuffer::zeroed(&self.stream, len).map_err(cuda_error)
    }

    fn output_buffer(&self, len: usize) -> Result<DeviceBuffer<f32>> {
        // SAFETY: callers use this only for kernel outputs. Every output
        // element is written on this stream before the buffer is read.
        unsafe { DeviceBuffer::uninitialized_async(&self.stream, len) }.map_err(cuda_error)
    }

    fn upload(&mut self, data: &Arc<[f32]>) -> Result<DeviceBuffer<f32>> {
        const MAX_PENDING_HOST_BYTES: usize = 64 * 1024 * 1024;
        let bytes = data.len().saturating_mul(std::mem::size_of::<f32>());
        if self.pending_host_bytes.saturating_add(bytes) > MAX_PENDING_HOST_BYTES {
            self.synchronize()?;
        }
        let mut output = self.output_buffer(data.len())?;
        // SAFETY: the Arc clone below keeps this exact allocation alive and
        // immutable until synchronize() confirms that the H2D copy finished.
        unsafe { output.copy_from_host_async_unchecked(&self.stream, data) }.map_err(cuda_error)?;
        self.pending_host_copies.push(data.clone());
        self.pending_host_bytes = self.pending_host_bytes.saturating_add(bytes);
        Ok(output)
    }

    fn synchronize(&mut self) -> Result<()> {
        self.stream.synchronize().map_err(cuda_error)?;
        self.pending_host_copies.clear();
        self.pending_host_bytes = 0;
        Ok(())
    }

    fn clear_completed_host_copies(&mut self) {
        self.pending_host_copies.clear();
        self.pending_host_bytes = 0;
    }

    fn cache_value(&mut self, tensor: &Tensor, output: DeviceBuffer<f32>) -> Result<Buffer> {
        let output = Arc::new(output);
        self.values.insert(tensor.node.id, output.clone());
        Ok(output)
    }

    fn eval_node(&mut self, tensor: &Tensor) -> Result<Buffer> {
        if let Some(value) = self.values.get(&tensor.node.id) {
            return Ok(value.clone());
        }
        let output = match &tensor.node.op {
            Op::Data(data) => self.upload(data)?,
            Op::Placeholder(slot) => {
                return Err(Error::Execution(format!(
                    "unbound CUDA JIT placeholder {slot}"
                )));
            }
            Op::Add(a, b) | Op::Mul(a, b) => {
                let a = self.eval_node(a)?;
                let b = self.eval_node(b)?;
                let mut output = self.output_buffer(tensor.numel())?;
                let config = launch_config(tensor.numel())?;
                // SAFETY: Tensor::binary validates equal lengths. Output is a
                // distinct allocation and every kernel write is bounds-checked.
                unsafe {
                    match &tensor.node.op {
                        Op::Add(_, _) => self.module.add(&self.stream, config, &a, &b, &mut output),
                        Op::Mul(_, _) => self.module.mul(&self.stream, config, &a, &b, &mut output),
                        _ => unreachable!(),
                    }
                }
                .map_err(cuda_error)?;
                output
            }
            Op::Relu(input) => {
                let input = self.eval_node(input)?;
                let mut output = self.output_buffer(tensor.numel())?;
                // SAFETY: input and output have tensor.numel() elements and
                // output is not aliased by any input buffer.
                unsafe {
                    self.module.relu(
                        &self.stream,
                        launch_config(tensor.numel())?,
                        &input,
                        &mut output,
                    )
                }
                .map_err(cuda_error)?;
                output
            }
            Op::MatMul(a, b) => {
                let a_buffer = self.eval_node(a)?;
                let b_buffer = self.eval_node(b)?;
                let mut output = self.output_buffer(tensor.numel())?;
                // SAFETY: Tensor::matmul validated M/K/N and buffer lengths;
                // the output kernel guards its one-dimensional index.
                unsafe {
                    self.module.matmul(
                        &self.stream,
                        launch_config(tensor.numel())?,
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
                let input_buffer = self.eval_node(input)?;
                let weight_buffer = self.eval_node(weight)?;
                let bias_buffer = self.eval_node(bias)?;
                let mut output = self.output_buffer(tensor.numel())?;
                #[cfg(feature = "cudnn")]
                if let Some(cudnn) = &mut self.cudnn {
                    let result = cudnn.forward(
                        cudnn_shape(input, weight, tensor, *stride, *padding, *groups),
                        &self.stream,
                        &input_buffer,
                        &weight_buffer,
                        &bias_buffer,
                        &mut output,
                    );
                    match result {
                        Ok(()) => return self.cache_value(tensor, output),
                        Err(error) => {
                            eprintln!("cuDNN disabled after convolution error: {error}");
                            self.cudnn = None;
                        }
                    }
                }
                // SAFETY: Tensor::conv2d validated NCHW/OIHW dimensions,
                // grouping, kernel extent, and output length.
                unsafe {
                    self.module.conv2d(
                        &self.stream,
                        tiled_launch_config(
                            input.shape()[0] * weight.shape()[0],
                            tensor.shape()[2] * tensor.shape()[3],
                        )?,
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
                let input_buffer = self.eval_node(input)?;
                let mut output = self.output_buffer(tensor.numel())?;
                // SAFETY: avg_pool2d validated all spatial extents. The input
                // and output allocations are distinct and correctly sized.
                unsafe {
                    self.module.avg_pool2d(
                        &self.stream,
                        launch_config(tensor.numel())?,
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
            Op::Reshape(input) => return self.eval_node(input),
            Op::CrossEntropy { logits, targets } => {
                let logits_buffer = self.eval_node(logits)?;
                let targets_buffer = self.eval_node(targets)?;
                let mut output = self.output_buffer(1)?;
                // SAFETY: cross_entropy validates [N,C] logits and [N]
                // targets; the kernel checks target class bounds.
                unsafe {
                    self.module.cross_entropy_forward(
                        &self.stream,
                        launch_config(1)?,
                        logits.shape()[0],
                        logits.shape()[1],
                        &logits_buffer,
                        &targets_buffer,
                        &mut output,
                    )
                }
                .map_err(cuda_error)?;
                output
            }
            Op::BatchNorm2d {
                input,
                weight,
                bias,
                state,
                training,
                momentum,
                epsilon,
                ..
            } => self.eval_batch_norm(
                tensor, input, weight, bias, state, *training, *momentum, *epsilon,
            )?,
        };
        let output = Arc::new(output);
        self.values.insert(tensor.node.id, output.clone());
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_batch_norm(
        &mut self,
        tensor: &Tensor,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        state: &Arc<Mutex<BatchNormState>>,
        training: bool,
        momentum: f32,
        epsilon: f32,
    ) -> Result<DeviceBuffer<f32>> {
        let input_buffer = self.eval_node(input)?;
        let weight_buffer = self.eval_node(weight)?;
        let bias_buffer = self.eval_node(bias)?;
        let channels = input.shape()[1];
        let spatial = input.shape()[2] * input.shape()[3];
        let samples = input.shape()[0] * spatial;
        if training && samples <= 1 {
            return Err(Error::Execution(
                "BatchNorm training requires more than one value per channel".into(),
            ));
        }
        let state_key = Arc::as_ptr(state) as usize;
        if !self.batch_norm_states.contains_key(&state_key) {
            let state = state
                .lock()
                .map_err(|_| Error::Execution("BatchNorm state lock was poisoned".into()))?;
            let running_mean = Arc::new(
                DeviceBuffer::from_host(&self.stream, &state.running_mean).map_err(cuda_error)?,
            );
            let running_variance = Arc::new(
                DeviceBuffer::from_host(&self.stream, &state.running_variance)
                    .map_err(cuda_error)?,
            );
            self.batch_norm_states.insert(
                state_key,
                BatchNormDeviceState {
                    running_mean,
                    running_variance,
                },
            );
        }

        let statistics = if training {
            let mut mean = self.output_buffer(channels)?;
            let mut inverse_std = self.output_buffer(channels)?;
            let mut unbiased_variance = self.output_buffer(channels)?;
            // SAFETY: each statistics output has one element per channel;
            // samples > 1 and the NCHW input length was validated.
            unsafe {
                self.module.batch_norm_statistics(
                    &self.stream,
                    launch_config(channels)?,
                    input.shape()[0],
                    channels,
                    spatial,
                    epsilon,
                    &input_buffer,
                    &mut mean,
                    &mut inverse_std,
                    &mut unbiased_variance,
                )
            }
            .map_err(cuda_error)?;
            let old = self
                .batch_norm_states
                .remove(&state_key)
                .ok_or_else(|| Error::Execution("missing BatchNorm CUDA state".into()))?;
            let mut running_mean = self.output_buffer(channels)?;
            let mut running_variance = self.output_buffer(channels)?;
            // SAFETY: all seven buffers contain exactly `channels` values and
            // output allocations do not alias any input allocation.
            unsafe {
                self.module.batch_norm_update_running(
                    &self.stream,
                    launch_config(channels)?,
                    momentum,
                    &mean,
                    &unbiased_variance,
                    &old.running_mean,
                    &old.running_variance,
                    &mut running_mean,
                    &mut running_variance,
                )
            }
            .map_err(cuda_error)?;
            self.batch_norm_states.insert(
                state_key,
                BatchNormDeviceState {
                    running_mean: Arc::new(running_mean),
                    running_variance: Arc::new(running_variance),
                },
            );
            BatchNormDeviceStatistics {
                mean: Arc::new(mean),
                inverse_standard_deviation: Arc::new(inverse_std),
            }
        } else {
            let running = self
                .batch_norm_states
                .get(&state_key)
                .ok_or_else(|| Error::Execution("missing BatchNorm CUDA state".into()))?;
            let mean = running.running_mean.clone();
            let variance = running.running_variance.clone();
            let mut inverse_std = self.output_buffer(channels)?;
            // SAFETY: variance and inverse_std both contain `channels`
            // elements and are separate allocations.
            unsafe {
                self.module.batch_norm_running_inverse_std(
                    &self.stream,
                    launch_config(channels)?,
                    epsilon,
                    &variance,
                    &mut inverse_std,
                )
            }
            .map_err(cuda_error)?;
            BatchNormDeviceStatistics {
                mean,
                inverse_standard_deviation: Arc::new(inverse_std),
            }
        };
        let mut output = self.output_buffer(tensor.numel())?;
        // SAFETY: affine parameters/statistics have one value per channel;
        // input and output have matching NCHW lengths and do not alias.
        unsafe {
            self.module.batch_norm_apply(
                &self.stream,
                launch_config(tensor.numel())?,
                channels,
                spatial,
                &input_buffer,
                &weight_buffer,
                &bias_buffer,
                &statistics.mean,
                &statistics.inverse_standard_deviation,
                &mut output,
            )
        }
        .map_err(cuda_error)?;
        self.batch_norm_statistics
            .insert(tensor.node.id, statistics);
        Ok(output)
    }

    fn accumulate_gradient(&mut self, tensor: &Tensor, incoming: &Buffer) -> Result<()> {
        let gradient = if let Some(current) = self.gradients.remove(&tensor.node.id) {
            let mut output = self.output_buffer(tensor.numel())?;
            // SAFETY: current, incoming, and output have tensor.numel()
            // elements; output is a distinct allocation.
            unsafe {
                self.module.accumulate(
                    &self.stream,
                    launch_config(tensor.numel())?,
                    &current,
                    incoming,
                    &mut output,
                )
            }
            .map_err(cuda_error)?;
            Arc::new(output)
        } else {
            incoming.clone()
        };
        self.gradients.insert(tensor.node.id, gradient);
        Ok(())
    }

    fn propagate_node(&mut self, tensor: &Tensor, gradient: Buffer) -> Result<()> {
        match &tensor.node.op {
            Op::Data(_) | Op::Placeholder(_) => Ok(()),
            Op::Add(left, right) => {
                self.accumulate_gradient(left, &gradient)?;
                self.accumulate_gradient(right, &gradient)
            }
            Op::Mul(left, right) => {
                let left_value = self.eval_node(left)?;
                let right_value = self.eval_node(right)?;
                let mut left_gradient = self.output_buffer(left.numel())?;
                let mut right_gradient = self.output_buffer(right.numel())?;
                // SAFETY: elementwise operands and gradients all have the
                // same validated length; outputs are distinct allocations.
                unsafe {
                    self.module.mul_backward(
                        &self.stream,
                        launch_config(left.numel())?,
                        &gradient,
                        &right_value,
                        &mut left_gradient,
                    )
                }
                .map_err(cuda_error)?;
                // SAFETY: same invariant as above; this writes the other
                // distinct gradient allocation.
                unsafe {
                    self.module.mul_backward(
                        &self.stream,
                        launch_config(right.numel())?,
                        &gradient,
                        &left_value,
                        &mut right_gradient,
                    )
                }
                .map_err(cuda_error)?;
                self.accumulate_gradient(left, &Arc::new(left_gradient))?;
                self.accumulate_gradient(right, &Arc::new(right_gradient))
            }
            Op::Relu(input) => {
                let input_value = self.eval_node(input)?;
                let mut input_gradient = self.output_buffer(input.numel())?;
                // SAFETY: all buffers have input.numel() elements and output
                // does not alias the input buffers.
                unsafe {
                    self.module.relu_backward(
                        &self.stream,
                        launch_config(input.numel())?,
                        &gradient,
                        &input_value,
                        &mut input_gradient,
                    )
                }
                .map_err(cuda_error)?;
                self.accumulate_gradient(input, &Arc::new(input_gradient))
            }
            Op::Reshape(input) => self.accumulate_gradient(input, &gradient),
            Op::MatMul(left, right) => self.backward_matmul(left, right, &gradient),
            Op::Conv2d {
                input,
                weight,
                bias,
                stride,
                padding,
                groups,
            } => self.backward_conv2d(input, weight, bias, *stride, *padding, *groups, &gradient),
            Op::AvgPool2d {
                input,
                kernel,
                stride,
            } => self.backward_avg_pool(input, *kernel, *stride, &gradient),
            Op::CrossEntropy { logits, targets } => {
                let logits_value = self.eval_node(logits)?;
                let targets_value = self.eval_node(targets)?;
                let mut logits_gradient = self.output_buffer(logits.numel())?;
                // SAFETY: cross_entropy validated [N,C]/[N] shapes and the
                // kernel guards target classes before indexing logits.
                unsafe {
                    self.module.cross_entropy_backward(
                        &self.stream,
                        launch_config(logits.numel())?,
                        logits.shape()[0],
                        logits.shape()[1],
                        &logits_value,
                        &targets_value,
                        &gradient,
                        &mut logits_gradient,
                    )
                }
                .map_err(cuda_error)?;
                self.accumulate_gradient(logits, &Arc::new(logits_gradient))
            }
            Op::BatchNorm2d {
                input,
                weight,
                bias,
                training,
                ..
            } => self.backward_batch_norm(tensor, input, weight, bias, *training, &gradient),
        }
    }

    fn backward_matmul(&mut self, left: &Tensor, right: &Tensor, gradient: &Buffer) -> Result<()> {
        let left_value = self.eval_node(left)?;
        let right_value = self.eval_node(right)?;
        let rows = left.shape()[0];
        let inner = left.shape()[1];
        let columns = right.shape()[1];
        let mut left_gradient = self.output_buffer(left.numel())?;
        let mut right_gradient = self.output_buffer(right.numel())?;
        // SAFETY: matmul shape validation establishes all M/K/N buffer
        // lengths; each kernel writes to a distinct, correctly sized output.
        unsafe {
            self.module.matmul_left_backward(
                &self.stream,
                launch_config(left.numel())?,
                rows,
                inner,
                columns,
                gradient,
                &right_value,
                &mut left_gradient,
            )
        }
        .map_err(cuda_error)?;
        // SAFETY: same validated M/K/N dimensions; the right-gradient
        // allocation is distinct from all inputs.
        unsafe {
            self.module.matmul_right_backward(
                &self.stream,
                launch_config(right.numel())?,
                rows,
                inner,
                columns,
                gradient,
                &left_value,
                &mut right_gradient,
            )
        }
        .map_err(cuda_error)?;
        self.accumulate_gradient(left, &Arc::new(left_gradient))?;
        self.accumulate_gradient(right, &Arc::new(right_gradient))
    }

    #[allow(clippy::too_many_arguments)]
    fn backward_conv2d(
        &mut self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        stride: usize,
        padding: usize,
        groups: usize,
        gradient: &Buffer,
    ) -> Result<()> {
        let input_value = self.eval_node(input)?;
        let weight_value = self.eval_node(weight)?;
        let out_channels = weight.shape()[0];
        let out_height = (input.shape()[2] + 2 * padding - weight.shape()[2]) / stride + 1;
        let out_width = (input.shape()[3] + 2 * padding - weight.shape()[3]) / stride + 1;
        let mut input_gradient = self.output_buffer(input.numel())?;
        let mut weight_gradient = self.output_buffer(weight.numel())?;
        let mut bias_gradient = self.output_buffer(bias.numel())?;
        #[cfg(feature = "cudnn")]
        if let Some(cudnn) = &mut self.cudnn {
            let result = cudnn.backward(
                cudnn_shape_from_parts(
                    input, weight, out_height, out_width, stride, padding, groups,
                ),
                &self.stream,
                &input_value,
                &weight_value,
                gradient,
                &mut input_gradient,
                &mut weight_gradient,
                &mut bias_gradient,
            );
            match result {
                Ok(()) => {
                    self.accumulate_gradient(input, &Arc::new(input_gradient))?;
                    self.accumulate_gradient(weight, &Arc::new(weight_gradient))?;
                    return self.accumulate_gradient(bias, &Arc::new(bias_gradient));
                }
                Err(error) => {
                    eprintln!("cuDNN disabled after convolution backward error: {error}");
                    self.cudnn = None;
                }
            }
        }
        // SAFETY: conv2d validated all NCHW/OIHW/group dimensions. Each
        // backward kernel writes one independent output element.
        unsafe {
            self.module.conv2d_input_backward(
                &self.stream,
                tiled_launch_config(
                    input.shape()[0] * input.shape()[1],
                    input.shape()[2] * input.shape()[3],
                )?,
                input.shape()[0],
                input.shape()[1],
                input.shape()[2],
                input.shape()[3],
                out_channels,
                out_height,
                out_width,
                weight.shape()[2],
                stride,
                padding,
                groups,
                gradient,
                &weight_value,
                &mut input_gradient,
            )
        }
        .map_err(cuda_error)?;
        // SAFETY: the convolution dimensions are unchanged and this kernel
        // writes only the distinct weight-gradient allocation.
        unsafe {
            self.module.conv2d_weight_backward(
                &self.stream,
                tiled_launch_config_with_tile(
                    weight.shape()[0],
                    weight.shape()[1] * weight.shape()[2] * weight.shape()[3],
                    32,
                )?,
                input.shape()[0],
                input.shape()[1],
                input.shape()[2],
                input.shape()[3],
                out_channels,
                out_height,
                out_width,
                weight.shape()[2],
                stride,
                padding,
                groups,
                gradient,
                &input_value,
                &mut weight_gradient,
            )
        }
        .map_err(cuda_error)?;
        // SAFETY: one thread writes each independent bias-gradient element.
        unsafe {
            self.module.conv2d_bias_backward(
                &self.stream,
                launch_config(bias.numel())?,
                input.shape()[0],
                out_channels,
                out_height * out_width,
                gradient,
                &mut bias_gradient,
            )
        }
        .map_err(cuda_error)?;
        self.accumulate_gradient(input, &Arc::new(input_gradient))?;
        self.accumulate_gradient(weight, &Arc::new(weight_gradient))?;
        self.accumulate_gradient(bias, &Arc::new(bias_gradient))
    }

    fn backward_avg_pool(
        &mut self,
        input: &Tensor,
        kernel: [usize; 2],
        stride: [usize; 2],
        gradient: &Buffer,
    ) -> Result<()> {
        let out_height = (input.shape()[2] - kernel[0]) / stride[0] + 1;
        let out_width = (input.shape()[3] - kernel[1]) / stride[1] + 1;
        let mut input_gradient = self.output_buffer(input.numel())?;
        // SAFETY: avg_pool2d validated kernel/stride/output extents and the
        // backward kernel writes one value per input element.
        unsafe {
            self.module.avg_pool2d_backward(
                &self.stream,
                launch_config(input.numel())?,
                input.shape()[1],
                input.shape()[2],
                input.shape()[3],
                out_height,
                out_width,
                kernel[0],
                kernel[1],
                stride[0],
                stride[1],
                gradient,
                &mut input_gradient,
            )
        }
        .map_err(cuda_error)?;
        self.accumulate_gradient(input, &Arc::new(input_gradient))
    }

    fn backward_batch_norm(
        &mut self,
        tensor: &Tensor,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        training: bool,
        gradient: &Buffer,
    ) -> Result<()> {
        let _ = self.eval_node(tensor)?;
        let input_value = self.eval_node(input)?;
        let weight_value = self.eval_node(weight)?;
        let statistics = self
            .batch_norm_statistics
            .get(&tensor.node.id)
            .cloned()
            .ok_or_else(|| Error::Execution("missing BatchNorm CUDA statistics".into()))?;
        let channels = input.shape()[1];
        let spatial = input.shape()[2] * input.shape()[3];
        let mut input_gradient = self.output_buffer(input.numel())?;
        let mut weight_gradient = self.output_buffer(weight.numel())?;
        let mut bias_gradient = self.output_buffer(bias.numel())?;
        // SAFETY: affine outputs are distinct channel-sized allocations and
        // all read buffers retain the validated NCHW layout.
        unsafe {
            self.module.batch_norm_affine_backward(
                &self.stream,
                launch_config(channels)?,
                input.shape()[0],
                channels,
                spatial,
                gradient,
                &input_value,
                &statistics.mean,
                &statistics.inverse_standard_deviation,
                &mut weight_gradient,
                &mut bias_gradient,
            )
        }
        .map_err(cuda_error)?;
        // SAFETY: all NCHW, affine, statistics, and pre-reduced gradient
        // lengths were validated by batch_norm2d.
        unsafe {
            self.module.batch_norm_input_backward(
                &self.stream,
                launch_config(input.numel())?,
                channels,
                spatial,
                input.shape()[0] * spatial,
                training,
                gradient,
                &input_value,
                &weight_value,
                &statistics.mean,
                &statistics.inverse_standard_deviation,
                &weight_gradient,
                &bias_gradient,
                &mut input_gradient,
            )
        }
        .map_err(cuda_error)?;
        self.accumulate_gradient(input, &Arc::new(input_gradient))?;
        self.accumulate_gradient(weight, &Arc::new(weight_gradient))?;
        self.accumulate_gradient(bias, &Arc::new(bias_gradient))
    }
}

static EXECUTORS: OnceLock<Mutex<HashMap<usize, Executor>>> = OnceLock::new();

fn executors() -> &'static Mutex<HashMap<usize, Executor>> {
    EXECUTORS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn with_executor<T>(
    device: usize,
    operation: impl FnOnce(&mut Executor) -> Result<T>,
) -> Result<T> {
    let mut devices = executors()
        .lock()
        .map_err(|_| Error::Execution("CUDA executor lock was poisoned".into()))?;
    if !devices.contains_key(&device) {
        devices.insert(device, Executor::new(device)?);
    }
    operation(
        devices
            .get_mut(&device)
            .ok_or_else(|| Error::Execution("missing CUDA executor".into()))?,
    )
}

pub(crate) fn eval(tensor: &Tensor, device: usize) -> Result<Vec<f32>> {
    with_executor(device, |executor| {
        let output = executor.eval_node(tensor)?;
        let host = output.to_host_vec(&executor.stream).map_err(cuda_error)?;
        executor.clear_completed_host_copies();
        Ok(host)
    })
}

pub(crate) fn materialize(tensor: &Tensor, device: usize) -> Result<()> {
    with_executor(device, |executor| {
        let _ = executor.eval_node(tensor)?;
        executor.synchronize()
    })
}

pub(crate) fn backward(tensor: &Tensor, device: usize) -> Result<()> {
    with_executor(device, |executor| {
        let mut seen = HashSet::new();
        let mut topological = Vec::new();
        collect_topological(tensor, &mut seen, &mut topological);
        for node in &topological {
            executor.gradients.remove(&node.node.id);
        }
        let _ = executor.eval_node(tensor)?;
        executor
            .gradients
            .insert(tensor.node.id, executor.root_gradient.clone());
        for node in topological.iter().rev() {
            if let Some(gradient) = executor.gradients.get(&node.node.id).cloned() {
                executor.propagate_node(node, gradient)?;
            }
        }
        Ok(())
    })
}

pub(crate) fn gradient(tensor: &Tensor, device: usize) -> Result<Option<Vec<f32>>> {
    with_executor(device, |executor| {
        let host = executor
            .gradients
            .get(&tensor.node.id)
            .map(|gradient| gradient.to_host_vec(&executor.stream).map_err(cuda_error))
            .transpose()?;
        if host.is_some() {
            executor.clear_completed_host_copies();
        }
        Ok(host)
    })
}

pub(crate) fn zero_gradient(tensor: &Tensor, device: usize) -> Result<()> {
    with_executor(device, |executor| {
        executor.gradients.remove(&tensor.node.id);
        Ok(())
    })
}

pub(crate) fn release_optimizer(id: u64) {
    let Some(executors) = EXECUTORS.get() else {
        return;
    };
    let Ok(mut devices) = executors.try_lock() else {
        return;
    };
    for executor in devices.values_mut() {
        executor
            .adam_moments
            .retain(|(optimizer_id, _), _| *optimizer_id != id);
    }
}

pub(crate) fn release_node(id: u64, device: usize) {
    let Some(executors) = EXECUTORS.get() else {
        return;
    };
    let Ok(mut devices) = executors.try_lock() else {
        return;
    };
    if let Some(executor) = devices.get_mut(&device) {
        executor.values.remove(&id);
        executor.gradients.remove(&id);
        executor.batch_norm_statistics.remove(&id);
    }
}

pub(crate) fn sync_batch_norm_state(state: &Arc<Mutex<BatchNormState>>) -> Result<()> {
    let (device, key) = {
        let state_guard = state
            .lock()
            .map_err(|_| Error::Execution("BatchNorm state lock was poisoned".into()))?;
        let Device::Cuda(device) = state_guard.device else {
            return Ok(());
        };
        (device, Arc::as_ptr(state) as usize)
    };
    let values = with_executor(device, |executor| {
        executor
            .batch_norm_states
            .get(&key)
            .map(|gpu_state| {
                Ok((
                    gpu_state
                        .running_mean
                        .to_host_vec(&executor.stream)
                        .map_err(cuda_error)?,
                    gpu_state
                        .running_variance
                        .to_host_vec(&executor.stream)
                        .map_err(cuda_error)?,
                ))
            })
            .transpose()
    })?;
    if let Some((mean, variance)) = values {
        let mut state = state
            .lock()
            .map_err(|_| Error::Execution("BatchNorm state lock was poisoned".into()))?;
        state.running_mean = mean;
        state.running_variance = variance;
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(crate) struct AdamWHyperparameters {
    pub(crate) learning_rate: f32,
    pub(crate) weight_decay: f32,
    pub(crate) beta1: f32,
    pub(crate) beta2: f32,
    pub(crate) first_correction: f32,
    pub(crate) second_correction: f32,
    pub(crate) epsilon: f32,
}

pub(crate) fn adamw_step(
    parameter: &mut Parameter,
    optimizer_id: u64,
    hyperparameters: AdamWHyperparameters,
) -> Result<()> {
    let Device::Cuda(device) = parameter.value().device() else {
        return Err(Error::Execution(
            "CUDA AdamW received a CPU parameter".into(),
        ));
    };
    let parameter_id = parameter.id();
    let parameter_tensor = parameter.value().clone();
    let updated = with_executor(device, |executor| {
        let Some(gradient) = executor.gradients.get(&parameter_tensor.node.id).cloned() else {
            return Ok(None);
        };
        let parameter_value = executor.eval_node(&parameter_tensor)?;
        let moments = executor.adam_moments.remove(&(optimizer_id, parameter_id));
        let (first, second) = if let Some(moments) = moments {
            (moments.first, moments.second)
        } else {
            (
                Arc::new(executor.zero_buffer(parameter_tensor.numel())?),
                Arc::new(executor.zero_buffer(parameter_tensor.numel())?),
            )
        };
        let mut new_parameter = executor.output_buffer(parameter_tensor.numel())?;
        let mut new_first = executor.output_buffer(parameter_tensor.numel())?;
        let mut new_second = executor.output_buffer(parameter_tensor.numel())?;
        // SAFETY: parameter, gradient, moments, and all outputs contain the
        // same number of f32 elements; the three outputs are distinct.
        unsafe {
            executor.module.adamw(
                &executor.stream,
                launch_config(parameter_tensor.numel())?,
                hyperparameters.learning_rate,
                hyperparameters.weight_decay,
                hyperparameters.beta1,
                hyperparameters.beta2,
                hyperparameters.first_correction,
                hyperparameters.second_correction,
                hyperparameters.epsilon,
                &parameter_value,
                &gradient,
                &first,
                &second,
                &mut new_parameter,
                &mut new_first,
                &mut new_second,
            )
        }
        .map_err(cuda_error)?;
        executor.adam_moments.insert(
            (optimizer_id, parameter_id),
            AdamMoments {
                first: Arc::new(new_first),
                second: Arc::new(new_second),
            },
        );
        let tensor = Tensor::zeros(parameter_tensor.shape().to_vec())?.to(Device::Cuda(device));
        executor
            .values
            .insert(tensor.node.id, Arc::new(new_parameter));
        Ok(Some(tensor))
    })?;
    if let Some(updated) = updated {
        parameter.replace_value(updated)?;
    }
    Ok(())
}

fn collect_topological(tensor: &Tensor, seen: &mut HashSet<u64>, output: &mut Vec<Tensor>) {
    if !seen.insert(tensor.node.id) {
        return;
    }
    match &tensor.node.op {
        Op::Data(_) | Op::Placeholder(_) => {}
        Op::Add(left, right) | Op::Mul(left, right) | Op::MatMul(left, right) => {
            collect_topological(left, seen, output);
            collect_topological(right, seen, output);
        }
        Op::Relu(input) | Op::Reshape(input) | Op::AvgPool2d { input, .. } => {
            collect_topological(input, seen, output);
        }
        Op::Conv2d {
            input,
            weight,
            bias,
            ..
        }
        | Op::BatchNorm2d {
            input,
            weight,
            bias,
            ..
        } => {
            collect_topological(input, seen, output);
            collect_topological(weight, seen, output);
            collect_topological(bias, seen, output);
        }
        Op::CrossEntropy { logits, targets } => {
            collect_topological(logits, seen, output);
            collect_topological(targets, seen, output);
        }
    }
    output.push(tensor.clone());
}

fn launch_config(numel: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig::for_num_elems(u32::try_from(numel).map_err(
        |_| Error::Execution("tensor is too large for a CUDA grid".into()),
    )?))
}

fn tiled_launch_config(planes: usize, elements_per_plane: usize) -> Result<LaunchConfig> {
    tiled_launch_config_with_tile(planes, elements_per_plane, 256)
}

fn tiled_launch_config_with_tile(
    planes: usize,
    elements_per_plane: usize,
    tile: usize,
) -> Result<LaunchConfig> {
    let tiles_per_plane = elements_per_plane / tile + usize::from(elements_per_plane % tile != 0);
    let blocks = planes
        .checked_mul(tiles_per_plane)
        .ok_or_else(|| Error::Execution("CUDA tiled grid size overflow".into()))?;
    Ok(LaunchConfig {
        grid_dim: (
            u32::try_from(blocks)
                .map_err(|_| Error::Execution("CUDA tiled grid exceeds u32".into()))?,
            1,
            1,
        ),
        block_dim: (
            u32::try_from(tile)
                .map_err(|_| Error::Execution("CUDA tile size exceeds u32".into()))?,
            1,
            1,
        ),
        shared_mem_bytes: 0,
    })
}

fn cuda_error(error: impl std::fmt::Display) -> Error {
    Error::Execution(error.to_string())
}

#[cfg(feature = "cudnn")]
fn cudnn_shape(
    input: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    stride: usize,
    padding: usize,
    groups: usize,
) -> crate::cudnn::ConvShape {
    cudnn_shape_from_parts(
        input,
        weight,
        output.shape()[2],
        output.shape()[3],
        stride,
        padding,
        groups,
    )
}

#[cfg(feature = "cudnn")]
fn cudnn_shape_from_parts(
    input: &Tensor,
    weight: &Tensor,
    out_height: usize,
    out_width: usize,
    stride: usize,
    padding: usize,
    groups: usize,
) -> crate::cudnn::ConvShape {
    crate::cudnn::ConvShape {
        batch: input.shape()[0],
        in_channels: input.shape()[1],
        height: input.shape()[2],
        width: input.shape()[3],
        out_channels: weight.shape()[0],
        out_height,
        out_width,
        kernel: weight.shape()[2],
        stride,
        padding,
        groups,
    }
}
