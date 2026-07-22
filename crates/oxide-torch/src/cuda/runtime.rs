use crate::nn::Parameter;
use crate::tensor::{BatchNormState, Op, Tensor};
use crate::{CustomInput, Device, Error, Result};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};

use super::kernels;

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
    transformer: kernels::transformer::LoadedModule,
    values: HashMap<u64, Buffer>,
    gradients: HashMap<u64, Buffer>,
    batch_norm_states: HashMap<usize, BatchNormDeviceState>,
    batch_norm_statistics: HashMap<u64, BatchNormDeviceStatistics>,
    adam_moments: HashMap<(u64, u64), AdamMoments>,
    root_gradient: Buffer,
    pending_host_copies: Vec<Arc<[f32]>>,
    pending_host_bytes: usize,
    #[cfg(feature = "cudnn")]
    cudnn: Option<oxide_torch_cuda::cudnn::Cudnn>,
}

impl Executor {
    fn new(device: usize) -> Result<Self> {
        let context = CudaContext::new(device).map_err(cuda_error)?;
        let stream = context.default_stream();
        let module = oxide_torch_cuda::load_kernels(&context).map_err(cuda_error)?;
        let transformer =
            kernels::transformer::LoadedModule::from_parent(&module).map_err(cuda_error)?;
        let root_gradient = Arc::new(DeviceBuffer::from_host(&stream, &[1.0]).map_err(cuda_error)?);
        #[cfg(feature = "cudnn")]
        let cudnn = oxide_torch_cuda::cudnn::Cudnn::try_new();
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
            transformer,
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
            Op::Linear {
                input,
                weight,
                bias,
            } => self.eval_transformer_cuda(
                tensor,
                &[input, weight, bias],
                crate::transformer::Primitive::Linear,
            )?,
            Op::Gelu(input) => {
                self.eval_transformer_cuda(tensor, &[input], crate::transformer::Primitive::Gelu)?
            }
            Op::Tanh(input) => {
                self.eval_transformer_cuda(tensor, &[input], crate::transformer::Primitive::Tanh)?
            }
            Op::Embedding { ids, weight } => self.eval_transformer_cuda(
                tensor,
                &[ids, weight],
                crate::transformer::Primitive::Embedding,
            )?,
            Op::LayerNorm {
                input,
                weight,
                bias,
                epsilon,
            } => self.eval_transformer_cuda(
                tensor,
                &[input, weight, bias],
                crate::transformer::Primitive::LayerNorm { epsilon: *epsilon },
            )?,
            Op::SelectFirst(input) => self.eval_transformer_cuda(
                tensor,
                &[input],
                crate::transformer::Primitive::SelectFirst,
            )?,
            Op::ScaledDotProductAttention {
                input,
                mask,
                query_weight,
                query_bias,
                key_weight,
                key_bias,
                value_weight,
                value_bias,
                heads,
            } => self.eval_transformer_cuda(
                tensor,
                &[
                    input,
                    mask,
                    query_weight,
                    query_bias,
                    key_weight,
                    key_bias,
                    value_weight,
                    value_bias,
                ],
                crate::transformer::Primitive::ScaledDotProductAttention { heads: *heads },
            )?,
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
            Op::Custom { .. } => {
                return Err(Error::Execution(
                    "user-defined custom operations currently require the CPU backend".into(),
                ));
            }
        };
        let output = Arc::new(output);
        self.values.insert(tensor.node.id, output.clone());
        Ok(output)
    }

    fn eval_transformer_cuda(
        &mut self,
        tensor: &Tensor,
        inputs: &[Tensor],
        primitive: crate::transformer::Primitive,
    ) -> Result<DeviceBuffer<f32>> {
        let buffers = inputs
            .iter()
            .map(|input| self.eval_node(input))
            .collect::<Result<Vec<_>>>()?;
        if let crate::transformer::Primitive::ScaledDotProductAttention { heads } = primitive {
            return self.eval_attention_custom(tensor, inputs, &buffers, heads);
        }
        let mut output = self.output_buffer(tensor.numel())?;
        let config = launch_config(tensor.numel())?;
        // SAFETY: the public transformer constructors validate shapes and all
        // kernels bounds-check their output index.
        unsafe {
            match primitive {
                crate::transformer::Primitive::Linear => self.transformer.linear(
                    &self.stream,
                    config,
                    inputs[0].shape()[inputs[0].shape().len() - 1],
                    inputs[1].shape()[0],
                    &buffers[0],
                    &buffers[1],
                    &buffers[2],
                    &mut output,
                ),
                crate::transformer::Primitive::Gelu => {
                    self.transformer
                        .gelu(&self.stream, config, &buffers[0], &mut output)
                }
                crate::transformer::Primitive::Tanh => {
                    self.transformer
                        .tanh(&self.stream, config, &buffers[0], &mut output)
                }
                crate::transformer::Primitive::Embedding => self.transformer.embedding(
                    &self.stream,
                    config,
                    inputs[1].shape()[1],
                    inputs[1].shape()[0],
                    &buffers[0],
                    &buffers[1],
                    &mut output,
                ),
                crate::transformer::Primitive::LayerNorm { epsilon } => {
                    self.transformer.layer_norm(
                        &self.stream,
                        config,
                        inputs[1].numel(),
                        epsilon,
                        &buffers[0],
                        &buffers[1],
                        &buffers[2],
                        &mut output,
                    )
                }
                crate::transformer::Primitive::SelectFirst => self.transformer.select_first(
                    &self.stream,
                    config,
                    inputs[0].shape()[1],
                    inputs[0].shape()[2],
                    &buffers[0],
                    &mut output,
                ),
                crate::transformer::Primitive::ScaledDotProductAttention { .. } => unreachable!(),
            }
        }
        .map_err(cuda_error)?;
        Ok(output)
    }

    fn eval_attention_custom(
        &self,
        tensor: &Tensor,
        inputs: &[Tensor],
        buffers: &[Buffer],
        heads: usize,
    ) -> Result<DeviceBuffer<f32>> {
        let hidden = inputs[0].shape()[2];
        let elements = inputs[0].numel();
        let mut query = self.output_buffer(elements)?;
        let mut key = self.output_buffer(elements)?;
        let mut value = self.output_buffer(elements)?;
        let mut output = self.output_buffer(tensor.numel())?;
        let projection_config = launch_config(elements)?;
        // SAFETY: Q/K/V projection tensors all use validated [hidden, hidden]
        // weights and distinct output buffers.
        unsafe {
            self.transformer
                .linear(
                    &self.stream,
                    projection_config,
                    hidden,
                    hidden,
                    &buffers[0],
                    &buffers[2],
                    &buffers[3],
                    &mut query,
                )
                .map_err(cuda_error)?;
            self.transformer
                .linear(
                    &self.stream,
                    projection_config,
                    hidden,
                    hidden,
                    &buffers[0],
                    &buffers[4],
                    &buffers[5],
                    &mut key,
                )
                .map_err(cuda_error)?;
            self.transformer
                .linear(
                    &self.stream,
                    projection_config,
                    hidden,
                    hidden,
                    &buffers[0],
                    &buffers[6],
                    &buffers[7],
                    &mut value,
                )
                .map_err(cuda_error)?;
            self.transformer
                .projected_attention(
                    &self.stream,
                    launch_config(tensor.numel())?,
                    inputs[0].shape()[1],
                    hidden,
                    heads,
                    &query,
                    &key,
                    &value,
                    &buffers[1],
                    &mut output,
                )
                .map_err(cuda_error)?;
        }
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
            Op::Linear {
                input,
                weight,
                bias,
            } => self.backward_transformer_cuda(
                &[input, weight, bias],
                crate::transformer::Primitive::Linear,
                &gradient,
            ),
            Op::Gelu(input) => self.backward_transformer_cuda(
                &[input],
                crate::transformer::Primitive::Gelu,
                &gradient,
            ),
            Op::Tanh(input) => self.backward_transformer_cuda(
                &[input],
                crate::transformer::Primitive::Tanh,
                &gradient,
            ),
            Op::Embedding { ids, weight } => self.backward_transformer_cuda(
                &[ids, weight],
                crate::transformer::Primitive::Embedding,
                &gradient,
            ),
            Op::LayerNorm {
                input,
                weight,
                bias,
                epsilon,
            } => self.backward_transformer_cuda(
                &[input, weight, bias],
                crate::transformer::Primitive::LayerNorm { epsilon: *epsilon },
                &gradient,
            ),
            Op::SelectFirst(input) => self.backward_transformer_cuda(
                &[input],
                crate::transformer::Primitive::SelectFirst,
                &gradient,
            ),
            Op::ScaledDotProductAttention {
                input,
                mask,
                query_weight,
                query_bias,
                key_weight,
                key_bias,
                value_weight,
                value_bias,
                heads,
            } => self.backward_transformer_cuda(
                &[
                    input,
                    mask,
                    query_weight,
                    query_bias,
                    key_weight,
                    key_bias,
                    value_weight,
                    value_bias,
                ],
                crate::transformer::Primitive::ScaledDotProductAttention { heads: *heads },
                &gradient,
            ),
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
            Op::Custom { .. } => Err(Error::Execution(
                "user-defined custom operations currently require CPU autograd".into(),
            )),
        }
    }

    fn backward_transformer_cuda(
        &mut self,
        inputs: &[Tensor],
        primitive: crate::transformer::Primitive,
        gradient: &Buffer,
    ) -> Result<()> {
        let input_values = inputs
            .iter()
            .map(|input| {
                self.eval_node(input)?
                    .to_host_vec(&self.stream)
                    .map_err(cuda_error)
            })
            .collect::<Result<Vec<_>>>()?;
        let host_gradient = gradient.to_host_vec(&self.stream).map_err(cuda_error)?;
        let custom_inputs = inputs
            .iter()
            .zip(&input_values)
            .map(|(input, values)| CustomInput {
                shape: input.shape(),
                values,
            })
            .collect::<Vec<_>>();
        let gradients = crate::transformer::backward(primitive, &custom_inputs, &host_gradient)?;
        if gradients.len() != inputs.len() {
            return Err(Error::Execution(
                "built-in custom operation returned the wrong gradient count".into(),
            ));
        }
        for (input, gradient) in inputs.iter().zip(gradients) {
            if let Some(gradient) = gradient {
                let gradient =
                    Arc::new(DeviceBuffer::from_host(&self.stream, &gradient).map_err(cuda_error)?);
                self.accumulate_gradient(input, &gradient)?;
            }
        }
        Ok(())
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
        Op::Relu(input)
        | Op::Gelu(input)
        | Op::Tanh(input)
        | Op::SelectFirst(input)
        | Op::Reshape(input)
        | Op::AvgPool2d { input, .. } => {
            collect_topological(input, seen, output);
        }
        Op::Linear {
            input,
            weight,
            bias,
        }
        | Op::LayerNorm {
            input,
            weight,
            bias,
            ..
        } => {
            collect_topological(input, seen, output);
            collect_topological(weight, seen, output);
            collect_topological(bias, seen, output);
        }
        Op::Embedding { ids, weight } => {
            collect_topological(ids, seen, output);
            collect_topological(weight, seen, output);
        }
        Op::ScaledDotProductAttention {
            input,
            mask,
            query_weight,
            query_bias,
            key_weight,
            key_bias,
            value_weight,
            value_bias,
            ..
        } => {
            for operand in [
                input,
                mask,
                query_weight,
                query_bias,
                key_weight,
                key_bias,
                value_weight,
                value_bias,
            ] {
                collect_topological(operand, seen, output);
            }
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
        Op::Custom { inputs, .. } => {
            for input in inputs {
                collect_topological(input, seen, output);
            }
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
) -> oxide_torch_cuda::cudnn::ConvShape {
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
) -> oxide_torch_cuda::cudnn::ConvShape {
    oxide_torch_cuda::cudnn::ConvShape {
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
