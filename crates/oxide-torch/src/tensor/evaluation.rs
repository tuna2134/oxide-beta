use super::{
    Arc, BatchNormState, BatchNormStatistics, CustomInput, Device, Error, HashMap, Mutex, Op,
    Result, Tensor,
};

#[allow(clippy::too_many_lines)]
pub(crate) fn eval_cpu(
    tensor: &Tensor,
    cache: &mut HashMap<u64, Vec<f32>>,
    inputs: Option<&[Vec<f32>]>,
) -> Result<Vec<f32>> {
    if let Some(value) = cache.get(&tensor.node.id) {
        return Ok(value.clone());
    }
    let value = match &tensor.node.op {
        Op::Data(data) => data.to_vec(),
        Op::Placeholder(slot) => inputs
            .and_then(|values| values.get(*slot))
            .cloned()
            .ok_or_else(|| Error::Trace(format!("missing traced input {slot}")))?,
        Op::Add(lhs, rhs) => zip_map(
            eval_cpu(lhs, cache, inputs)?,
            eval_cpu(rhs, cache, inputs)?,
            |a, b| a + b,
        ),
        Op::Mul(lhs, rhs) => zip_map(
            eval_cpu(lhs, cache, inputs)?,
            eval_cpu(rhs, cache, inputs)?,
            |a, b| a * b,
        ),
        Op::Relu(input) => eval_cpu(input, cache, inputs)?
            .into_iter()
            .map(|value| value.max(0.0))
            .collect(),
        Op::MatMul(lhs, rhs) => {
            let lhs_data = eval_cpu(lhs, cache, inputs)?;
            let rhs_data = eval_cpu(rhs, cache, inputs)?;
            let rows = lhs.shape()[0];
            let inner_size = lhs.shape()[1];
            let columns = rhs.shape()[1];
            let mut output = vec![0.0; rows * columns];
            for row in 0..rows {
                for col in 0..columns {
                    output[row * columns + col] = (0..inner_size)
                        .map(|inner| {
                            lhs_data[row * inner_size + inner] * rhs_data[inner * columns + col]
                        })
                        .sum();
                }
            }
            output
        }
        Op::Linear {
            input,
            weight,
            bias,
        } => eval_transformer(
            crate::transformer::Primitive::Linear,
            &[input, weight, bias],
            cache,
            inputs,
        )?,
        Op::Gelu(input) => {
            eval_transformer(crate::transformer::Primitive::Gelu, &[input], cache, inputs)?
        }
        Op::Tanh(input) => {
            eval_transformer(crate::transformer::Primitive::Tanh, &[input], cache, inputs)?
        }
        Op::Embedding { ids, weight } => eval_transformer(
            crate::transformer::Primitive::Embedding,
            &[ids, weight],
            cache,
            inputs,
        )?,
        Op::LayerNorm {
            input,
            weight,
            bias,
            epsilon,
        } => eval_transformer(
            crate::transformer::Primitive::LayerNorm { epsilon: *epsilon },
            &[input, weight, bias],
            cache,
            inputs,
        )?,
        Op::SelectFirst(input) => eval_transformer(
            crate::transformer::Primitive::SelectFirst,
            &[input],
            cache,
            inputs,
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
        } => eval_transformer(
            crate::transformer::Primitive::ScaledDotProductAttention { heads: *heads },
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
            cache,
            inputs,
        )?,
        Op::Conv2d {
            input,
            weight,
            bias,
            stride,
            padding,
            groups,
        } => eval_conv2d(
            input, weight, bias, *stride, *padding, *groups, cache, inputs,
        )?,
        Op::AvgPool2d {
            input,
            kernel,
            stride,
        } => eval_avg_pool2d(input, *kernel, *stride, cache, inputs)?,
        Op::Reshape(input) => eval_cpu(input, cache, inputs)?,
        Op::CrossEntropy { logits, targets } => eval_cross_entropy(logits, targets, cache, inputs)?,
        Op::BatchNorm2d {
            input,
            weight,
            bias,
            state,
            saved_statistics,
            training,
            momentum,
            epsilon,
        } => eval_batch_norm2d(
            input,
            weight,
            bias,
            state,
            saved_statistics,
            *training,
            *momentum,
            *epsilon,
            cache,
            inputs,
        )?,
        Op::Custom {
            inputs: operands,
            operation,
        } => {
            let values = operands
                .iter()
                .map(|operand| eval_cpu(operand, cache, inputs))
                .collect::<Result<Vec<_>>>()?;
            let custom_inputs = operands
                .iter()
                .zip(&values)
                .map(|(operand, values)| CustomInput {
                    shape: operand.shape(),
                    values,
                })
                .collect::<Vec<_>>();
            operation.forward(&custom_inputs)?
        }
    };
    cache.insert(tensor.node.id, value.clone());
    Ok(value)
}

fn eval_transformer(
    primitive: crate::transformer::Primitive,
    operands: &[&Tensor],
    cache: &mut HashMap<u64, Vec<f32>>,
    inputs: Option<&[Vec<f32>]>,
) -> Result<Vec<f32>> {
    let values = operands
        .iter()
        .map(|operand| eval_cpu(operand, cache, inputs))
        .collect::<Result<Vec<_>>>()?;
    let inputs = operands
        .iter()
        .zip(&values)
        .map(|(operand, values)| CustomInput {
            shape: operand.shape(),
            values,
        })
        .collect::<Vec<_>>();
    crate::transformer::forward(primitive, &inputs)
}

#[allow(clippy::too_many_lines)]
pub(super) fn clone_to_device(
    tensor: &Tensor,
    device: Device,
    cache: &mut HashMap<u64, Tensor>,
) -> Tensor {
    if let Some(value) = cache.get(&tensor.node.id) {
        return value.clone();
    }
    let op = match &tensor.node.op {
        Op::Data(data) => Op::Data(data.clone()),
        Op::Placeholder(slot) => Op::Placeholder(*slot),
        Op::Add(a, b) => Op::Add(
            clone_to_device(a, device, cache),
            clone_to_device(b, device, cache),
        ),
        Op::Mul(a, b) => Op::Mul(
            clone_to_device(a, device, cache),
            clone_to_device(b, device, cache),
        ),
        Op::Relu(input) => Op::Relu(clone_to_device(input, device, cache)),
        Op::MatMul(a, b) => Op::MatMul(
            clone_to_device(a, device, cache),
            clone_to_device(b, device, cache),
        ),
        Op::Linear {
            input,
            weight,
            bias,
        } => Op::Linear {
            input: clone_to_device(input, device, cache),
            weight: clone_to_device(weight, device, cache),
            bias: clone_to_device(bias, device, cache),
        },
        Op::Gelu(input) => Op::Gelu(clone_to_device(input, device, cache)),
        Op::Tanh(input) => Op::Tanh(clone_to_device(input, device, cache)),
        Op::Embedding { ids, weight } => Op::Embedding {
            ids: clone_to_device(ids, device, cache),
            weight: clone_to_device(weight, device, cache),
        },
        Op::LayerNorm {
            input,
            weight,
            bias,
            epsilon,
        } => Op::LayerNorm {
            input: clone_to_device(input, device, cache),
            weight: clone_to_device(weight, device, cache),
            bias: clone_to_device(bias, device, cache),
            epsilon: *epsilon,
        },
        Op::SelectFirst(input) => Op::SelectFirst(clone_to_device(input, device, cache)),
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
        } => Op::ScaledDotProductAttention {
            input: clone_to_device(input, device, cache),
            mask: clone_to_device(mask, device, cache),
            query_weight: clone_to_device(query_weight, device, cache),
            query_bias: clone_to_device(query_bias, device, cache),
            key_weight: clone_to_device(key_weight, device, cache),
            key_bias: clone_to_device(key_bias, device, cache),
            value_weight: clone_to_device(value_weight, device, cache),
            value_bias: clone_to_device(value_bias, device, cache),
            heads: *heads,
        },
        Op::Conv2d {
            input,
            weight,
            bias,
            stride,
            padding,
            groups,
        } => Op::Conv2d {
            input: clone_to_device(input, device, cache),
            weight: clone_to_device(weight, device, cache),
            bias: clone_to_device(bias, device, cache),
            stride: *stride,
            padding: *padding,
            groups: *groups,
        },
        Op::AvgPool2d {
            input,
            kernel,
            stride,
        } => Op::AvgPool2d {
            input: clone_to_device(input, device, cache),
            kernel: *kernel,
            stride: *stride,
        },
        Op::Reshape(input) => Op::Reshape(clone_to_device(input, device, cache)),
        Op::CrossEntropy { logits, targets } => Op::CrossEntropy {
            logits: clone_to_device(logits, device, cache),
            targets: clone_to_device(targets, device, cache),
        },
        Op::BatchNorm2d {
            input,
            weight,
            bias,
            state,
            training,
            momentum,
            epsilon,
            ..
        } => Op::BatchNorm2d {
            input: clone_to_device(input, device, cache),
            weight: clone_to_device(weight, device, cache),
            bias: clone_to_device(bias, device, cache),
            state: state.clone(),
            saved_statistics: Arc::new(Mutex::new(None)),
            training: *training,
            momentum: *momentum,
            epsilon: *epsilon,
        },
        Op::Custom { inputs, operation } => Op::Custom {
            inputs: inputs
                .iter()
                .map(|input| clone_to_device(input, device, cache))
                .collect(),
            operation: Arc::clone(operation),
        },
    };
    let result = Tensor::new(tensor.node.shape.clone(), device, op);
    cache.insert(tensor.node.id, result.clone());
    result
}

#[allow(clippy::too_many_arguments)]
fn eval_conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    stride: usize,
    padding: usize,
    groups: usize,
    cache: &mut HashMap<u64, Vec<f32>>,
    inputs: Option<&[Vec<f32>]>,
) -> Result<Vec<f32>> {
    let input_data = eval_cpu(input, cache, inputs)?;
    let weight_data = eval_cpu(weight, cache, inputs)?;
    let bias_data = eval_cpu(bias, cache, inputs)?;
    let [batch, in_channels, height, width] = [
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    ];
    let [out_channels, kernel_h, kernel_w] =
        [weight.shape()[0], weight.shape()[2], weight.shape()[3]];
    let out_h = (height + 2 * padding - kernel_h) / stride + 1;
    let out_w = (width + 2 * padding - kernel_w) / stride + 1;
    let in_per_group = in_channels / groups;
    let out_per_group = out_channels / groups;
    let mut output = vec![0.0; batch * out_channels * out_h * out_w];
    for batch_index in 0..batch {
        for (out_channel, &channel_bias) in bias_data.iter().enumerate().take(out_channels) {
            let group = out_channel / out_per_group;
            for out_y in 0..out_h {
                for out_x in 0..out_w {
                    let mut sum = channel_bias;
                    for local_channel in 0..in_per_group {
                        let in_channel = group * in_per_group + local_channel;
                        for kernel_y in 0..kernel_h {
                            let padded_y = out_y * stride + kernel_y;
                            if padded_y < padding || padded_y - padding >= height {
                                continue;
                            }
                            let in_y = padded_y - padding;
                            for kernel_x in 0..kernel_w {
                                let padded_x = out_x * stride + kernel_x;
                                if padded_x < padding || padded_x - padding >= width {
                                    continue;
                                }
                                let in_x = padded_x - padding;
                                let input_index =
                                    ((batch_index * in_channels + in_channel) * height + in_y)
                                        * width
                                        + in_x;
                                let weight_index = ((out_channel * in_per_group + local_channel)
                                    * kernel_h
                                    + kernel_y)
                                    * kernel_w
                                    + kernel_x;
                                sum += input_data[input_index] * weight_data[weight_index];
                            }
                        }
                    }
                    let output_index = ((batch_index * out_channels + out_channel) * out_h + out_y)
                        * out_w
                        + out_x;
                    output[output_index] = sum;
                }
            }
        }
    }
    Ok(output)
}

#[allow(clippy::cast_precision_loss)]
fn eval_avg_pool2d(
    input: &Tensor,
    kernel: [usize; 2],
    stride: [usize; 2],
    cache: &mut HashMap<u64, Vec<f32>>,
    inputs: Option<&[Vec<f32>]>,
) -> Result<Vec<f32>> {
    let input_data = eval_cpu(input, cache, inputs)?;
    let [batch, channels, height, width] = [
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    ];
    let out_h = (height - kernel[0]) / stride[0] + 1;
    let out_w = (width - kernel[1]) / stride[1] + 1;
    let denominator = (kernel[0] * kernel[1]) as f32;
    let mut output = vec![0.0; batch * channels * out_h * out_w];
    for batch_index in 0..batch {
        for channel in 0..channels {
            for out_y in 0..out_h {
                for out_x in 0..out_w {
                    let mut sum = 0.0;
                    for kernel_y in 0..kernel[0] {
                        for kernel_x in 0..kernel[1] {
                            let in_y = out_y * stride[0] + kernel_y;
                            let in_x = out_x * stride[1] + kernel_x;
                            let index =
                                ((batch_index * channels + channel) * height + in_y) * width + in_x;
                            sum += input_data[index];
                        }
                    }
                    let index =
                        ((batch_index * channels + channel) * out_h + out_y) * out_w + out_x;
                    output[index] = sum / denominator;
                }
            }
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn eval_batch_norm2d(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    state: &Arc<Mutex<BatchNormState>>,
    saved_statistics: &Arc<Mutex<Option<BatchNormStatistics>>>,
    training: bool,
    momentum: f32,
    epsilon: f32,
    cache: &mut HashMap<u64, Vec<f32>>,
    inputs: Option<&[Vec<f32>]>,
) -> Result<Vec<f32>> {
    let input_data = eval_cpu(input, cache, inputs)?;
    let weight_data = eval_cpu(weight, cache, inputs)?;
    let bias_data = eval_cpu(bias, cache, inputs)?;
    let statistics = {
        let mut saved = saved_statistics
            .lock()
            .map_err(|_| Error::Execution("BatchNorm statistics lock was poisoned".into()))?;
        if let Some(statistics) = saved.as_ref() {
            statistics.clone()
        } else {
            let statistics = if training {
                let statistics = batch_statistics(input, &input_data, epsilon)?;
                update_running_statistics(input, &input_data, state, momentum)?;
                statistics
            } else {
                running_statistics(state, epsilon)?
            };
            *saved = Some(statistics.clone());
            statistics
        }
    };
    Ok(apply_batch_norm(
        input,
        &input_data,
        &weight_data,
        &bias_data,
        &statistics,
    ))
}

pub(super) fn batch_statistics(
    input: &Tensor,
    input_data: &[f32],
    epsilon: f32,
) -> Result<BatchNormStatistics> {
    let channels = input.shape()[1];
    let spatial = input.shape()[2] * input.shape()[3];
    let samples = input.shape()[0] * spatial;
    if samples <= 1 {
        return Err(Error::Execution(
            "BatchNorm training requires more than one value per channel".into(),
        ));
    }
    #[allow(clippy::cast_precision_loss)]
    let denominator = samples as f32;
    let mut mean = vec![0.0; channels];
    for batch in 0..input.shape()[0] {
        for (channel, channel_mean) in mean.iter_mut().enumerate() {
            let start = (batch * channels + channel) * spatial;
            *channel_mean += input_data[start..start + spatial].iter().sum::<f32>();
        }
    }
    for value in &mut mean {
        *value /= denominator;
    }
    let mut variance = vec![0.0; channels];
    for batch in 0..input.shape()[0] {
        for channel in 0..channels {
            let start = (batch * channels + channel) * spatial;
            variance[channel] += input_data[start..start + spatial]
                .iter()
                .map(|value| (value - mean[channel]).powi(2))
                .sum::<f32>();
        }
    }
    for value in &mut variance {
        *value /= denominator;
    }
    Ok(BatchNormStatistics {
        mean,
        inverse_standard_deviation: variance
            .into_iter()
            .map(|value| (value + epsilon).sqrt().recip())
            .collect(),
    })
}

fn update_running_statistics(
    input: &Tensor,
    input_data: &[f32],
    state: &Arc<Mutex<BatchNormState>>,
    momentum: f32,
) -> Result<()> {
    let statistics = batch_statistics(input, input_data, 0.0)?;
    let samples = input.shape()[0] * input.shape()[2] * input.shape()[3];
    #[allow(clippy::cast_precision_loss)]
    let unbiased_correction = samples as f32 / (samples - 1) as f32;
    let mut buffers = state
        .lock()
        .map_err(|_| Error::Execution("BatchNorm state lock was poisoned".into()))?;
    for channel in 0..input.shape()[1] {
        let biased_variance = statistics.inverse_standard_deviation[channel].powi(-2);
        buffers.running_mean[channel] =
            (1.0 - momentum) * buffers.running_mean[channel] + momentum * statistics.mean[channel];
        buffers.running_variance[channel] = (1.0 - momentum) * buffers.running_variance[channel]
            + momentum * biased_variance * unbiased_correction;
    }
    Ok(())
}

pub(super) fn running_statistics(
    state: &Arc<Mutex<BatchNormState>>,
    epsilon: f32,
) -> Result<BatchNormStatistics> {
    let buffers = state
        .lock()
        .map_err(|_| Error::Execution("BatchNorm state lock was poisoned".into()))?;
    Ok(BatchNormStatistics {
        mean: buffers.running_mean.clone(),
        inverse_standard_deviation: buffers
            .running_variance
            .iter()
            .map(|value| (value + epsilon).sqrt().recip())
            .collect(),
    })
}

fn apply_batch_norm(
    input: &Tensor,
    input_data: &[f32],
    weight: &[f32],
    bias: &[f32],
    statistics: &BatchNormStatistics,
) -> Vec<f32> {
    let channels = input.shape()[1];
    let spatial = input.shape()[2] * input.shape()[3];
    input_data
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let channel = (index / spatial) % channels;
            (value - statistics.mean[channel])
                * statistics.inverse_standard_deviation[channel]
                * weight[channel]
                + bias[channel]
        })
        .collect()
}

fn eval_cross_entropy(
    logits: &Tensor,
    targets: &Tensor,
    cache: &mut HashMap<u64, Vec<f32>>,
    inputs: Option<&[Vec<f32>]>,
) -> Result<Vec<f32>> {
    let logits_data = eval_cpu(logits, cache, inputs)?;
    let targets_data = eval_cpu(targets, cache, inputs)?;
    let batch = logits.shape()[0];
    let classes = logits.shape()[1];
    let mut loss = 0.0;
    for batch_index in 0..batch {
        let target = class_index(targets_data[batch_index], classes)?;
        let row = &logits_data[batch_index * classes..(batch_index + 1) * classes];
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let normalizer: f32 = row.iter().map(|value| (*value - maximum).exp()).sum();
        loss += normalizer.ln() + maximum - row[target];
    }
    #[allow(clippy::cast_precision_loss)]
    let batch_f32 = batch as f32;
    Ok(vec![loss / batch_f32])
}

pub(super) fn class_index(value: f32, classes: usize) -> Result<usize> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return Err(Error::InvalidShape(format!(
            "target class must be a non-negative integer, got {value}"
        )));
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let index = value as usize;
    if index >= classes {
        return Err(Error::InvalidShape(format!(
            "target class {index} is outside 0..{classes}"
        )));
    }
    Ok(index)
}

fn zip_map(lhs: Vec<f32>, rhs: Vec<f32>, op: impl Fn(f32, f32) -> f32) -> Vec<f32> {
    lhs.into_iter().zip(rhs).map(|(a, b)| op(a, b)).collect()
}
