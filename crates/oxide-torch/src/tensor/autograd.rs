use super::evaluation::{batch_statistics, class_index, eval_cpu, running_statistics};
use super::{
    Arc, BatchNormState, BatchNormStatistics, CustomInput, Error, HashMap, HashSet, Mutex, Op,
    Result, Tensor,
};

pub(super) fn clear_graph_grads(tensor: &Tensor, visited: &mut HashSet<u64>) -> Result<()> {
    if !visited.insert(tensor.node.id) {
        return Ok(());
    }
    tensor.zero_grad()?;
    match &tensor.node.op {
        Op::Data(_) | Op::Placeholder(_) => {}
        Op::Add(left, right) | Op::Mul(left, right) | Op::MatMul(left, right) => {
            clear_graph_grads(left, visited)?;
            clear_graph_grads(right, visited)?;
        }
        Op::Relu(input)
        | Op::Gelu(input)
        | Op::Tanh(input)
        | Op::SelectFirst(input)
        | Op::Reshape(input)
        | Op::AvgPool2d { input, .. } => {
            clear_graph_grads(input, visited)?;
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
        }
        | Op::Conv2d {
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
            clear_graph_grads(input, visited)?;
            clear_graph_grads(weight, visited)?;
            clear_graph_grads(bias, visited)?;
        }
        Op::Embedding { ids, weight } => {
            clear_graph_grads(ids, visited)?;
            clear_graph_grads(weight, visited)?;
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
                clear_graph_grads(operand, visited)?;
            }
        }
        Op::CrossEntropy { logits, targets } => {
            clear_graph_grads(logits, visited)?;
            clear_graph_grads(targets, visited)?;
        }
        Op::Custom { inputs, .. } => {
            for input in inputs {
                clear_graph_grads(input, visited)?;
            }
        }
    }
    Ok(())
}

fn accumulate_gradient(tensor: &Tensor, incoming: &[f32]) -> Result<()> {
    let mut gradient = tensor
        .node
        .grad
        .lock()
        .map_err(|_| Error::Execution("gradient lock was poisoned".into()))?;
    if let Some(current) = gradient.as_mut() {
        for (value, addition) in current.iter_mut().zip(incoming) {
            *value += addition;
        }
    } else {
        *gradient = Some(incoming.to_vec());
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
pub(super) fn backward_node(
    tensor: &Tensor,
    gradient: Vec<f32>,
    values: &mut HashMap<u64, Vec<f32>>,
) -> Result<()> {
    accumulate_gradient(tensor, &gradient)?;
    match &tensor.node.op {
        Op::Data(_) | Op::Placeholder(_) => Ok(()),
        Op::Add(left, right) => {
            backward_node(left, gradient.clone(), values)?;
            backward_node(right, gradient, values)
        }
        Op::Mul(left, right) => {
            let left_values = eval_cpu(left, values, None)?;
            let right_values = eval_cpu(right, values, None)?;
            let left_gradient: Vec<_> = gradient
                .iter()
                .zip(&right_values)
                .map(|(grad, right)| grad * right)
                .collect();
            let right_gradient: Vec<_> = gradient
                .iter()
                .zip(&left_values)
                .map(|(grad, left)| grad * left)
                .collect();
            backward_node(left, left_gradient, values)?;
            backward_node(right, right_gradient, values)
        }
        Op::Relu(input) => {
            let input_values = eval_cpu(input, values, None)?;
            let input_gradient = gradient
                .into_iter()
                .zip(input_values)
                .map(|(grad, value)| if value > 0.0 { grad } else { 0.0 })
                .collect();
            backward_node(input, input_gradient, values)
        }
        Op::Reshape(input) => backward_node(input, gradient, values),
        Op::MatMul(left, right) => backward_matmul(left, right, &gradient, values),
        Op::Linear {
            input,
            weight,
            bias,
        } => backward_transformer(
            crate::transformer::Primitive::Linear,
            &[input, weight, bias],
            &gradient,
            values,
        ),
        Op::Gelu(input) => backward_transformer(
            crate::transformer::Primitive::Gelu,
            &[input],
            &gradient,
            values,
        ),
        Op::Tanh(input) => backward_transformer(
            crate::transformer::Primitive::Tanh,
            &[input],
            &gradient,
            values,
        ),
        Op::Embedding { ids, weight } => backward_transformer(
            crate::transformer::Primitive::Embedding,
            &[ids, weight],
            &gradient,
            values,
        ),
        Op::LayerNorm {
            input,
            weight,
            bias,
            epsilon,
        } => backward_transformer(
            crate::transformer::Primitive::LayerNorm { epsilon: *epsilon },
            &[input, weight, bias],
            &gradient,
            values,
        ),
        Op::SelectFirst(input) => backward_transformer(
            crate::transformer::Primitive::SelectFirst,
            &[input],
            &gradient,
            values,
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
        } => backward_transformer(
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
            &gradient,
            values,
        ),
        Op::Conv2d {
            input,
            weight,
            bias,
            stride,
            padding,
            groups,
        } => backward_conv2d(
            input, weight, bias, *stride, *padding, *groups, &gradient, values,
        ),
        Op::AvgPool2d {
            input,
            kernel,
            stride,
        } => backward_avg_pool2d(input, *kernel, *stride, &gradient, values),
        Op::CrossEntropy { logits, targets } => {
            backward_cross_entropy(logits, targets, gradient[0], values)
        }
        Op::BatchNorm2d {
            input,
            weight,
            bias,
            state,
            saved_statistics,
            training,
            epsilon,
            ..
        } => backward_batch_norm2d(
            input,
            weight,
            bias,
            state,
            saved_statistics,
            *training,
            *epsilon,
            &gradient,
            values,
        ),
        Op::Custom { inputs, operation } => {
            let input_values = inputs
                .iter()
                .map(|input| eval_cpu(input, values, None))
                .collect::<Result<Vec<_>>>()?;
            let custom_inputs = inputs
                .iter()
                .zip(&input_values)
                .map(|(input, values)| CustomInput {
                    shape: input.shape(),
                    values,
                })
                .collect::<Vec<_>>();
            let gradients = operation.backward(&custom_inputs, &gradient)?;
            if gradients.len() != inputs.len() {
                return Err(Error::Execution(
                    "custom op returned the wrong gradient count".into(),
                ));
            }
            for (input, gradient) in inputs.iter().zip(gradients) {
                if let Some(gradient) = gradient {
                    backward_node(input, gradient, values)?;
                }
            }
            Ok(())
        }
    }
}

fn backward_transformer(
    primitive: crate::transformer::Primitive,
    operands: &[&Tensor],
    output_gradient: &[f32],
    values: &mut HashMap<u64, Vec<f32>>,
) -> Result<()> {
    let input_values = operands
        .iter()
        .map(|input| eval_cpu(input, values, None))
        .collect::<Result<Vec<_>>>()?;
    let inputs = operands
        .iter()
        .zip(&input_values)
        .map(|(input, values)| CustomInput {
            shape: input.shape(),
            values,
        })
        .collect::<Vec<_>>();
    let gradients = crate::transformer::backward(primitive, &inputs, output_gradient)?;
    for (input, gradient) in operands.iter().zip(gradients) {
        if let Some(gradient) = gradient {
            backward_node(input, gradient, values)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn backward_batch_norm2d(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    state: &Arc<Mutex<BatchNormState>>,
    saved_statistics: &Arc<Mutex<Option<BatchNormStatistics>>>,
    training: bool,
    epsilon: f32,
    gradient: &[f32],
    values: &mut HashMap<u64, Vec<f32>>,
) -> Result<()> {
    let input_values = eval_cpu(input, values, None)?;
    let weight_values = eval_cpu(weight, values, None)?;
    let statistics = saved_statistics
        .lock()
        .map_err(|_| Error::Execution("BatchNorm statistics lock was poisoned".into()))?
        .clone()
        .map_or_else(
            || {
                if training {
                    batch_statistics(input, &input_values, epsilon)
                } else {
                    running_statistics(state, epsilon)
                }
            },
            Ok,
        )?;
    let channels = input.shape()[1];
    let spatial = input.shape()[2] * input.shape()[3];
    let samples = input.shape()[0] * spatial;
    #[allow(clippy::cast_precision_loss)]
    let samples_f32 = samples as f32;
    let mut input_gradient = vec![0.0; input.numel()];
    let mut weight_gradient = vec![0.0; channels];
    let mut bias_gradient = vec![0.0; channels];
    let mut gradient_sum = vec![0.0; channels];
    let mut gradient_normalized_sum = vec![0.0; channels];

    for (index, (&output_gradient, &input_value)) in gradient.iter().zip(&input_values).enumerate()
    {
        let channel = (index / spatial) % channels;
        let normalized = (input_value - statistics.mean[channel])
            * statistics.inverse_standard_deviation[channel];
        weight_gradient[channel] += output_gradient * normalized;
        bias_gradient[channel] += output_gradient;
        gradient_sum[channel] += output_gradient;
        gradient_normalized_sum[channel] += output_gradient * normalized;
    }

    for (index, (&output_gradient, &input_value)) in gradient.iter().zip(&input_values).enumerate()
    {
        let channel = (index / spatial) % channels;
        input_gradient[index] = if training {
            let normalized = (input_value - statistics.mean[channel])
                * statistics.inverse_standard_deviation[channel];
            weight_values[channel] * statistics.inverse_standard_deviation[channel] / samples_f32
                * (samples_f32 * output_gradient
                    - gradient_sum[channel]
                    - normalized * gradient_normalized_sum[channel])
        } else {
            output_gradient
                * weight_values[channel]
                * statistics.inverse_standard_deviation[channel]
        };
    }
    backward_node(input, input_gradient, values)?;
    backward_node(weight, weight_gradient, values)?;
    backward_node(bias, bias_gradient, values)
}

fn backward_matmul(
    left: &Tensor,
    right: &Tensor,
    gradient: &[f32],
    values: &mut HashMap<u64, Vec<f32>>,
) -> Result<()> {
    let left_values = eval_cpu(left, values, None)?;
    let right_values = eval_cpu(right, values, None)?;
    let [rows, inner, columns] = [left.shape()[0], left.shape()[1], right.shape()[1]];
    let mut left_gradient = vec![0.0; left.numel()];
    let mut right_gradient = vec![0.0; right.numel()];
    for row in 0..rows {
        for column in 0..columns {
            let output_gradient = gradient[row * columns + column];
            for inner_index in 0..inner {
                left_gradient[row * inner + inner_index] +=
                    output_gradient * right_values[inner_index * columns + column];
                right_gradient[inner_index * columns + column] +=
                    output_gradient * left_values[row * inner + inner_index];
            }
        }
    }
    backward_node(left, left_gradient, values)?;
    backward_node(right, right_gradient, values)
}

#[allow(clippy::too_many_arguments)]
fn backward_conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    stride: usize,
    padding: usize,
    groups: usize,
    gradient: &[f32],
    values: &mut HashMap<u64, Vec<f32>>,
) -> Result<()> {
    let input_values = eval_cpu(input, values, None)?;
    let weight_values = eval_cpu(weight, values, None)?;
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
    let mut input_gradient = vec![0.0; input.numel()];
    let mut weight_gradient = vec![0.0; weight.numel()];
    let mut bias_gradient = vec![0.0; bias.numel()];
    for batch_index in 0..batch {
        for (out_channel, bias_grad) in bias_gradient.iter_mut().enumerate().take(out_channels) {
            let group = out_channel / out_per_group;
            for out_y in 0..out_h {
                for out_x in 0..out_w {
                    let output_index = ((batch_index * out_channels + out_channel) * out_h + out_y)
                        * out_w
                        + out_x;
                    let output_gradient = gradient[output_index];
                    *bias_grad += output_gradient;
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
                                input_gradient[input_index] +=
                                    output_gradient * weight_values[weight_index];
                                weight_gradient[weight_index] +=
                                    output_gradient * input_values[input_index];
                            }
                        }
                    }
                }
            }
        }
    }
    backward_node(input, input_gradient, values)?;
    backward_node(weight, weight_gradient, values)?;
    backward_node(bias, bias_gradient, values)
}

fn backward_avg_pool2d(
    input: &Tensor,
    kernel: [usize; 2],
    stride: [usize; 2],
    gradient: &[f32],
    values: &mut HashMap<u64, Vec<f32>>,
) -> Result<()> {
    let [batch, channels, height, width] = [
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    ];
    let out_h = (height - kernel[0]) / stride[0] + 1;
    let out_w = (width - kernel[1]) / stride[1] + 1;
    #[allow(clippy::cast_precision_loss)]
    let denominator = (kernel[0] * kernel[1]) as f32;
    let mut input_gradient = vec![0.0; input.numel()];
    for batch_index in 0..batch {
        for channel in 0..channels {
            for out_y in 0..out_h {
                for out_x in 0..out_w {
                    let output_index =
                        ((batch_index * channels + channel) * out_h + out_y) * out_w + out_x;
                    let contribution = gradient[output_index] / denominator;
                    for kernel_y in 0..kernel[0] {
                        for kernel_x in 0..kernel[1] {
                            let in_y = out_y * stride[0] + kernel_y;
                            let in_x = out_x * stride[1] + kernel_x;
                            let input_index =
                                ((batch_index * channels + channel) * height + in_y) * width + in_x;
                            input_gradient[input_index] += contribution;
                        }
                    }
                }
            }
        }
    }
    backward_node(input, input_gradient, values)
}

fn backward_cross_entropy(
    logits: &Tensor,
    targets: &Tensor,
    outer_gradient: f32,
    values: &mut HashMap<u64, Vec<f32>>,
) -> Result<()> {
    let logits_values = eval_cpu(logits, values, None)?;
    let targets_values = eval_cpu(targets, values, None)?;
    let batch = logits.shape()[0];
    let classes = logits.shape()[1];
    #[allow(clippy::cast_precision_loss)]
    let scale = outer_gradient / batch as f32;
    let mut logits_gradient = vec![0.0; logits.numel()];
    for batch_index in 0..batch {
        let target = class_index(targets_values[batch_index], classes)?;
        let row = &logits_values[batch_index * classes..(batch_index + 1) * classes];
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let normalizer: f32 = row.iter().map(|value| (*value - maximum).exp()).sum();
        for class in 0..classes {
            let probability = (row[class] - maximum).exp() / normalizer;
            logits_gradient[batch_index * classes + class] =
                scale * (probability - f32::from(class == target));
        }
    }
    backward_node(logits, logits_gradient, values)
}
