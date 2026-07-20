use crate::{Error, Result};
use std::collections::{HashMap, HashSet};
use std::ops::{Add, Mul};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Device {
    #[default]
    Cpu,
    Cuda(usize),
}

#[derive(Clone, Debug)]
#[must_use]
pub struct Tensor {
    pub(crate) node: Arc<Node>,
}

#[derive(Debug)]
pub(crate) struct Node {
    pub(crate) id: u64,
    pub(crate) shape: Vec<usize>,
    pub(crate) device: Device,
    pub(crate) op: Op,
    grad: Mutex<Option<Vec<f32>>>,
}

#[derive(Debug)]
pub(crate) struct BatchNormState {
    pub(crate) running_mean: Vec<f32>,
    pub(crate) running_variance: Vec<f32>,
}

#[derive(Clone, Debug)]
struct BatchNormStatistics {
    mean: Vec<f32>,
    inverse_standard_deviation: Vec<f32>,
}

#[derive(Debug)]
pub(crate) enum Op {
    Data(Arc<[f32]>),
    Placeholder(usize),
    Add(Tensor, Tensor),
    Mul(Tensor, Tensor),
    Relu(Tensor),
    MatMul(Tensor, Tensor),
    Conv2d {
        input: Tensor,
        weight: Tensor,
        bias: Tensor,
        stride: usize,
        padding: usize,
        groups: usize,
    },
    AvgPool2d {
        input: Tensor,
        kernel: [usize; 2],
        stride: [usize; 2],
    },
    Reshape(Tensor),
    CrossEntropy {
        logits: Tensor,
        targets: Tensor,
    },
    BatchNorm2d {
        input: Tensor,
        weight: Tensor,
        bias: Tensor,
        state: Arc<Mutex<BatchNormState>>,
        saved_statistics: Arc<Mutex<Option<BatchNormStatistics>>>,
        training: bool,
        momentum: f32,
        epsilon: f32,
    },
}

impl Tensor {
    /// Constructs a CPU tensor from row-major data.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidShape`] when `data.len()` does not equal the
    /// shape's element count or that count overflows.
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        validate_numel(data.len(), &shape)?;
        Ok(Self::new(shape, Device::Cpu, Op::Data(data.into())))
    }

    /// Constructs a zero-filled CPU tensor.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidShape`] for an empty or overflowing shape.
    pub fn zeros(shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = shape.into();
        let len = checked_numel(&shape)?;
        Self::from_vec(vec![0.0; len], shape)
    }

    /// Constructs a one-filled CPU tensor.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidShape`] for an empty or overflowing shape.
    pub fn ones(shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = shape.into();
        let len = checked_numel(&shape)?;
        Self::from_vec(vec![1.0; len], shape)
    }

    #[allow(clippy::cast_precision_loss)]
    pub fn arange(end: usize) -> Self {
        let data: Vec<f32> = (0..end).map(|value| value as f32).collect();
        Self::new(vec![end], Device::Cpu, Op::Data(data.into()))
    }

    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.node.shape
    }

    #[must_use]
    pub fn numel(&self) -> usize {
        self.node.shape.iter().product()
    }

    #[must_use]
    pub fn device(&self) -> Device {
        self.node.device
    }

    /// Moves the lazy graph to a device. Data is transferred only on evaluation.
    pub fn to(&self, device: Device) -> Self {
        clone_to_device(self, device, &mut HashMap::new())
    }

    /// Adds two equal-shaped tensors lazily.
    ///
    /// # Errors
    ///
    /// Returns an error when shapes or devices differ.
    pub fn add(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, Op::Add)
    }

    /// Multiplies two equal-shaped tensors element by element.
    ///
    /// # Errors
    ///
    /// Returns an error when shapes or devices differ.
    pub fn mul(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, Op::Mul)
    }

    pub fn relu(&self) -> Self {
        Self::new(
            self.node.shape.clone(),
            self.node.device,
            Op::Relu(self.clone()),
        )
    }

    /// Performs lazy row-major rank-2 matrix multiplication.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible shapes, ranks, or devices.
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        if self.device() != rhs.device() {
            return Err(Error::DeviceMismatch);
        }
        if self.shape().len() != 2 || rhs.shape().len() != 2 {
            return Err(Error::InvalidShape(
                "matmul expects two rank-2 tensors".into(),
            ));
        }
        let [m, k] = [self.shape()[0], self.shape()[1]];
        let [rhs_k, n] = [rhs.shape()[0], rhs.shape()[1]];
        if k != rhs_k {
            return Err(Error::InvalidShape(format!(
                "matmul dimensions do not align: [{m}, {k}] @ [{rhs_k}, {n}]"
            )));
        }
        Ok(Self::new(
            vec![m, n],
            self.device(),
            Op::MatMul(self.clone(), rhs.clone()),
        ))
    }

    /// Applies an NCHW grouped 2D cross-correlation.
    ///
    /// `weight` is `[out_channels, in_channels / groups, kernel_h, kernel_w]`
    /// and `bias` is `[out_channels]`.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid ranks, groups, channels, shapes, or devices.
    pub fn conv2d(
        &self,
        weight: &Self,
        bias: &Self,
        stride: usize,
        padding: usize,
        groups: usize,
    ) -> Result<Self> {
        if self.device() != weight.device() || self.device() != bias.device() {
            return Err(Error::DeviceMismatch);
        }
        if self.shape().len() != 4 || weight.shape().len() != 4 || bias.shape().len() != 1 {
            return Err(Error::InvalidShape(
                "conv2d expects NCHW input, OIHW weight, and rank-1 bias".into(),
            ));
        }
        if stride == 0 || groups == 0 {
            return Err(Error::InvalidShape(
                "conv2d stride and groups must be non-zero".into(),
            ));
        }
        let [batch, in_channels, height, width] = [
            self.shape()[0],
            self.shape()[1],
            self.shape()[2],
            self.shape()[3],
        ];
        let [out_channels, weight_channels, kernel_h, kernel_w] = [
            weight.shape()[0],
            weight.shape()[1],
            weight.shape()[2],
            weight.shape()[3],
        ];
        if kernel_h == 0 || kernel_w == 0 || kernel_h != kernel_w {
            return Err(Error::InvalidShape(
                "conv2d currently requires a non-zero square kernel".into(),
            ));
        }
        if in_channels % groups != 0
            || out_channels % groups != 0
            || weight_channels != in_channels / groups
            || bias.shape()[0] != out_channels
        {
            return Err(Error::InvalidShape(
                "conv2d group/channel dimensions are inconsistent".into(),
            ));
        }
        let padded_h = height
            .checked_add(padding.saturating_mul(2))
            .ok_or_else(|| Error::InvalidShape("conv2d padded height overflow".into()))?;
        let padded_w = width
            .checked_add(padding.saturating_mul(2))
            .ok_or_else(|| Error::InvalidShape("conv2d padded width overflow".into()))?;
        if padded_h < kernel_h || padded_w < kernel_w {
            return Err(Error::InvalidShape("conv2d kernel exceeds input".into()));
        }
        let out_h = (padded_h - kernel_h) / stride + 1;
        let out_w = (padded_w - kernel_w) / stride + 1;
        Ok(Self::new(
            vec![batch, out_channels, out_h, out_w],
            self.device(),
            Op::Conv2d {
                input: self.clone(),
                weight: weight.clone(),
                bias: bias.clone(),
                stride,
                padding,
                groups,
            },
        ))
    }

    /// Applies a non-overlapping or strided NCHW average pool.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid rank, zero sizes, or an oversized kernel.
    pub fn avg_pool2d(&self, kernel: [usize; 2], stride: [usize; 2]) -> Result<Self> {
        if self.shape().len() != 4 {
            return Err(Error::InvalidShape("avg_pool2d expects NCHW input".into()));
        }
        if kernel.contains(&0) || stride.contains(&0) {
            return Err(Error::InvalidShape(
                "avg_pool2d kernel and stride must be non-zero".into(),
            ));
        }
        let [height, width] = [self.shape()[2], self.shape()[3]];
        if kernel[0] > height || kernel[1] > width {
            return Err(Error::InvalidShape(
                "avg_pool2d kernel exceeds input".into(),
            ));
        }
        Ok(Self::new(
            vec![
                self.shape()[0],
                self.shape()[1],
                (height - kernel[0]) / stride[0] + 1,
                (width - kernel[1]) / stride[1] + 1,
            ],
            self.device(),
            Op::AvgPool2d {
                input: self.clone(),
                kernel,
                stride,
            },
        ))
    }

    /// Averages the complete spatial extent of an NCHW tensor.
    ///
    /// # Errors
    ///
    /// Returns an error unless the input is rank 4.
    pub fn global_avg_pool2d(&self) -> Result<Self> {
        if self.shape().len() != 4 {
            return Err(Error::InvalidShape(
                "global_avg_pool2d expects NCHW input".into(),
            ));
        }
        self.avg_pool2d(
            [self.shape()[2], self.shape()[3]],
            [self.shape()[2], self.shape()[3]],
        )
    }

    /// Returns a lazy view with a different shape and identical element count.
    ///
    /// # Errors
    ///
    /// Returns an error when the shape has a different element count.
    pub fn reshape(&self, shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = shape.into();
        if checked_numel(&shape)? != self.numel() {
            return Err(Error::InvalidShape(
                "reshape must preserve the element count".into(),
            ));
        }
        Ok(Self::new(shape, self.device(), Op::Reshape(self.clone())))
    }

    /// Computes mean sparse cross-entropy from `[batch, classes]` logits.
    ///
    /// Targets are represented by a rank-1 tensor containing integer class
    /// indices as `f32` values.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid shapes, devices, or target values.
    pub fn cross_entropy(&self, targets: &Self) -> Result<Self> {
        if self.device() != targets.device() {
            return Err(Error::DeviceMismatch);
        }
        if self.shape().len() != 2
            || targets.shape().len() != 1
            || self.shape()[0] != targets.shape()[0]
        {
            return Err(Error::InvalidShape(
                "cross_entropy expects [batch, classes] logits and [batch] targets".into(),
            ));
        }
        Ok(Self::new(
            vec![1],
            self.device(),
            Op::CrossEntropy {
                logits: self.clone(),
                targets: targets.clone(),
            },
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batch_norm2d(
        &self,
        weight: &Self,
        bias: &Self,
        state: Arc<Mutex<BatchNormState>>,
        training: bool,
        momentum: f32,
        epsilon: f32,
    ) -> Result<Self> {
        if self.device() != weight.device() || self.device() != bias.device() {
            return Err(Error::DeviceMismatch);
        }
        if self.shape().len() != 4 || weight.shape().len() != 1 || bias.shape().len() != 1 {
            return Err(Error::InvalidShape(
                "batch_norm2d expects NCHW input and rank-1 weight/bias".into(),
            ));
        }
        let channels = self.shape()[1];
        if weight.shape() != [channels] || bias.shape() != [channels] {
            return Err(Error::InvalidShape(
                "batch_norm2d weight/bias must match the channel count".into(),
            ));
        }
        if !momentum.is_finite() || !(0.0..=1.0).contains(&momentum) {
            return Err(Error::Execution(
                "batch_norm2d momentum must be finite and in 0..=1".into(),
            ));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Error::Execution(
                "batch_norm2d epsilon must be positive and finite".into(),
            ));
        }
        {
            let buffers = state
                .lock()
                .map_err(|_| Error::Execution("BatchNorm state lock was poisoned".into()))?;
            if buffers.running_mean.len() != channels
                || buffers.running_variance.len() != channels
            {
                return Err(Error::InvalidShape(
                    "batch_norm2d running statistics must match the channel count".into(),
                ));
            }
        }
        Ok(Self::new(
            self.shape().to_vec(),
            self.device(),
            Op::BatchNorm2d {
                input: self.clone(),
                weight: weight.clone(),
                bias: bias.clone(),
                state,
                saved_statistics: Arc::new(Mutex::new(None)),
                training,
                momentum,
                epsilon,
            },
        ))
    }

    /// Runs reverse-mode automatic differentiation from a scalar tensor.
    ///
    /// CUDA graphs currently use a correct CPU fallback for backward while
    /// retaining CUDA forward execution.
    ///
    /// # Errors
    ///
    /// Returns an error unless this tensor has one element or graph evaluation fails.
    pub fn backward(&self) -> Result<()> {
        if self.numel() != 1 {
            return Err(Error::Execution(
                "backward requires a scalar loss tensor".into(),
            ));
        }
        clear_graph_grads(self, &mut HashSet::new())?;
        backward_node(self, vec![1.0], &mut HashMap::new())
    }

    /// Returns the accumulated gradient, if this tensor participated in backward.
    ///
    /// # Errors
    ///
    /// Returns an error if the gradient lock was poisoned.
    pub fn grad(&self) -> Result<Option<Vec<f32>>> {
        self.node
            .grad
            .lock()
            .map(|gradient| gradient.clone())
            .map_err(|_| Error::Execution("gradient lock was poisoned".into()))
    }

    /// Clears this tensor's accumulated gradient.
    ///
    /// # Errors
    ///
    /// Returns an error if the gradient lock was poisoned.
    pub fn zero_grad(&self) -> Result<()> {
        *self
            .node
            .grad
            .lock()
            .map_err(|_| Error::Execution("gradient lock was poisoned".into()))? = None;
        Ok(())
    }

    /// Returns the single materialized value.
    ///
    /// # Errors
    ///
    /// Returns an error for non-scalar tensors or backend evaluation failure.
    pub fn item(&self) -> Result<f32> {
        if self.numel() != 1 {
            return Err(Error::InvalidShape("item requires one element".into()));
        }
        self.to_vec().map(|values| values[0])
    }

    /// Materializes this lazy tensor on the host.
    ///
    /// # Errors
    ///
    /// Returns an error when the chosen backend is unavailable or execution
    /// fails.
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        match self.device() {
            Device::Cpu => eval_cpu(self, &mut HashMap::new(), None),
            Device::Cuda(device) => {
                #[cfg(feature = "cuda")]
                {
                    crate::cuda::eval(self, device)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = device;
                    Err(Error::CudaUnavailable)
                }
            }
        }
    }

    pub(crate) fn placeholder(slot: usize, shape: Vec<usize>, device: Device) -> Self {
        Self::new(shape, device, Op::Placeholder(slot))
    }

    pub(crate) fn new(shape: Vec<usize>, device: Device, op: Op) -> Self {
        Self {
            node: Arc::new(Node {
                id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
                shape,
                device,
                op,
                grad: Mutex::new(None),
            }),
        }
    }

    fn binary(&self, rhs: &Self, make_op: fn(Tensor, Tensor) -> Op) -> Result<Self> {
        if self.device() != rhs.device() {
            return Err(Error::DeviceMismatch);
        }
        if self.shape() != rhs.shape() {
            return Err(Error::InvalidShape(format!(
                "elementwise operands differ: {:?} and {:?}",
                self.shape(),
                rhs.shape()
            )));
        }
        Ok(Self::new(
            self.node.shape.clone(),
            self.device(),
            make_op(self.clone(), rhs.clone()),
        ))
    }
}

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Self::Output {
        Tensor::add(self, rhs).expect("tensor addition failed")
    }
}

impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        Tensor::mul(self, rhs).expect("tensor multiplication failed")
    }
}

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
    };
    cache.insert(tensor.node.id, value.clone());
    Ok(value)
}

fn clone_to_device(tensor: &Tensor, device: Device, cache: &mut HashMap<u64, Tensor>) -> Tensor {
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

fn batch_statistics(
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
        for channel in 0..channels {
            let start = (batch * channels + channel) * spatial;
            mean[channel] += input_data[start..start + spatial].iter().sum::<f32>();
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
        buffers.running_variance[channel] = (1.0 - momentum)
            * buffers.running_variance[channel]
            + momentum * biased_variance * unbiased_correction;
    }
    Ok(())
}

fn running_statistics(
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

fn class_index(value: f32, classes: usize) -> Result<usize> {
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

fn clear_graph_grads(tensor: &Tensor, visited: &mut HashSet<u64>) -> Result<()> {
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
        Op::Relu(input) | Op::Reshape(input) | Op::AvgPool2d { input, .. } => {
            clear_graph_grads(input, visited)?;
        }
        Op::Conv2d {
            input,
            weight,
            bias,
            ..
        } => {
            clear_graph_grads(input, visited)?;
            clear_graph_grads(weight, visited)?;
            clear_graph_grads(bias, visited)?;
        }
        Op::CrossEntropy { logits, targets } => {
            clear_graph_grads(logits, visited)?;
            clear_graph_grads(targets, visited)?;
        }
        Op::BatchNorm2d {
            input,
            weight,
            bias,
            ..
        } => {
            clear_graph_grads(input, visited)?;
            clear_graph_grads(weight, visited)?;
            clear_graph_grads(bias, visited)?;
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

fn backward_node(
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
    }
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

    for (index, (&output_gradient, &input_value)) in
        gradient.iter().zip(&input_values).enumerate()
    {
        let channel = (index / spatial) % channels;
        let normalized = (input_value - statistics.mean[channel])
            * statistics.inverse_standard_deviation[channel];
        weight_gradient[channel] += output_gradient * normalized;
        bias_gradient[channel] += output_gradient;
        gradient_sum[channel] += output_gradient;
        gradient_normalized_sum[channel] += output_gradient * normalized;
    }

    for (index, (&output_gradient, &input_value)) in
        gradient.iter().zip(&input_values).enumerate()
    {
        let channel = (index / spatial) % channels;
        input_gradient[index] = if training {
            let normalized = (input_value - statistics.mean[channel])
                * statistics.inverse_standard_deviation[channel];
            weight_values[channel] * statistics.inverse_standard_deviation[channel]
                / samples_f32
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

fn zip_map(lhs: Vec<f32>, rhs: Vec<f32>, op: impl Fn(f32, f32) -> f32) -> Vec<f32> {
    lhs.into_iter().zip(rhs).map(|(a, b)| op(a, b)).collect()
}

fn validate_numel(data_len: usize, shape: &[usize]) -> Result<()> {
    let expected = checked_numel(shape)?;
    if data_len != expected {
        return Err(Error::InvalidShape(format!(
            "shape {shape:?} contains {expected} elements, but data contains {data_len}"
        )));
    }
    Ok(())
}

fn checked_numel(shape: &[usize]) -> Result<usize> {
    if shape.is_empty() {
        return Err(Error::InvalidShape(
            "scalar shapes are not implemented".into(),
        ));
    }
    shape.iter().try_fold(1usize, |total, &dim| {
        total
            .checked_mul(dim)
            .ok_or_else(|| Error::InvalidShape("element count overflow".into()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lazy_elementwise_graph_evaluates() {
        let x = Tensor::from_vec(vec![-1.0, 2.0, -3.0, 4.0], vec![2, 2]).unwrap();
        let y = Tensor::ones(vec![2, 2]).unwrap();
        let output = (&x + &y).relu();
        assert_eq!(output.to_vec().unwrap(), vec![0.0, 3.0, 0.0, 5.0]);
    }

    #[test]
    fn matrix_multiply_evaluates() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        assert_eq!(
            a.matmul(&b).unwrap().to_vec().unwrap(),
            vec![19.0, 22.0, 43.0, 50.0]
        );
    }

    #[test]
    fn grouped_convolution_and_average_pool_evaluate() {
        let input =
            Tensor::from_vec((1_i16..=9).map(f32::from).collect(), vec![1, 1, 3, 3]).unwrap();
        let weight = Tensor::from_vec(vec![1.0; 4], vec![1, 1, 2, 2]).unwrap();
        let bias = Tensor::zeros(vec![1]).unwrap();
        let convolved = input.conv2d(&weight, &bias, 1, 0, 1).unwrap();
        assert_eq!(convolved.shape(), &[1, 1, 2, 2]);
        assert_eq!(convolved.to_vec().unwrap(), vec![12.0, 16.0, 24.0, 28.0]);

        let pooled = Tensor::from_vec((1_i16..=16).map(f32::from).collect(), vec![1, 1, 4, 4])
            .unwrap()
            .avg_pool2d([2, 2], [2, 2])
            .unwrap();
        assert_eq!(pooled.to_vec().unwrap(), vec![3.5, 5.5, 11.5, 13.5]);

        let depthwise_input =
            Tensor::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![1, 2, 2, 2]).unwrap();
        let depthwise_weight = Tensor::from_vec(vec![2., 3.], vec![2, 1, 1, 1]).unwrap();
        let depthwise_bias = Tensor::zeros(vec![2]).unwrap();
        assert_eq!(
            depthwise_input
                .conv2d(&depthwise_weight, &depthwise_bias, 1, 0, 2)
                .unwrap()
                .to_vec()
                .unwrap(),
            vec![2., 4., 6., 8., 15., 18., 21., 24.]
        );
    }

    #[test]
    fn cross_entropy_backpropagates_softmax_gradient() {
        let logits = Tensor::zeros(vec![2, 2]).unwrap();
        let targets = Tensor::from_vec(vec![0.0, 1.0], vec![2]).unwrap();
        let loss = logits.cross_entropy(&targets).unwrap();
        assert!((loss.item().unwrap() - 2.0_f32.ln()).abs() < 1e-6);
        loss.backward().unwrap();
        let gradient = logits.grad().unwrap().unwrap();
        assert_eq!(gradient, vec![-0.25, 0.25, 0.25, -0.25]);
    }
}
