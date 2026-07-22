use crate::{Error, Result};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

mod autograd;
mod evaluation;
mod graph;
mod operators;
mod shape;

use autograd::{backward_node, clear_graph_grads};
use evaluation::clone_to_device;
pub(crate) use evaluation::eval_cpu;
pub(crate) use graph::{BatchNormState, BatchNormStatistics, NEXT_ID, Node, Op};
pub use graph::{CustomInput, CustomOp, Device, Tensor};
use shape::{checked_numel, validate_numel};

impl Tensor {
    /// Creates a differentiable custom CPU operation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidShape`] for an invalid output shape or
    /// [`Error::DeviceMismatch`] when inputs are placed on different devices.
    pub fn custom(
        inputs: Vec<Tensor>,
        shape: impl Into<Vec<usize>>,
        operation: Arc<dyn CustomOp>,
    ) -> Result<Self> {
        let shape = shape.into();
        checked_numel(&shape)?;
        let device = inputs.first().map_or(Device::Cpu, Tensor::device);
        if inputs.iter().any(|input| input.device() != device) {
            return Err(Error::DeviceMismatch);
        }
        Ok(Self::new(shape, device, Op::Custom { inputs, operation }))
    }
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
            if buffers.running_mean.len() != channels || buffers.running_variance.len() != channels
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
        match self.device() {
            Device::Cpu => {
                clear_graph_grads(self, &mut HashSet::new())?;
                backward_node(self, vec![1.0], &mut HashMap::new())
            }
            Device::Cuda(device) => {
                #[cfg(feature = "cuda")]
                {
                    crate::cuda::backward(self, device)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = device;
                    Err(Error::CudaUnavailable)
                }
            }
        }
    }

    /// Returns the accumulated gradient, if this tensor participated in backward.
    ///
    /// # Errors
    ///
    /// Returns an error if the gradient lock was poisoned.
    pub fn grad(&self) -> Result<Option<Vec<f32>>> {
        let host_gradient = self
            .node
            .grad
            .lock()
            .map(|gradient| gradient.clone())
            .map_err(|_| Error::Execution("gradient lock was poisoned".into()))?;
        if host_gradient.is_some() || self.device() == Device::Cpu {
            return Ok(host_gradient);
        }
        #[cfg(feature = "cuda")]
        if let Device::Cuda(device) = self.device() {
            return crate::cuda::gradient(self, device);
        }
        Ok(None)
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
        #[cfg(feature = "cuda")]
        if let Device::Cuda(device) = self.device() {
            crate::cuda::zero_gradient(self, device)?;
        }
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

    /// Evaluates pending work and waits for this tensor's device stream.
    ///
    /// This is intended for profiling and explicit synchronization; normal
    /// execution should remain asynchronous.
    ///
    /// # Errors
    ///
    /// Returns an error when evaluation or device synchronization fails.
    pub fn synchronize(&self) -> Result<()> {
        match self.device() {
            Device::Cpu => {
                let _ = self.to_vec()?;
                Ok(())
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(device) => crate::cuda::materialize(self, device),
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => Err(Error::CudaUnavailable),
        }
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

impl Drop for Node {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(device) = self.device {
            crate::cuda::release_node(self.id, device);
        }
    }
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

    #[test]
    fn batch_norm_input_gradient_matches_finite_difference() {
        fn loss(input_data: Vec<f32>) -> f32 {
            let input = Tensor::from_vec(input_data, vec![3, 2, 1, 1]).unwrap();
            let weight = Tensor::from_vec(vec![1.2, 0.7], vec![2]).unwrap();
            let bias = Tensor::from_vec(vec![0.1, -0.2], vec![2]).unwrap();
            let state = Arc::new(Mutex::new(BatchNormState {
                running_mean: vec![0.0; 2],
                running_variance: vec![1.0; 2],
                #[cfg(feature = "cuda")]
                device: Device::Cpu,
            }));
            let targets = Tensor::from_vec(vec![0.0, 1.0, 0.0], vec![3]).unwrap();
            input
                .batch_norm2d(&weight, &bias, state, true, 0.1, 1e-5)
                .unwrap()
                .reshape(vec![3, 2])
                .unwrap()
                .cross_entropy(&targets)
                .unwrap()
                .item()
                .unwrap()
        }

        let data = vec![1.0, 4.0, 2.0, 0.0, -1.0, 3.0];
        let input = Tensor::from_vec(data.clone(), vec![3, 2, 1, 1]).unwrap();
        let weight = Tensor::from_vec(vec![1.2, 0.7], vec![2]).unwrap();
        let bias = Tensor::from_vec(vec![0.1, -0.2], vec![2]).unwrap();
        let state = Arc::new(Mutex::new(BatchNormState {
            running_mean: vec![0.0; 2],
            running_variance: vec![1.0; 2],
            #[cfg(feature = "cuda")]
            device: Device::Cpu,
        }));
        let targets = Tensor::from_vec(vec![0.0, 1.0, 0.0], vec![3]).unwrap();
        let loss_tensor = input
            .batch_norm2d(&weight, &bias, state, true, 0.1, 1e-5)
            .unwrap()
            .reshape(vec![3, 2])
            .unwrap()
            .cross_entropy(&targets)
            .unwrap();
        loss_tensor.backward().unwrap();
        let analytic = input.grad().unwrap().unwrap();
        let step = 1e-3;
        for index in 0..data.len() {
            let mut positive = data.clone();
            positive[index] += step;
            let mut negative = data.clone();
            negative[index] -= step;
            let numerical = (loss(positive) - loss(negative)) / (2.0 * step);
            assert!(
                (analytic[index] - numerical).abs() < 2e-3,
                "gradient {index}: analytic={} numerical={numerical}",
                analytic[index]
            );
        }
        assert!(weight.grad().unwrap().is_some());
        assert!(bias.grad().unwrap().is_some());
    }
}
