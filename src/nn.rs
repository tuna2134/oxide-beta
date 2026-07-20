//! Minimal neural-network layers used by the paper models.

use crate::tensor::BatchNormState;
use crate::{Device, Error, Result, Tensor};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

static NEXT_PARAMETER_ID: AtomicU64 = AtomicU64::new(1);

pub trait Module {
    /// Runs a lazy forward pass.
    ///
    /// # Errors
    ///
    /// Returns an error when an input shape/device is incompatible with the module.
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
}

/// Switches stateful layers between training and inference behavior.
pub trait ModuleMode {
    fn set_training(&mut self, training: bool);

    fn train(&mut self) {
        self.set_training(true);
    }

    fn eval(&mut self) {
        self.set_training(false);
    }
}

/// A trainable tensor with an optimizer-stable identity.
#[derive(Clone, Debug)]
pub struct Parameter {
    id: u64,
    value: Tensor,
}

impl Parameter {
    #[must_use]
    pub fn new(value: Tensor) -> Self {
        Self {
            id: NEXT_PARAMETER_ID.fetch_add(1, Ordering::Relaxed),
            value,
        }
    }

    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn value(&self) -> &Tensor {
        &self.value
    }

    /// Replaces the parameter data while retaining optimizer identity.
    ///
    /// # Errors
    ///
    /// Returns an error when data length differs from the parameter shape.
    pub fn replace_data(&mut self, data: Vec<f32>) -> Result<()> {
        let device = self.value.device();
        self.value = Tensor::from_vec(data, self.value.shape().to_vec())?.to(device);
        Ok(())
    }

    pub(crate) fn replace_value(&mut self, value: Tensor) -> Result<()> {
        if value.shape() != self.value.shape() || value.device() != self.value.device() {
            return Err(Error::Execution(
                "replacement parameter shape/device mismatch".into(),
            ));
        }
        self.value = value;
        Ok(())
    }
}

/// Provides structured traversal without exposing a model's internal layout.
pub trait Trainable {
    fn visit_parameters(&self, visitor: &mut dyn FnMut(&Parameter));
    fn visit_parameters_mut(&mut self, visitor: &mut dyn FnMut(&mut Parameter));

    /// Visits persistent, non-optimizer state such as running statistics.
    ///
    /// # Errors
    ///
    /// Returns an error when a state lock is poisoned.
    fn visit_buffers(&self, _visitor: &mut dyn FnMut(&[usize], &[f32])) -> Result<()> {
        Ok(())
    }
}

/// NCHW Batch Normalization with affine parameters and running statistics.
#[derive(Clone, Debug)]
pub struct BatchNorm2d {
    weight: Parameter,
    bias: Parameter,
    state: Arc<Mutex<BatchNormState>>,
    training: bool,
    momentum: f32,
    epsilon: f32,
}

impl BatchNorm2d {
    /// Creates a `BatchNorm` layer with γ=1, β=0, running mean=0, and variance=1.
    ///
    /// # Errors
    ///
    /// Returns an error when `channels` is zero.
    pub fn new(channels: usize, device: Device) -> Result<Self> {
        if channels == 0 {
            return Err(Error::InvalidShape(
                "BatchNorm2d channels must be non-zero".into(),
            ));
        }
        Ok(Self {
            weight: Parameter::new(Tensor::ones(vec![channels])?.to(device)),
            bias: Parameter::new(Tensor::zeros(vec![channels])?.to(device)),
            state: Arc::new(Mutex::new(BatchNormState {
                running_mean: vec![0.0; channels],
                running_variance: vec![1.0; channels],
                device,
            })),
            training: true,
            momentum: 0.1,
            epsilon: 1e-5,
        })
    }

    #[must_use]
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Returns a copy of the running mean.
    ///
    /// # Errors
    ///
    /// Returns an error when the state lock is poisoned.
    pub fn running_mean(&self) -> Result<Vec<f32>> {
        #[cfg(feature = "cuda")]
        crate::cuda::sync_batch_norm_state(&self.state)?;
        self.state
            .lock()
            .map(|state| state.running_mean.clone())
            .map_err(|_| Error::Execution("BatchNorm state lock was poisoned".into()))
    }

    /// Returns a copy of the running variance.
    ///
    /// # Errors
    ///
    /// Returns an error when the state lock is poisoned.
    pub fn running_variance(&self) -> Result<Vec<f32>> {
        #[cfg(feature = "cuda")]
        crate::cuda::sync_batch_norm_state(&self.state)?;
        self.state
            .lock()
            .map(|state| state.running_variance.clone())
            .map_err(|_| Error::Execution("BatchNorm state lock was poisoned".into()))
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.batch_norm2d(
            self.weight.value(),
            self.bias.value(),
            self.state.clone(),
            self.training,
            self.momentum,
            self.epsilon,
        )
    }
}

impl ModuleMode for BatchNorm2d {
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

impl Trainable for BatchNorm2d {
    fn visit_parameters(&self, visitor: &mut dyn FnMut(&Parameter)) {
        visitor(&self.weight);
        visitor(&self.bias);
    }

    fn visit_parameters_mut(&mut self, visitor: &mut dyn FnMut(&mut Parameter)) {
        visitor(&mut self.weight);
        visitor(&mut self.bias);
    }

    fn visit_buffers(&self, visitor: &mut dyn FnMut(&[usize], &[f32])) -> Result<()> {
        #[cfg(feature = "cuda")]
        crate::cuda::sync_batch_norm_state(&self.state)?;
        let state = self
            .state
            .lock()
            .map_err(|_| Error::Execution("BatchNorm state lock was poisoned".into()))?;
        let shape = [state.running_mean.len()];
        visitor(&shape, &state.running_mean);
        visitor(&shape, &state.running_variance);
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Conv2d {
    weight: Parameter,
    bias: Parameter,
    stride: usize,
    padding: usize,
    groups: usize,
}

impl Conv2d {
    /// Creates a convolution with deterministic variance-scaled weights and zero bias.
    ///
    /// Bias is always present so a trained `BatchNorm` can be folded into this
    /// layer before inference.
    ///
    /// # Errors
    ///
    /// Returns an error for inconsistent channels, groups, or a zero kernel.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        device: Device,
    ) -> Result<Self> {
        if kernel_size == 0
            || groups == 0
            || in_channels % groups != 0
            || out_channels % groups != 0
        {
            return Err(Error::InvalidShape(
                "invalid Conv2d channel, group, or kernel configuration".into(),
            ));
        }
        let weight_shape = vec![out_channels, in_channels / groups, kernel_size, kernel_size];
        let weight = Parameter::new(
            Tensor::from_vec(
                initialized_weights(&weight_shape, in_channels / groups),
                weight_shape,
            )?
            .to(device),
        );
        let bias = Parameter::new(Tensor::zeros(vec![out_channels])?.to(device));
        Ok(Self {
            weight,
            bias,
            stride,
            padding: kernel_size / 2,
            groups,
        })
    }

    /// Creates a convolution from explicit tensors, primarily for loading weights.
    ///
    /// # Errors
    ///
    /// Returns an error when a small validation convolution cannot be constructed.
    pub fn from_tensors(
        weight: Tensor,
        bias: Tensor,
        stride: usize,
        padding: usize,
        groups: usize,
    ) -> Result<Self> {
        if weight.shape().len() != 4 || bias.shape().len() != 1 {
            return Err(Error::InvalidShape(
                "Conv2d weight must be OIHW and bias rank 1".into(),
            ));
        }
        Ok(Self {
            weight: Parameter::new(weight),
            bias: Parameter::new(bias),
            stride,
            padding,
            groups,
        })
    }

    pub fn weight(&self) -> &Tensor {
        self.weight.value()
    }

    pub fn bias(&self) -> &Tensor {
        self.bias.value()
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.conv2d(
            self.weight.value(),
            self.bias.value(),
            self.stride,
            self.padding,
            self.groups,
        )
    }
}

impl Trainable for Conv2d {
    fn visit_parameters(&self, visitor: &mut dyn FnMut(&Parameter)) {
        visitor(&self.weight);
        visitor(&self.bias);
    }

    fn visit_parameters_mut(&mut self, visitor: &mut dyn FnMut(&mut Parameter)) {
        visitor(&mut self.weight);
        visitor(&mut self.bias);
    }
}

#[derive(Clone, Debug)]
pub struct ConvNormAct {
    conv: Conv2d,
    norm: BatchNorm2d,
    activation: bool,
}

impl ConvNormAct {
    /// Creates a convolution followed by `BatchNorm` and an optional `ReLU`.
    ///
    /// # Errors
    ///
    /// Returns an error when convolution parameters are invalid.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        activation: bool,
        device: Device,
    ) -> Result<Self> {
        Ok(Self {
            conv: Conv2d::new(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                groups,
                device,
            )?,
            norm: BatchNorm2d::new(out_channels, device)?,
            activation,
        })
    }
}

impl Module for ConvNormAct {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let output = self.norm.forward(&self.conv.forward(input)?)?;
        Ok(if self.activation {
            output.relu()
        } else {
            output
        })
    }
}

impl Trainable for ConvNormAct {
    fn visit_parameters(&self, visitor: &mut dyn FnMut(&Parameter)) {
        self.conv.visit_parameters(visitor);
        self.norm.visit_parameters(visitor);
    }

    fn visit_parameters_mut(&mut self, visitor: &mut dyn FnMut(&mut Parameter)) {
        self.conv.visit_parameters_mut(visitor);
        self.norm.visit_parameters_mut(visitor);
    }

    fn visit_buffers(&self, visitor: &mut dyn FnMut(&[usize], &[f32])) -> Result<()> {
        self.norm.visit_buffers(visitor)
    }
}

impl ModuleMode for ConvNormAct {
    fn set_training(&mut self, training: bool) {
        self.norm.set_training(training);
    }
}

#[derive(Clone, Debug)]
pub struct UniversalInvertedBottleneck {
    start_depthwise: Option<ConvNormAct>,
    expand: ConvNormAct,
    middle_depthwise: Option<ConvNormAct>,
    project: ConvNormAct,
    residual: bool,
}

impl UniversalInvertedBottleneck {
    /// Creates the UIB family used by `MobileNetV4`.
    ///
    /// `(start_dw, middle_dw)` selects `ExtraDW`, `ConvNext`, IB, or FFN. The
    /// paper's `middle_dw_downsample=true` is used, so stride is placed on the
    /// middle depthwise convolution.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid dimensions or when stride has no spatial op.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        expanded_channels: usize,
        start_dw: Option<usize>,
        middle_dw: Option<usize>,
        stride: usize,
        device: Device,
    ) -> Result<Self> {
        if stride > 1 && middle_dw.is_none() {
            return Err(Error::InvalidShape(
                "a strided UIB requires a middle depthwise convolution".into(),
            ));
        }
        let start_depthwise = start_dw
            .map(|kernel| {
                ConvNormAct::new(
                    in_channels,
                    in_channels,
                    kernel,
                    1,
                    in_channels,
                    true,
                    device,
                )
            })
            .transpose()?;
        let expand = ConvNormAct::new(in_channels, expanded_channels, 1, 1, 1, true, device)?;
        let middle_depthwise = middle_dw
            .map(|kernel| {
                ConvNormAct::new(
                    expanded_channels,
                    expanded_channels,
                    kernel,
                    stride,
                    expanded_channels,
                    true,
                    device,
                )
            })
            .transpose()?;
        let project = ConvNormAct::new(expanded_channels, out_channels, 1, 1, 1, false, device)?;
        Ok(Self {
            start_depthwise,
            expand,
            middle_depthwise,
            project,
            residual: stride == 1 && in_channels == out_channels,
        })
    }
}

impl Module for UniversalInvertedBottleneck {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let residual = input;
        let mut output = input.clone();
        if let Some(layer) = &self.start_depthwise {
            output = layer.forward(&output)?;
        }
        output = self.expand.forward(&output)?;
        if let Some(layer) = &self.middle_depthwise {
            output = layer.forward(&output)?;
        }
        output = self.project.forward(&output)?;
        if self.residual {
            output.add(residual)
        } else {
            Ok(output)
        }
    }
}

impl Trainable for UniversalInvertedBottleneck {
    fn visit_parameters(&self, visitor: &mut dyn FnMut(&Parameter)) {
        if let Some(layer) = &self.start_depthwise {
            layer.visit_parameters(visitor);
        }
        self.expand.visit_parameters(visitor);
        if let Some(layer) = &self.middle_depthwise {
            layer.visit_parameters(visitor);
        }
        self.project.visit_parameters(visitor);
    }

    fn visit_parameters_mut(&mut self, visitor: &mut dyn FnMut(&mut Parameter)) {
        if let Some(layer) = &mut self.start_depthwise {
            layer.visit_parameters_mut(visitor);
        }
        self.expand.visit_parameters_mut(visitor);
        if let Some(layer) = &mut self.middle_depthwise {
            layer.visit_parameters_mut(visitor);
        }
        self.project.visit_parameters_mut(visitor);
    }

    fn visit_buffers(&self, visitor: &mut dyn FnMut(&[usize], &[f32])) -> Result<()> {
        if let Some(layer) = &self.start_depthwise {
            layer.visit_buffers(visitor)?;
        }
        self.expand.visit_buffers(visitor)?;
        if let Some(layer) = &self.middle_depthwise {
            layer.visit_buffers(visitor)?;
        }
        self.project.visit_buffers(visitor)
    }
}

impl ModuleMode for UniversalInvertedBottleneck {
    fn set_training(&mut self, training: bool) {
        if let Some(layer) = &mut self.start_depthwise {
            layer.set_training(training);
        }
        self.expand.set_training(training);
        if let Some(layer) = &mut self.middle_depthwise {
            layer.set_training(training);
        }
        self.project.set_training(training);
    }
}

#[derive(Clone, Debug)]
pub struct FusedInvertedBottleneck {
    expand: ConvNormAct,
    project: ConvNormAct,
    residual: bool,
}

impl FusedInvertedBottleneck {
    /// Creates the fixed `FusedIB` stem block from the supplement tables.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid convolution dimensions.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        expanded_channels: usize,
        kernel_size: usize,
        stride: usize,
        device: Device,
    ) -> Result<Self> {
        Ok(Self {
            expand: ConvNormAct::new(
                in_channels,
                expanded_channels,
                kernel_size,
                stride,
                1,
                true,
                device,
            )?,
            project: ConvNormAct::new(expanded_channels, out_channels, 1, 1, 1, false, device)?,
            residual: stride == 1 && in_channels == out_channels,
        })
    }
}

impl Module for FusedInvertedBottleneck {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let output = self.project.forward(&self.expand.forward(input)?)?;
        if self.residual {
            output.add(input)
        } else {
            Ok(output)
        }
    }
}

impl Trainable for FusedInvertedBottleneck {
    fn visit_parameters(&self, visitor: &mut dyn FnMut(&Parameter)) {
        self.expand.visit_parameters(visitor);
        self.project.visit_parameters(visitor);
    }

    fn visit_parameters_mut(&mut self, visitor: &mut dyn FnMut(&mut Parameter)) {
        self.expand.visit_parameters_mut(visitor);
        self.project.visit_parameters_mut(visitor);
    }

    fn visit_buffers(&self, visitor: &mut dyn FnMut(&[usize], &[f32])) -> Result<()> {
        self.expand.visit_buffers(visitor)?;
        self.project.visit_buffers(visitor)
    }
}

impl ModuleMode for FusedInvertedBottleneck {
    fn set_training(&mut self, training: bool) {
        self.expand.set_training(training);
        self.project.set_training(training);
    }
}

#[allow(clippy::cast_precision_loss)]
fn initialized_weights(shape: &[usize], input_channels_per_group: usize) -> Vec<f32> {
    let kernel_area = shape[2] * shape[3];
    let fan_in = (input_channels_per_group * kernel_area) as f32;
    let scale = (2.0 / fan_in).sqrt();
    let mut state = (shape.iter().fold(0x9E37_79B9_u64, |seed, value| {
        seed ^ (*value as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
    })) | 1;
    (0..shape.iter().product())
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = ((state >> 40) as u32) as f32 / ((1_u32 << 24) as f32);
            (unit * 2.0 - 1.0) * scale
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_norm_updates_running_statistics_and_switches_to_eval() {
        let mut layer = BatchNorm2d::new(2, Device::Cpu).unwrap();
        let input = Tensor::from_vec(
            vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0],
            vec![2, 2, 1, 2],
        )
        .unwrap();
        let output = layer.forward(&input).unwrap().to_vec().unwrap();
        for channel in 0..2 {
            let start = channel * 2;
            let values = [
                output[start],
                output[start + 1],
                output[4 + start],
                output[5 + start],
            ];
            let mean = values.into_iter().sum::<f32>() / 4.0;
            assert!(mean.abs() < 1e-5);
        }
        let running_mean = layer.running_mean().unwrap();
        assert!((running_mean[0] - 0.4).abs() < 1e-5);
        assert!((running_mean[1] - 0.5).abs() < 1e-5);
        assert!(layer.running_variance().unwrap()[0] > 1.0);

        layer.eval();
        assert!(!layer.is_training());
        let eval_output = layer.forward(&input).unwrap().to_vec().unwrap();
        assert_ne!(output, eval_output);
    }
}
