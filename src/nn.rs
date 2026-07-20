//! Minimal neural-network layers used by the paper models.

use crate::{Device, Error, Result, Tensor};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_PARAMETER_ID: AtomicU64 = AtomicU64::new(1);

pub trait Module {
    /// Runs a lazy forward pass.
    ///
    /// # Errors
    ///
    /// Returns an error when an input shape/device is incompatible with the module.
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
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
}

/// Provides structured traversal without exposing a model's internal layout.
pub trait Trainable {
    fn visit_parameters(&self, visitor: &mut dyn FnMut(&Parameter));
    fn visit_parameters_mut(&mut self, visitor: &mut dyn FnMut(&mut Parameter));
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
    activation: bool,
}

impl ConvNormAct {
    /// Creates a convolution whose `BatchNorm` is represented as folded weights.
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
            activation,
        })
    }
}

impl Module for ConvNormAct {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let output = self.conv.forward(input)?;
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
    }

    fn visit_parameters_mut(&mut self, visitor: &mut dyn FnMut(&mut Parameter)) {
        self.conv.visit_parameters_mut(visitor);
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
