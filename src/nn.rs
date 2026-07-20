//! Minimal neural-network layers used by the paper models.

use crate::{Device, Error, Result, Tensor};

pub trait Module {
    /// Runs a lazy forward pass.
    ///
    /// # Errors
    ///
    /// Returns an error when an input shape/device is incompatible with the module.
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
}

#[derive(Clone, Debug)]
pub struct Conv2d {
    weight: Tensor,
    bias: Tensor,
    stride: usize,
    padding: usize,
    groups: usize,
}

impl Conv2d {
    /// Creates a convolution with zero-initialized, inference-ready parameters.
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
        let weight = Tensor::zeros(vec![
            out_channels,
            in_channels / groups,
            kernel_size,
            kernel_size,
        ])?
        .to(device);
        let bias = Tensor::zeros(vec![out_channels])?.to(device);
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
            weight,
            bias,
            stride,
            padding,
            groups,
        })
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> &Tensor {
        &self.bias
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.conv2d(
            &self.weight,
            &self.bias,
            self.stride,
            self.padding,
            self.groups,
        )
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
