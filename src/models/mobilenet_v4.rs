//! `MobileNetV4` models described by tables 11-15 of `mobilenet.pdf`.

use crate::nn::{ConvNormAct, FusedInvertedBottleneck, Module, UniversalInvertedBottleneck};
use crate::{Device, Error, Result, Tensor};

#[derive(Clone, Debug)]
enum Block {
    Conv(ConvNormAct),
    Fused(FusedInvertedBottleneck),
    Uib(UniversalInvertedBottleneck),
    GlobalAverage,
}

impl Block {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::Conv(layer) => layer.forward(input),
            Self::Fused(layer) => layer.forward(input),
            Self::Uib(layer) => layer.forward(input),
            Self::GlobalAverage => input.global_avg_pool2d(),
        }
    }
}

/// The convolution-only small model from supplementary table 11.
#[derive(Clone, Debug)]
pub struct MobileNetV4ConvSmall {
    blocks: Vec<Block>,
    num_classes: usize,
    device: Device,
}

impl MobileNetV4ConvSmall {
    pub const INPUT_RESOLUTION: usize = 224;

    /// Builds the table-11 network with zero-initialized/folded parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if an internal paper specification is inconsistent.
    pub fn new(num_classes: usize, device: Device) -> Result<Self> {
        if num_classes == 0 {
            return Err(Error::InvalidShape("num_classes must be non-zero".into()));
        }
        let mut blocks = vec![Block::Conv(ConvNormAct::new(3, 32, 3, 2, 1, true, device)?)];
        blocks.push(Block::Fused(FusedInvertedBottleneck::new(
            32, 32, 32, 3, 2, device,
        )?));
        blocks.push(Block::Fused(FusedInvertedBottleneck::new(
            32, 64, 96, 3, 2, device,
        )?));

        let specs = [
            // in, out, expanded, start DW, middle DW, stride
            (64, 96, 192, Some(5), Some(5), 2),
            (96, 96, 192, None, Some(3), 1),
            (96, 96, 192, None, Some(3), 1),
            (96, 96, 192, None, Some(3), 1),
            (96, 96, 192, None, Some(3), 1),
            (96, 96, 384, Some(3), None, 1),
            (96, 128, 576, Some(3), Some(3), 2),
            (128, 128, 512, Some(5), Some(5), 1),
            (128, 128, 512, None, Some(5), 1),
            (128, 128, 384, None, Some(5), 1),
            (128, 128, 512, None, Some(3), 1),
            (128, 128, 512, None, Some(3), 1),
        ];
        for (input, output, expanded, start_dw, middle_dw, stride) in specs {
            blocks.push(Block::Uib(UniversalInvertedBottleneck::new(
                input, output, expanded, start_dw, middle_dw, stride, device,
            )?));
        }
        blocks.extend([
            Block::Conv(ConvNormAct::new(128, 960, 1, 1, 1, true, device)?),
            Block::GlobalAverage,
            Block::Conv(ConvNormAct::new(960, 1280, 1, 1, 1, true, device)?),
            Block::Conv(ConvNormAct::new(1280, num_classes, 1, 1, 1, false, device)?),
        ]);
        Ok(Self {
            blocks,
            num_classes,
            device,
        })
    }

    /// Returns every logical row output shape, including the classifier.
    ///
    /// # Errors
    ///
    /// Returns an error when the input is not `[N, 3, 224, 224]` or a layer fails.
    pub fn forward_with_shapes(&self, input: &Tensor) -> Result<(Tensor, Vec<Vec<usize>>)> {
        self.validate_input(input)?;
        let mut output = input.clone();
        let mut shapes = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            output = block.forward(&output)?;
            shapes.push(output.shape().to_vec());
        }
        output = output.reshape(vec![input.shape()[0], self.num_classes])?;
        if let Some(last) = shapes.last_mut() {
            *last = output.shape().to_vec();
        }
        Ok((output, shapes))
    }

    #[must_use]
    pub fn device(&self) -> Device {
        self.device
    }

    fn validate_input(&self, input: &Tensor) -> Result<()> {
        if input.device() != self.device {
            return Err(Error::DeviceMismatch);
        }
        if input.shape().len() != 4
            || input.shape()[1] != 3
            || input.shape()[2] != Self::INPUT_RESOLUTION
            || input.shape()[3] != Self::INPUT_RESOLUTION
        {
            return Err(Error::InvalidShape(format!(
                "MNv4-Conv-S expects [N, 3, 224, 224], got {:?}",
                input.shape()
            )));
        }
        Ok(())
    }
}

impl Module for MobileNetV4ConvSmall {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_with_shapes(input).map(|(output, _)| output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_small_matches_supplement_table_11_shapes() {
        let model = MobileNetV4ConvSmall::new(1000, Device::Cpu).unwrap();
        let input = Tensor::zeros(vec![1, 3, 224, 224]).unwrap();
        let (output, shapes) = model.forward_with_shapes(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1000]);
        assert_eq!(shapes.len(), 19);
        assert_eq!(shapes[0], [1, 32, 112, 112]);
        assert_eq!(shapes[1], [1, 32, 56, 56]);
        assert_eq!(shapes[2], [1, 64, 28, 28]);
        assert_eq!(shapes[3], [1, 96, 14, 14]);
        assert!(shapes[3..9].iter().all(|shape| shape == &[1, 96, 14, 14]));
        assert!(shapes[9..15].iter().all(|shape| shape == &[1, 128, 7, 7]));
        assert_eq!(shapes[15], [1, 960, 7, 7]);
        assert_eq!(shapes[16], [1, 960, 1, 1]);
        assert_eq!(shapes[17], [1, 1280, 1, 1]);
        assert_eq!(shapes[18], [1, 1000]);
    }
}
