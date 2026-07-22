//! `MobileNetV4` models described by tables 11-15 of `mobilenet.pdf`.

use oxide_torch::nn::{
    Conv2d, ConvNormAct, FusedInvertedBottleneck, Module, ModuleMode, Parameter, Trainable,
    UniversalInvertedBottleneck,
};
use oxide_torch::{Device, Error, Result, Tensor};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

#[derive(Clone, Debug)]
enum Block {
    Conv(Box<ConvNormAct>),
    Classifier(Conv2d),
    Fused(Box<FusedInvertedBottleneck>),
    Uib(Box<UniversalInvertedBottleneck>),
    GlobalAverage,
}

impl Block {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::Conv(layer) => layer.forward(input),
            Self::Classifier(layer) => layer.forward(input),
            Self::Fused(layer) => layer.forward(input),
            Self::Uib(layer) => layer.forward(input),
            Self::GlobalAverage => input.global_avg_pool2d(),
        }
    }
}

impl Trainable for Block {
    fn visit_parameters(&self, visitor: &mut dyn FnMut(&Parameter)) {
        match self {
            Self::Conv(layer) => layer.visit_parameters(visitor),
            Self::Classifier(layer) => layer.visit_parameters(visitor),
            Self::Fused(layer) => layer.visit_parameters(visitor),
            Self::Uib(layer) => layer.visit_parameters(visitor),
            Self::GlobalAverage => {}
        }
    }

    fn visit_parameters_mut(&mut self, visitor: &mut dyn FnMut(&mut Parameter)) {
        match self {
            Self::Conv(layer) => layer.visit_parameters_mut(visitor),
            Self::Classifier(layer) => layer.visit_parameters_mut(visitor),
            Self::Fused(layer) => layer.visit_parameters_mut(visitor),
            Self::Uib(layer) => layer.visit_parameters_mut(visitor),
            Self::GlobalAverage => {}
        }
    }

    fn visit_buffers(&self, visitor: &mut dyn FnMut(&[usize], &[f32])) -> Result<()> {
        match self {
            Self::Conv(layer) => layer.visit_buffers(visitor),
            Self::Classifier(_) | Self::GlobalAverage => Ok(()),
            Self::Fused(layer) => layer.visit_buffers(visitor),
            Self::Uib(layer) => layer.visit_buffers(visitor),
        }
    }

    fn visit_buffers_mut(&mut self, visitor: &mut dyn FnMut(&[usize], &mut [f32])) -> Result<()> {
        match self {
            Self::Conv(layer) => layer.visit_buffers_mut(visitor),
            Self::Classifier(_) | Self::GlobalAverage => Ok(()),
            Self::Fused(layer) => layer.visit_buffers_mut(visitor),
            Self::Uib(layer) => layer.visit_buffers_mut(visitor),
        }
    }
}

impl ModuleMode for Block {
    fn set_training(&mut self, training: bool) {
        match self {
            Self::Conv(layer) => layer.set_training(training),
            Self::Classifier(_) | Self::GlobalAverage => {}
            Self::Fused(layer) => layer.set_training(training),
            Self::Uib(layer) => layer.set_training(training),
        }
    }
}

/// The convolution-only small model from supplementary table 11.
#[derive(Clone, Debug)]
pub struct MobileNetV4ConvSmall {
    blocks: Vec<Block>,
    num_classes: usize,
    device: Device,
    input_channels: usize,
    input_resolution: usize,
}

impl MobileNetV4ConvSmall {
    pub const INPUT_RESOLUTION: usize = 224;

    /// Builds the table-11 network with trainable convolution and `BatchNorm` parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if an internal paper specification is inconsistent.
    pub fn new(num_classes: usize, device: Device) -> Result<Self> {
        Self::build(num_classes, 3, Self::INPUT_RESOLUTION, device)
    }

    /// Builds an MNIST variant accepting `[N, 1, 28, 28]` and producing 10 logits.
    ///
    /// # Errors
    ///
    /// Returns an error if an internal model specification is inconsistent.
    pub fn mnist(device: Device) -> Result<Self> {
        Self::build(10, 1, 28, device)
    }

    fn build(
        num_classes: usize,
        input_channels: usize,
        input_resolution: usize,
        device: Device,
    ) -> Result<Self> {
        if num_classes == 0 {
            return Err(Error::InvalidShape("num_classes must be non-zero".into()));
        }
        let mut blocks = vec![Block::Conv(Box::new(ConvNormAct::new(
            input_channels,
            32,
            3,
            2,
            1,
            true,
            device,
        )?))];
        blocks.push(Block::Fused(Box::new(FusedInvertedBottleneck::new(
            32, 32, 32, 3, 2, device,
        )?)));
        blocks.push(Block::Fused(Box::new(FusedInvertedBottleneck::new(
            32, 64, 96, 3, 2, device,
        )?)));

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
            blocks.push(Block::Uib(Box::new(UniversalInvertedBottleneck::new(
                input, output, expanded, start_dw, middle_dw, stride, device,
            )?)));
        }
        blocks.extend([
            Block::Conv(Box::new(ConvNormAct::new(128, 960, 1, 1, 1, true, device)?)),
            Block::GlobalAverage,
            Block::Conv(Box::new(ConvNormAct::new(
                960, 1280, 1, 1, 1, true, device,
            )?)),
            Block::Classifier(Conv2d::new(1280, num_classes, 1, 1, 1, device)?),
        ]);
        Ok(Self {
            blocks,
            num_classes,
            device,
            input_channels,
            input_resolution,
        })
    }

    /// Returns every logical row output shape, including the classifier.
    ///
    /// # Errors
    ///
    /// Returns an error when input dimensions do not match this variant or a layer fails.
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

    /// Enables batch-statistics behavior for every `BatchNorm` layer.
    pub fn train(&mut self) {
        self.set_training(true);
    }

    /// Enables running-statistics behavior for every `BatchNorm` layer.
    pub fn eval(&mut self) {
        self.set_training(false);
    }

    /// Saves parameters and persistent buffers in the little-endian checkpoint format.
    ///
    /// # Errors
    ///
    /// Returns an error when parameters cannot be materialized or the file cannot be written.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut parameters = Vec::new();
        let mut materialize_error = None;
        self.visit_parameters(&mut |parameter| {
            if materialize_error.is_none() {
                match parameter.value().to_vec() {
                    Ok(data) => parameters.push((parameter.value().shape().to_vec(), data)),
                    Err(error) => materialize_error = Some(error),
                }
            }
        });
        if let Some(error) = materialize_error {
            return Err(error);
        }
        self.visit_buffers(&mut |shape, data| {
            parameters.push((shape.to_vec(), data.to_vec()));
        })?;
        let file = File::create(path.as_ref()).map_err(|error| {
            Error::Execution(format!(
                "failed to create checkpoint {}: {error}",
                path.as_ref().display()
            ))
        })?;
        let mut writer = BufWriter::new(file);
        writer
            .write_all(b"OXTR\x01")
            .and_then(|()| writer.write_all(&(parameters.len() as u64).to_le_bytes()))
            .map_err(checkpoint_error)?;
        for (shape, data) in parameters {
            writer
                .write_all(&(shape.len() as u64).to_le_bytes())
                .map_err(checkpoint_error)?;
            for dimension in shape {
                writer
                    .write_all(&(dimension as u64).to_le_bytes())
                    .map_err(checkpoint_error)?;
            }
            writer
                .write_all(&(data.len() as u64).to_le_bytes())
                .map_err(checkpoint_error)?;
            for value in data {
                writer
                    .write_all(&value.to_le_bytes())
                    .map_err(checkpoint_error)?;
            }
        }
        writer.flush().map_err(checkpoint_error)
    }

    /// Loads an MNIST model checkpoint and switches it to inference mode.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed, incompatible, or unreadable checkpoints.
    pub fn load_mnist(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        let tensors = read_checkpoint(path.as_ref())?;
        let mut model = Self::mnist(device)?;
        let mut next = 0;
        let mut load_error = None;
        model.visit_parameters_mut(&mut |parameter| {
            if load_error.is_some() {
                return;
            }
            let Some((shape, data)) = tensors.get(next) else {
                load_error = Some(Error::Execution("checkpoint has too few tensors".into()));
                return;
            };
            if shape != parameter.value().shape() {
                load_error = Some(Error::InvalidShape(format!(
                    "checkpoint tensor {next} has shape {shape:?}, expected {:?}",
                    parameter.value().shape()
                )));
                return;
            }
            load_error = parameter.replace_data(data.clone()).err();
            next += 1;
        });
        if let Some(error) = load_error {
            return Err(error);
        }
        model.visit_buffers_mut(&mut |shape, buffer| {
            if load_error.is_some() {
                return;
            }
            let Some((saved_shape, data)) = tensors.get(next) else {
                load_error = Some(Error::Execution("checkpoint has too few buffers".into()));
                return;
            };
            if saved_shape != shape || data.len() != buffer.len() {
                load_error = Some(Error::InvalidShape(format!(
                    "checkpoint buffer {next} shape mismatch"
                )));
                return;
            }
            buffer.copy_from_slice(data);
            next += 1;
        })?;
        if let Some(error) = load_error {
            return Err(error);
        }
        if next != tensors.len() {
            return Err(Error::Execution("checkpoint has extra tensors".into()));
        }
        model.eval();
        Ok(model)
    }

    fn validate_input(&self, input: &Tensor) -> Result<()> {
        if input.device() != self.device {
            return Err(Error::DeviceMismatch);
        }
        if input.shape().len() != 4
            || input.shape()[1] != self.input_channels
            || input.shape()[2] != self.input_resolution
            || input.shape()[3] != self.input_resolution
        {
            return Err(Error::InvalidShape(format!(
                "MNv4-Conv-S expects [N, {}, {}, {}], got {:?}",
                self.input_channels,
                self.input_resolution,
                self.input_resolution,
                input.shape()
            )));
        }
        Ok(())
    }
}

#[allow(clippy::needless_pass_by_value)]
fn checkpoint_error(error: std::io::Error) -> Error {
    Error::Execution(format!("checkpoint write failed: {error}"))
}

impl Module for MobileNetV4ConvSmall {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_with_shapes(input).map(|(output, _)| output)
    }
}

impl Trainable for MobileNetV4ConvSmall {
    fn visit_parameters(&self, visitor: &mut dyn FnMut(&Parameter)) {
        for block in &self.blocks {
            block.visit_parameters(visitor);
        }
    }

    fn visit_parameters_mut(&mut self, visitor: &mut dyn FnMut(&mut Parameter)) {
        for block in &mut self.blocks {
            block.visit_parameters_mut(visitor);
        }
    }

    fn visit_buffers(&self, visitor: &mut dyn FnMut(&[usize], &[f32])) -> Result<()> {
        for block in &self.blocks {
            block.visit_buffers(visitor)?;
        }
        Ok(())
    }

    fn visit_buffers_mut(&mut self, visitor: &mut dyn FnMut(&[usize], &mut [f32])) -> Result<()> {
        for block in &mut self.blocks {
            block.visit_buffers_mut(visitor)?;
        }
        Ok(())
    }
}

fn read_checkpoint(path: &Path) -> Result<Vec<(Vec<usize>, Vec<f32>)>> {
    let file = File::open(path).map_err(|error| {
        Error::Execution(format!(
            "failed to open checkpoint {}: {error}",
            path.display()
        ))
    })?;
    let mut reader = BufReader::new(file);
    let mut magic = [0; 5];
    reader.read_exact(&mut magic).map_err(checkpoint_error)?;
    if &magic != b"OXTR\x01" {
        return Err(Error::Execution("invalid OXTR checkpoint header".into()));
    }
    let count = read_usize(&mut reader)?;
    let mut tensors = Vec::with_capacity(count);
    for _ in 0..count {
        let rank = read_usize(&mut reader)?;
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            shape.push(read_usize(&mut reader)?);
        }
        let len = read_usize(&mut reader)?;
        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            let mut bytes = [0; 4];
            reader.read_exact(&mut bytes).map_err(checkpoint_error)?;
            data.push(f32::from_le_bytes(bytes));
        }
        tensors.push((shape, data));
    }
    Ok(tensors)
}

fn read_u64(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes).map_err(checkpoint_error)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_usize(reader: &mut impl Read) -> Result<usize> {
    usize::try_from(read_u64(reader)?)
        .map_err(|_| Error::Execution("checkpoint integer exceeds usize".into()))
}

impl ModuleMode for MobileNetV4ConvSmall {
    fn set_training(&mut self, training: bool) {
        for block in &mut self.blocks {
            block.set_training(training);
        }
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

    #[test]
    fn mnist_variant_accepts_grayscale_images() {
        let model = MobileNetV4ConvSmall::mnist(Device::Cpu).unwrap();
        let input = Tensor::zeros(vec![2, 1, 28, 28]).unwrap();
        assert_eq!(model.forward(&input).unwrap().shape(), &[2, 10]);
    }

    #[test]
    #[ignore = "full naive-CPU MobileNet backward is intentionally expensive"]
    fn mnist_variant_completes_a_full_training_step() {
        use oxide_torch::loss::cross_entropy;
        use oxide_torch::optim::{AdamW, Optimizer};

        let mut model = MobileNetV4ConvSmall::mnist(Device::Cpu).unwrap();
        let input = Tensor::zeros(vec![2, 1, 28, 28]).unwrap();
        let target = Tensor::from_vec(vec![3.0, 7.0], vec![2]).unwrap();
        let mut optimizer = AdamW::new(1e-3, 1e-4).unwrap();
        optimizer.zero_grad(&model).unwrap();
        let loss = cross_entropy(&model.forward(&input).unwrap(), &target).unwrap();
        assert!(loss.item().unwrap().is_finite());
        loss.backward().unwrap();
        optimizer.step(&mut model).unwrap();
    }
}
