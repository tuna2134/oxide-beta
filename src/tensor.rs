use crate::{Error, Result};
use std::collections::HashMap;
use std::ops::{Add, Mul};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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
}

#[derive(Debug)]
pub(crate) enum Op {
    Data(Arc<[f32]>),
    Placeholder(usize),
    Add(Tensor, Tensor),
    Mul(Tensor, Tensor),
    Relu(Tensor),
    MatMul(Tensor, Tensor),
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
    };
    let result = Tensor::new(tensor.node.shape.clone(), device, op);
    cache.insert(tensor.node.id, result.clone());
    result
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
}
