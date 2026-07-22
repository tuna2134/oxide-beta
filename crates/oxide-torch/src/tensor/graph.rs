use crate::Result;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

pub(crate) static NEXT_ID: AtomicU64 = AtomicU64::new(1);

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

/// Read-only input passed to a user-defined CPU tensor operation.
pub struct CustomInput<'a> {
    pub shape: &'a [usize],
    pub values: &'a [f32],
}

/// Extensible differentiable CPU operation used by higher-level model crates.
pub trait CustomOp: std::fmt::Debug + Send + Sync {
    /// Evaluates the operation on materialized CPU inputs.
    ///
    /// # Errors
    ///
    /// Returns an error when the input shapes or values are invalid.
    fn forward(&self, inputs: &[CustomInput<'_>]) -> Result<Vec<f32>>;

    /// Computes input gradients for a materialized output gradient.
    ///
    /// # Errors
    ///
    /// Returns an error when the gradient shape is invalid.
    fn backward(
        &self,
        inputs: &[CustomInput<'_>],
        output_gradient: &[f32],
    ) -> Result<Vec<Option<Vec<f32>>>>;
}

#[derive(Debug)]
pub(crate) struct Node {
    pub(crate) id: u64,
    pub(crate) shape: Vec<usize>,
    pub(crate) device: Device,
    pub(crate) op: Op,
    pub(super) grad: Mutex<Option<Vec<f32>>>,
}

#[derive(Debug)]
pub(crate) struct BatchNormState {
    pub(crate) running_mean: Vec<f32>,
    pub(crate) running_variance: Vec<f32>,
    #[cfg(feature = "cuda")]
    pub(crate) device: Device,
}

#[derive(Clone, Debug)]
pub(crate) struct BatchNormStatistics {
    pub(crate) mean: Vec<f32>,
    pub(crate) inverse_standard_deviation: Vec<f32>,
}

#[derive(Debug)]
pub(crate) enum Op {
    Data(Arc<[f32]>),
    Placeholder(usize),
    Add(Tensor, Tensor),
    Mul(Tensor, Tensor),
    Relu(Tensor),
    MatMul(Tensor, Tensor),
    Linear {
        input: Tensor,
        weight: Tensor,
        bias: Tensor,
    },
    Gelu(Tensor),
    Tanh(Tensor),
    Embedding {
        ids: Tensor,
        weight: Tensor,
    },
    LayerNorm {
        input: Tensor,
        weight: Tensor,
        bias: Tensor,
        epsilon: f32,
    },
    SelectFirst(Tensor),
    ScaledDotProductAttention {
        input: Tensor,
        mask: Tensor,
        query_weight: Tensor,
        query_bias: Tensor,
        key_weight: Tensor,
        key_bias: Tensor,
        value_weight: Tensor,
        value_bias: Tensor,
        heads: usize,
    },
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
    Custom {
        inputs: Vec<Tensor>,
        operation: Arc<dyn CustomOp>,
    },
}
