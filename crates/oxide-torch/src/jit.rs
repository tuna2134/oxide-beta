//! Trace-based JIT execution plans.
//!
//! Tracing replaces example inputs with placeholders and records the resulting
//! lazy graph. The first `run` for an input signature compiles and caches that
//! specialization. With CUDA enabled, execution dispatches the cuda-oxide PTX
//! kernels, which the CUDA driver JIT-links for the active GPU.

use crate::nn::Module;
use crate::tensor::{Op, eval_cpu};
use crate::{CustomInput, Device, Error, Result, Tensor};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub struct JitModule {
    graph: Tensor,
    input_shapes: Vec<Vec<usize>>,
    device: Device,
    specializations: Mutex<HashMap<Signature, Arc<CompiledPlan>>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Signature(Vec<Vec<usize>>, Device);

#[derive(Clone, Debug)]
pub(crate) struct GraphPlan {
    pub(crate) buffers: Vec<BufferPlan>,
    pub(crate) operations: Vec<PlanOperation>,
    pub(crate) inputs: Vec<usize>,
    pub(crate) output: usize,
}

#[derive(Clone, Debug)]
pub(crate) enum BufferPlan {
    Input { elements: usize },
    Constant(Arc<[f32]>),
    Workspace { elements: usize },
}

#[derive(Clone, Debug)]
pub(crate) enum PlanOperation {
    Add {
        left: usize,
        right: usize,
        output: usize,
    },
    Mul {
        left: usize,
        right: usize,
        output: usize,
    },
    Relu {
        input: usize,
        output: usize,
    },
    MatMul {
        left: usize,
        right: usize,
        output: usize,
        rows: usize,
        columns: usize,
        inner: usize,
    },
    Transformer {
        inputs: Vec<usize>,
        input_shapes: Vec<Vec<usize>>,
        primitive: crate::transformer::Primitive,
        output: usize,
    },
}

enum CompiledPlan {
    Cpu(CpuCompiledPlan),
    #[cfg(feature = "cuda")]
    Cuda(crate::cuda::CudaJitPlan),
    Eager,
}

struct CpuCompiledPlan {
    graph: GraphPlan,
    workspace: Mutex<Vec<Vec<f32>>>,
}

/// Traces `function` with placeholders shaped like `examples`.
///
/// # Errors
///
/// Returns an error for empty or mixed-device inputs, or when `function`
/// cannot construct a valid graph.
pub fn trace(
    examples: &[Tensor],
    function: impl FnOnce(&[Tensor]) -> Result<Tensor>,
) -> Result<JitModule> {
    let Some(first) = examples.first() else {
        return Err(Error::Trace(
            "at least one example input is required".into(),
        ));
    };
    let device = first.device();
    if examples.iter().any(|input| input.device() != device) {
        return Err(Error::DeviceMismatch);
    }
    let placeholders: Vec<_> = examples
        .iter()
        .enumerate()
        .map(|(slot, input)| Tensor::placeholder(slot, input.shape().to_vec(), device))
        .collect();
    let graph = function(&placeholders)?;
    if graph.device() != device {
        return Err(Error::DeviceMismatch);
    }
    Ok(JitModule {
        graph,
        input_shapes: examples
            .iter()
            .map(|input| input.shape().to_vec())
            .collect(),
        device,
        specializations: Mutex::new(HashMap::new()),
    })
}

/// Traces and compiles a user-defined [`Module`] for one example signature.
///
/// The model is only needed while its lazy graph is traced; parameters become
/// constants in the resulting execution plan.
///
/// # Errors
///
/// Returns an error when tracing fails or the model output is on a different
/// device than `example`.
pub fn compile<M: Module<Output = Tensor>>(model: &M, example: &Tensor) -> Result<JitModule> {
    trace(std::slice::from_ref(example), |inputs| {
        model.forward(&inputs[0])
    })
}

impl JitModule {
    /// Executes the traced graph and returns a materialized tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when input count, shape, or device differs from the
    /// trace, or when the selected backend cannot execute the graph.
    pub fn run(&self, inputs: &[Tensor]) -> Result<Tensor> {
        self.validate_inputs(inputs)?;
        let signature = Signature(
            inputs.iter().map(|input| input.shape().to_vec()).collect(),
            self.device,
        );
        let plan = {
            let mut cache = self
                .specializations
                .lock()
                .map_err(|_| Error::Execution("JIT cache lock was poisoned".into()))?;
            if let Some(plan) = cache.get(&signature) {
                Arc::clone(plan)
            } else {
                let plan = Arc::new(self.compile_plan()?);
                cache.insert(signature, Arc::clone(&plan));
                plan
            }
        };

        let output = match plan.as_ref() {
            CompiledPlan::Cpu(plan) => plan.run(inputs)?,
            #[cfg(feature = "cuda")]
            CompiledPlan::Cuda(plan) => plan.run(inputs)?,
            CompiledPlan::Eager => self.run_eager(inputs)?,
        };
        Ok(Tensor::from_vec(output, self.graph.shape().to_vec())?.to(self.device))
    }

    #[must_use]
    pub fn cached_specializations(&self) -> usize {
        self.specializations.lock().map_or(0, |cache| cache.len())
    }

    #[must_use]
    pub fn output_shape(&self) -> &[usize] {
        self.graph.shape()
    }

    fn validate_inputs(&self, inputs: &[Tensor]) -> Result<()> {
        if inputs.len() != self.input_shapes.len() {
            return Err(Error::Trace(format!(
                "expected {} inputs, got {}",
                self.input_shapes.len(),
                inputs.len()
            )));
        }
        for (index, (input, expected)) in inputs.iter().zip(&self.input_shapes).enumerate() {
            if input.shape() != expected {
                return Err(Error::Trace(format!(
                    "input {index} expected shape {expected:?}, got {:?}",
                    input.shape()
                )));
            }
            if input.device() != self.device {
                return Err(Error::DeviceMismatch);
            }
        }
        Ok(())
    }

    fn compile_plan(&self) -> Result<CompiledPlan> {
        let Some(graph) = GraphPlan::compile(&self.graph, self.input_shapes.len())? else {
            return Ok(CompiledPlan::Eager);
        };
        match self.device {
            Device::Cpu => Ok(CompiledPlan::Cpu(CpuCompiledPlan::new(graph))),
            Device::Cuda(device) => {
                #[cfg(feature = "cuda")]
                {
                    Ok(CompiledPlan::Cuda(crate::cuda::CudaJitPlan::compile(
                        graph, device,
                    )?))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = (graph, device);
                    Err(Error::CudaUnavailable)
                }
            }
        }
    }

    fn run_eager(&self, inputs: &[Tensor]) -> Result<Vec<f32>> {
        let bound = bind_placeholders(&self.graph, inputs)?;
        match self.device {
            Device::Cpu => eval_cpu(&bound, &mut HashMap::new(), None),
            Device::Cuda(device) => {
                #[cfg(feature = "cuda")]
                {
                    crate::cuda::eval(&bound, device)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = (bound, device);
                    Err(Error::CudaUnavailable)
                }
            }
        }
    }
}

impl GraphPlan {
    fn compile(graph: &Tensor, input_count: usize) -> Result<Option<Self>> {
        let mut builder = PlanBuilder {
            buffers: Vec::new(),
            operations: Vec::new(),
            inputs: vec![None; input_count],
            nodes: HashMap::new(),
            supported: true,
        };
        let output = builder.lower(graph)?;
        if !builder.supported {
            return Ok(None);
        }
        let inputs = builder
            .inputs
            .into_iter()
            .enumerate()
            .map(|(slot, buffer)| {
                buffer.ok_or_else(|| Error::Trace(format!("traced input {slot} is unused")))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(Self {
            buffers: builder.buffers,
            operations: builder.operations,
            inputs,
            output,
        }))
    }
}

struct PlanBuilder {
    buffers: Vec<BufferPlan>,
    operations: Vec<PlanOperation>,
    inputs: Vec<Option<usize>>,
    nodes: HashMap<u64, usize>,
    supported: bool,
}

impl PlanBuilder {
    #[allow(clippy::too_many_lines)]
    fn lower(&mut self, tensor: &Tensor) -> Result<usize> {
        if let Some(slot) = self.nodes.get(&tensor.node.id) {
            return Ok(*slot);
        }
        let slot = match &tensor.node.op {
            Op::Data(data) => self.push_buffer(BufferPlan::Constant(Arc::clone(data))),
            Op::Placeholder(input) => {
                let elements = tensor.numel();
                let slot = self.push_buffer(BufferPlan::Input { elements });
                let target = self
                    .inputs
                    .get_mut(*input)
                    .ok_or_else(|| Error::Trace(format!("invalid placeholder {input}")))?;
                *target = Some(slot);
                slot
            }
            Op::Add(left, right) | Op::Mul(left, right) => {
                let left_slot = self.lower(left)?;
                let right_slot = self.lower(right)?;
                let output = self.push_workspace(tensor.numel());
                self.operations
                    .push(if matches!(&tensor.node.op, Op::Add(_, _)) {
                        PlanOperation::Add {
                            left: left_slot,
                            right: right_slot,
                            output,
                        }
                    } else {
                        PlanOperation::Mul {
                            left: left_slot,
                            right: right_slot,
                            output,
                        }
                    });
                output
            }
            Op::Relu(input) => {
                let input = self.lower(input)?;
                let output = self.push_workspace(tensor.numel());
                self.operations.push(PlanOperation::Relu { input, output });
                output
            }
            Op::MatMul(left, right) => {
                let left_slot = self.lower(left)?;
                let right_slot = self.lower(right)?;
                let rows = left.shape()[0];
                let inner = left.shape()[1];
                let columns = right.shape()[1];
                let output = self.push_workspace(tensor.numel());
                self.operations.push(PlanOperation::MatMul {
                    left: left_slot,
                    right: right_slot,
                    output,
                    rows,
                    columns,
                    inner,
                });
                output
            }
            Op::Reshape(input) => self.lower(input)?,
            Op::Linear {
                input,
                weight,
                bias,
            } => self.lower_transformer(
                tensor,
                crate::transformer::Primitive::Linear,
                &[input, weight, bias],
            )?,
            Op::Gelu(input) => {
                self.lower_transformer(tensor, crate::transformer::Primitive::Gelu, &[input])?
            }
            Op::Tanh(input) => {
                self.lower_transformer(tensor, crate::transformer::Primitive::Tanh, &[input])?
            }
            Op::Embedding { ids, weight } => self.lower_transformer(
                tensor,
                crate::transformer::Primitive::Embedding,
                &[ids, weight],
            )?,
            Op::LayerNorm {
                input,
                weight,
                bias,
                epsilon,
            } => self.lower_transformer(
                tensor,
                crate::transformer::Primitive::LayerNorm { epsilon: *epsilon },
                &[input, weight, bias],
            )?,
            Op::SelectFirst(input) => self.lower_transformer(
                tensor,
                crate::transformer::Primitive::SelectFirst,
                &[input],
            )?,
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
            } => self.lower_transformer(
                tensor,
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
            )?,
            _ => {
                self.supported = false;
                self.push_workspace(tensor.numel())
            }
        };
        self.nodes.insert(tensor.node.id, slot);
        Ok(slot)
    }

    fn lower_transformer(
        &mut self,
        tensor: &Tensor,
        primitive: crate::transformer::Primitive,
        inputs: &[&Tensor],
    ) -> Result<usize> {
        let input_slots = inputs
            .iter()
            .map(|input| self.lower(input))
            .collect::<Result<Vec<_>>>()?;
        let output = self.push_workspace(tensor.numel());
        self.operations.push(PlanOperation::Transformer {
            inputs: input_slots,
            input_shapes: inputs.iter().map(|input| input.shape().to_vec()).collect(),
            primitive,
            output,
        });
        Ok(output)
    }

    fn push_workspace(&mut self, elements: usize) -> usize {
        self.push_buffer(BufferPlan::Workspace { elements })
    }

    fn push_buffer(&mut self, buffer: BufferPlan) -> usize {
        let slot = self.buffers.len();
        self.buffers.push(buffer);
        slot
    }
}

impl CpuCompiledPlan {
    fn new(graph: GraphPlan) -> Self {
        let workspace = graph
            .buffers
            .iter()
            .map(|buffer| match buffer {
                BufferPlan::Input { elements } | BufferPlan::Workspace { elements } => {
                    vec![0.0; *elements]
                }
                BufferPlan::Constant(data) => data.to_vec(),
            })
            .collect();
        Self {
            graph,
            workspace: Mutex::new(workspace),
        }
    }

    fn run(&self, inputs: &[Tensor]) -> Result<Vec<f32>> {
        let mut buffers = self
            .workspace
            .lock()
            .map_err(|_| Error::Execution("JIT workspace lock was poisoned".into()))?;
        for (input, slot) in inputs.iter().zip(&self.graph.inputs) {
            let values = input.to_vec()?;
            buffers[*slot].copy_from_slice(&values);
        }
        for operation in &self.graph.operations {
            execute_cpu(operation, &mut buffers)?;
        }
        Ok(buffers[self.graph.output].clone())
    }
}

fn execute_cpu(operation: &PlanOperation, buffers: &mut [Vec<f32>]) -> Result<()> {
    let (left, right, output) = match operation {
        PlanOperation::Add {
            left,
            right,
            output,
        }
        | PlanOperation::Mul {
            left,
            right,
            output,
        }
        | PlanOperation::MatMul {
            left,
            right,
            output,
            ..
        } => (*left, Some(*right), *output),
        PlanOperation::Relu { input, output } => (*input, None, *output),
        PlanOperation::Transformer { output, .. } => (0, None, *output),
    };
    let (sources, destination) = buffers.split_at_mut(output);
    let output_buffer = &mut destination[0];
    match operation {
        PlanOperation::Add { .. } => {
            for (index, value) in output_buffer.iter_mut().enumerate() {
                *value = sources[left][index] + sources[right.expect("right input")][index];
            }
        }
        PlanOperation::Mul { .. } => {
            for (index, value) in output_buffer.iter_mut().enumerate() {
                *value = sources[left][index] * sources[right.expect("right input")][index];
            }
        }
        PlanOperation::Relu { .. } => {
            for (value, input) in output_buffer.iter_mut().zip(&sources[left]) {
                *value = input.max(0.0);
            }
        }
        PlanOperation::MatMul {
            rows,
            columns,
            inner,
            ..
        } => {
            let right = right.expect("right input");
            for row in 0..*rows {
                for column in 0..*columns {
                    let mut sum = 0.0;
                    for index in 0..*inner {
                        sum += sources[left][row * inner + index]
                            * sources[right][index * columns + column];
                    }
                    output_buffer[row * columns + column] = sum;
                }
            }
        }
        PlanOperation::Transformer {
            inputs,
            input_shapes,
            primitive,
            ..
        } => {
            let custom_inputs = inputs
                .iter()
                .zip(input_shapes)
                .map(|(slot, shape)| CustomInput {
                    shape,
                    values: &sources[*slot],
                })
                .collect::<Vec<_>>();
            let values = crate::transformer::forward(*primitive, &custom_inputs)?;
            if values.len() != output_buffer.len() {
                return Err(Error::Execution(
                    "custom JIT operation returned the wrong output size".into(),
                ));
            }
            output_buffer.copy_from_slice(&values);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn bind_placeholders(graph: &Tensor, inputs: &[Tensor]) -> Result<Tensor> {
    #[allow(clippy::too_many_lines)]
    fn bind(
        tensor: &Tensor,
        inputs: &[Tensor],
        cache: &mut HashMap<u64, Tensor>,
    ) -> Result<Tensor> {
        if let Some(value) = cache.get(&tensor.node.id) {
            return Ok(value.clone());
        }
        let result = match &tensor.node.op {
            Op::Data(data) => Tensor::new(
                tensor.shape().to_vec(),
                tensor.device(),
                Op::Data(data.clone()),
            ),
            Op::Placeholder(slot) => inputs
                .get(*slot)
                .cloned()
                .ok_or_else(|| Error::Trace(format!("missing input {slot}")))?,
            Op::Add(a, b) => bind(a, inputs, cache)?.add(&bind(b, inputs, cache)?)?,
            Op::Mul(a, b) => bind(a, inputs, cache)?.mul(&bind(b, inputs, cache)?)?,
            Op::Relu(input) => bind(input, inputs, cache)?.relu(),
            Op::MatMul(a, b) => bind(a, inputs, cache)?.matmul(&bind(b, inputs, cache)?)?,
            Op::Linear {
                input,
                weight,
                bias,
            } => {
                let weight = bind(weight, inputs, cache)?;
                crate::transformer::linear(
                    &bind(input, inputs, cache)?,
                    &weight,
                    &bind(bias, inputs, cache)?,
                    weight.shape()[0],
                )?
            }
            Op::Gelu(input) => crate::transformer::gelu(&bind(input, inputs, cache)?)?,
            Op::Tanh(input) => crate::transformer::tanh(&bind(input, inputs, cache)?)?,
            Op::Embedding { ids, weight } => {
                let weight = bind(weight, inputs, cache)?;
                crate::transformer::embedding(
                    &bind(ids, inputs, cache)?,
                    &weight,
                    weight.shape()[1],
                )?
            }
            Op::LayerNorm {
                input,
                weight,
                bias,
                epsilon,
            } => crate::transformer::layer_norm(
                &bind(input, inputs, cache)?,
                &bind(weight, inputs, cache)?,
                &bind(bias, inputs, cache)?,
                *epsilon,
            )?,
            Op::SelectFirst(input) => {
                let input = bind(input, inputs, cache)?;
                crate::transformer::select_first(&input, input.shape()[2])?
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
                heads,
            } => crate::transformer::scaled_dot_product_attention(
                &bind(input, inputs, cache)?,
                &bind(mask, inputs, cache)?,
                &bind(query_weight, inputs, cache)?,
                &bind(query_bias, inputs, cache)?,
                &bind(key_weight, inputs, cache)?,
                &bind(key_bias, inputs, cache)?,
                &bind(value_weight, inputs, cache)?,
                &bind(value_bias, inputs, cache)?,
                *heads,
            )?,
            Op::Conv2d {
                input,
                weight,
                bias,
                stride,
                padding,
                groups,
            } => bind(input, inputs, cache)?.conv2d(
                &bind(weight, inputs, cache)?,
                &bind(bias, inputs, cache)?,
                *stride,
                *padding,
                *groups,
            )?,
            Op::AvgPool2d {
                input,
                kernel,
                stride,
            } => bind(input, inputs, cache)?.avg_pool2d(*kernel, *stride)?,
            Op::Reshape(input) => bind(input, inputs, cache)?.reshape(tensor.shape().to_vec())?,
            Op::CrossEntropy { logits, targets } => {
                bind(logits, inputs, cache)?.cross_entropy(&bind(targets, inputs, cache)?)?
            }
            Op::BatchNorm2d {
                input,
                weight,
                bias,
                state,
                training,
                momentum,
                epsilon,
                ..
            } => bind(input, inputs, cache)?.batch_norm2d(
                &bind(weight, inputs, cache)?,
                &bind(bias, inputs, cache)?,
                state.clone(),
                *training,
                *momentum,
                *epsilon,
            )?,
            Op::Custom {
                inputs: operands,
                operation,
            } => Tensor::custom(
                operands
                    .iter()
                    .map(|operand| bind(operand, inputs, cache))
                    .collect::<Result<Vec<_>>>()?,
                tensor.shape().to_vec(),
                Arc::clone(operation),
            )?,
        };
        cache.insert(tensor.node.id, result.clone());
        Ok(result)
    }
    bind(graph, inputs, &mut HashMap::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CustomModel {
        weight: Tensor,
        bias: Tensor,
    }

    impl Module for CustomModel {
        type Output = Tensor;

        fn forward(&self, input: &Tensor) -> Result<Tensor> {
            Ok(input.matmul(&self.weight)?.add(&self.bias)?.relu())
        }
    }

    #[test]
    fn trace_caches_and_reuses_a_graph() {
        let x = Tensor::from_vec(vec![-2.0, 3.0], vec![2]).unwrap();
        let bias = Tensor::ones(vec![2]).unwrap();
        let module = trace(&[x.clone(), bias.clone()], |inputs| {
            Ok(inputs[0].add(&inputs[1])?.relu())
        })
        .unwrap();

        assert_eq!(module.cached_specializations(), 0);
        assert_eq!(
            module.run(&[x, bias]).unwrap().to_vec().unwrap(),
            vec![0.0, 4.0]
        );
        assert_eq!(module.cached_specializations(), 1);
    }

    #[test]
    fn compiles_a_user_defined_module_and_reuses_fixed_workspace() {
        let model = CustomModel {
            weight: Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap(),
            bias: Tensor::from_vec(vec![1.0, -1.0], vec![1, 2]).unwrap(),
        };
        let example = Tensor::zeros(vec![1, 2]).unwrap();
        let compiled = compile(&model, &example).unwrap();

        let first = compiled
            .run(&[Tensor::from_vec(vec![1.0, 1.0], vec![1, 2]).unwrap()])
            .unwrap()
            .to_vec()
            .unwrap();
        let second = compiled
            .run(&[Tensor::from_vec(vec![2.0, -1.0], vec![1, 2]).unwrap()])
            .unwrap()
            .to_vec()
            .unwrap();

        assert_eq!(first, vec![5.0, 5.0]);
        assert_eq!(second, vec![0.0, 0.0]);
        assert_eq!(compiled.cached_specializations(), 1);
    }

    #[test]
    fn traced_batch_norm_recomputes_batch_statistics_each_run() {
        use crate::nn::{BatchNorm2d, Module};

        let layer = BatchNorm2d::new(1, Device::Cpu).unwrap();
        let example = Tensor::from_vec(vec![1.0, 3.0], vec![1, 1, 1, 2]).unwrap();
        let module = trace(&[example], |inputs| layer.forward(&inputs[0])).unwrap();
        let first = module
            .run(&[Tensor::from_vec(vec![1.0, 3.0], vec![1, 1, 1, 2]).unwrap()])
            .unwrap()
            .to_vec()
            .unwrap();
        let second = module
            .run(&[Tensor::from_vec(vec![10.0, 20.0], vec![1, 1, 1, 2]).unwrap()])
            .unwrap()
            .to_vec()
            .unwrap();
        assert!((first[0] + 1.0).abs() < 1e-4 && (first[1] - 1.0).abs() < 1e-4);
        assert!((second[0] + 1.0).abs() < 1e-4 && (second[1] - 1.0).abs() < 1e-4);
    }
}
