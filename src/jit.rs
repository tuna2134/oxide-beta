//! Trace-based JIT execution plans.
//!
//! Tracing replaces example inputs with placeholders and records the resulting
//! lazy graph. The first `run` for an input signature compiles and caches that
//! specialization. With CUDA enabled, execution dispatches the cuda-oxide PTX
//! kernels, which the CUDA driver JIT-links for the active GPU.

use crate::tensor::{Op, eval_cpu};
use crate::{Device, Error, Result, Tensor};
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

#[derive(Debug)]
pub struct JitModule {
    graph: Tensor,
    input_shapes: Vec<Vec<usize>>,
    device: Device,
    specializations: Mutex<HashSet<Signature>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Signature(Vec<Vec<usize>>, Device);

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
        specializations: Mutex::new(HashSet::new()),
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
        self.specializations
            .lock()
            .map_err(|_| Error::Execution("JIT cache lock was poisoned".into()))?
            .insert(signature);

        let output = match self.device {
            Device::Cpu => {
                let values: Vec<Vec<f32>> =
                    inputs.iter().map(Tensor::to_vec).collect::<Result<_>>()?;
                eval_cpu(&self.graph, &mut HashMap::new(), Some(&values))?
            }
            Device::Cuda(device) => {
                let bound = bind_placeholders(&self.graph, inputs)?;
                #[cfg(feature = "cuda")]
                {
                    crate::cuda::eval(&bound, device)?
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = (bound, device);
                    return Err(Error::CudaUnavailable);
                }
            }
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
}

fn bind_placeholders(graph: &Tensor, inputs: &[Tensor]) -> Result<Tensor> {
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
        };
        cache.insert(tensor.node.id, result.clone());
        Ok(result)
    }
    bind(graph, inputs, &mut HashMap::new())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
