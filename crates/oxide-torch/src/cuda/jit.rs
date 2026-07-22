//! Fixed-buffer CUDA execution plans for the public trace JIT.

use super::kernels;
use crate::jit::{BufferPlan, GraphPlan, PlanOperation};
use crate::{Error, Result, Tensor};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use oxide_torch_cuda::cuda_graph::CudaGraphExec;
use std::sync::{Arc, Mutex};

pub(crate) struct CudaJitPlan {
    state: Mutex<CudaJitState>,
}

struct CudaJitState {
    _context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
    transformer: kernels::transformer::LoadedModule,
    buffers: Vec<DeviceBuffer<f32>>,
    inputs: Vec<usize>,
    output: usize,
    graph: CudaGraphExec,
}

impl CudaJitPlan {
    pub(crate) fn compile(plan: GraphPlan, device: usize) -> Result<Self> {
        let context = CudaContext::new(device).map_err(cuda_error)?;
        // Legacy default streams cannot be captured. One dedicated stream also
        // gives copies and graph replays a stable ordering.
        let stream = context.new_stream().map_err(cuda_error)?;
        let module = oxide_torch_cuda::load_kernels(&context).map_err(cuda_error)?;
        let transformer =
            kernels::transformer::LoadedModule::from_parent(&module).map_err(cuda_error)?;
        let mut buffers = plan
            .buffers
            .iter()
            .map(|buffer| match buffer {
                BufferPlan::Input { elements } | BufferPlan::Workspace { elements } => {
                    DeviceBuffer::zeroed(&stream, *elements).map_err(cuda_error)
                }
                BufferPlan::Constant(values) => {
                    DeviceBuffer::from_host(&stream, values).map_err(cuda_error)
                }
            })
            .collect::<Result<Vec<_>>>()?;

        let graph = CudaGraphExec::capture(&stream, || {
            launch_operations(
                &module,
                &transformer,
                &stream,
                &plan.operations,
                &mut buffers,
            )
        })?;
        Ok(Self {
            state: Mutex::new(CudaJitState {
                _context: context,
                stream,
                module,
                transformer,
                buffers,
                inputs: plan.inputs,
                output: plan.output,
                graph,
            }),
        })
    }

    pub(crate) fn run(&self, inputs: &[Tensor]) -> Result<Vec<f32>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::Execution("CUDA JIT workspace lock was poisoned".into()))?;
        let CudaJitState {
            stream,
            buffers,
            inputs: input_slots,
            output,
            graph,
            ..
        } = &mut *state;

        // Keep host allocations alive until the graph result copy synchronizes
        // the stream. The device addresses themselves never change.
        let host_inputs = inputs
            .iter()
            .map(Tensor::to_vec)
            .collect::<Result<Vec<_>>>()?;
        for (values, slot) in host_inputs.iter().zip(input_slots.iter()) {
            // SAFETY: `values` stays alive through the synchronizing D2H copy
            // below and the destination buffer has the traced fixed extent.
            unsafe { buffers[*slot].copy_from_host_async_unchecked(stream, values) }
                .map_err(cuda_error)?;
        }
        graph.launch(stream)?;
        buffers[*output].to_host_vec(stream).map_err(cuda_error)
    }
}

fn launch_operations(
    module: &kernels::LoadedModule,
    transformer: &kernels::transformer::LoadedModule,
    stream: &CudaStream,
    operations: &[PlanOperation],
    buffers: &mut [DeviceBuffer<f32>],
) -> Result<()> {
    for operation in operations {
        let (left, right, output, elements) = match operation {
            PlanOperation::Add {
                left,
                right,
                output,
            }
            | PlanOperation::Mul {
                left,
                right,
                output,
            } => (*left, Some(*right), *output, buffers[*output].len()),
            PlanOperation::Relu { input, output } => {
                (*input, None, *output, buffers[*output].len())
            }
            PlanOperation::MatMul {
                left,
                right,
                output,
                rows,
                columns,
                ..
            } => (*left, Some(*right), *output, rows * columns),
            PlanOperation::Transformer { output, .. } => (0, None, *output, buffers[*output].len()),
        };
        // PlanBuilder emits each output after all of its inputs. Splitting here
        // makes non-aliasing visible to Rust as well as to the CUDA launcher.
        let (sources, destinations) = buffers.split_at_mut(output);
        let destination = &mut destinations[0];
        let config = launch_config(elements)?;
        // SAFETY: lowering validated tensor shapes, every source precedes its
        // distinct output allocation, and kernels guard their global index.
        unsafe {
            match operation {
                PlanOperation::Add { .. } => module.add(
                    stream,
                    config,
                    &sources[left],
                    &sources[right.expect("binary right input")],
                    destination,
                ),
                PlanOperation::Mul { .. } => module.mul(
                    stream,
                    config,
                    &sources[left],
                    &sources[right.expect("binary right input")],
                    destination,
                ),
                PlanOperation::Relu { .. } => {
                    module.relu(stream, config, &sources[left], destination)
                }
                PlanOperation::MatMul {
                    rows,
                    columns,
                    inner,
                    ..
                } => module.matmul(
                    stream,
                    config,
                    *rows,
                    *columns,
                    *inner,
                    &sources[left],
                    &sources[right.expect("matmul right input")],
                    destination,
                ),
                PlanOperation::Transformer {
                    inputs,
                    input_shapes,
                    primitive,
                    ..
                } => launch_transformer(
                    transformer,
                    stream,
                    config,
                    *primitive,
                    inputs,
                    input_shapes,
                    sources,
                    destination,
                ),
            }
        }
        .map_err(cuda_error)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_transformer(
    module: &kernels::transformer::LoadedModule,
    stream: &CudaStream,
    config: LaunchConfig,
    primitive: crate::transformer::Primitive,
    inputs: &[usize],
    shapes: &[Vec<usize>],
    buffers: &[DeviceBuffer<f32>],
    output: &mut DeviceBuffer<f32>,
) -> std::result::Result<(), cuda_core::DriverError> {
    // SAFETY: GraphPlan shape validation and fixed non-aliasing buffers satisfy
    // every generated kernel launch contract.
    unsafe {
        match primitive {
            crate::transformer::Primitive::Linear => module.linear(
                stream,
                config,
                shapes[0][shapes[0].len() - 1],
                shapes[1][0],
                &buffers[inputs[0]],
                &buffers[inputs[1]],
                &buffers[inputs[2]],
                output,
            ),
            crate::transformer::Primitive::Gelu => {
                module.gelu(stream, config, &buffers[inputs[0]], output)
            }
            crate::transformer::Primitive::Tanh => {
                module.tanh(stream, config, &buffers[inputs[0]], output)
            }
            crate::transformer::Primitive::Embedding => module.embedding(
                stream,
                config,
                shapes[1][1],
                shapes[1][0],
                &buffers[inputs[0]],
                &buffers[inputs[1]],
                output,
            ),
            crate::transformer::Primitive::LayerNorm { epsilon } => module.layer_norm(
                stream,
                config,
                shapes[1].iter().product(),
                epsilon,
                &buffers[inputs[0]],
                &buffers[inputs[1]],
                &buffers[inputs[2]],
                output,
            ),
            crate::transformer::Primitive::SelectFirst => module.select_first(
                stream,
                config,
                shapes[0][1],
                shapes[0][2],
                &buffers[inputs[0]],
                output,
            ),
            crate::transformer::Primitive::ScaledDotProductAttention { heads } => module
                .scaled_dot_product_attention(
                    stream,
                    config,
                    shapes[0][1],
                    shapes[0][2],
                    heads,
                    &buffers[inputs[0]],
                    &buffers[inputs[1]],
                    &buffers[inputs[2]],
                    &buffers[inputs[3]],
                    &buffers[inputs[4]],
                    &buffers[inputs[5]],
                    &buffers[inputs[6]],
                    &buffers[inputs[7]],
                    output,
                ),
        }
    }
}

fn launch_config(elements: usize) -> Result<LaunchConfig> {
    let elements = u32::try_from(elements)
        .map_err(|_| Error::Execution("CUDA JIT launch is too large".into()))?;
    Ok(LaunchConfig::for_num_elems(elements))
}

fn cuda_error(error: impl std::fmt::Display) -> Error {
    Error::Execution(format!("CUDA JIT execution failed: {error}"))
}
