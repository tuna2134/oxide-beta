//! Fixed-buffer CUDA execution plans for the public trace JIT.

use super::kernels;
use crate::cuda_graph::CudaGraphExec;
use crate::jit::{BufferPlan, GraphPlan, PlanOperation};
use crate::{Error, Result, Tensor};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use std::sync::{Arc, Mutex};

pub(crate) struct CudaJitPlan {
    state: Mutex<CudaJitState>,
}

struct CudaJitState {
    _context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
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
        let module = kernels::load(&context).map_err(cuda_error)?;
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
            launch_operations(&module, &stream, &plan.operations, &mut buffers)
        })?;
        Ok(Self {
            state: Mutex::new(CudaJitState {
                _context: context,
                stream,
                module,
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
            }
        }
        .map_err(cuda_error)?;
    }
    Ok(())
}

fn launch_config(elements: usize) -> Result<LaunchConfig> {
    let elements = u32::try_from(elements)
        .map_err(|_| Error::Execution("CUDA JIT launch is too large".into()))?;
    Ok(LaunchConfig::for_num_elems(elements))
}

fn cuda_error(error: impl std::fmt::Display) -> Error {
    Error::Execution(format!("CUDA JIT execution failed: {error}"))
}
