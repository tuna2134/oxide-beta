//! Gemma 4 persistent BF16 CUDA state.
#![allow(unsafe_code)]

use crate::models::gemma4::{Gemma4ForCausalLM, Gemma4TextConfig};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use oxide_torch::{Error, Result};
use oxide_torch_cuda::cublas::Cublas;
use oxide_torch_cuda::cuda_graph::CudaGraphExec;
use safetensors::Dtype;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::sync::Arc;

type BufferPool<T> = Rc<RefCell<HashMap<usize, Vec<DeviceBuffer<T>>>>>;

/// Stream-ordered temporary allocation returned to the per-model pool on
/// drop. Since every Gemma operation uses one CUDA stream, a buffer can be
/// reused immediately: later writes are ordered after all earlier readers.
struct WorkspaceBuffer<T> {
    buffer: Option<DeviceBuffer<T>>,
    pool: BufferPool<T>,
}

impl<T> WorkspaceBuffer<T> {
    fn into_inner(mut self) -> DeviceBuffer<T> {
        self.buffer.take().expect("workspace buffer is present")
    }
}

impl<T> Deref for WorkspaceBuffer<T> {
    type Target = DeviceBuffer<T>;

    fn deref(&self) -> &Self::Target {
        self.buffer.as_ref().expect("workspace buffer is present")
    }
}

impl<T> DerefMut for WorkspaceBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buffer.as_mut().expect("workspace buffer is present")
    }
}

impl<T> Drop for WorkspaceBuffer<T> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool
                .borrow_mut()
                .entry(buffer.len())
                .or_default()
                .push(buffer);
        }
    }
}

type WorkspaceF32 = WorkspaceBuffer<f32>;
type WorkspaceBf16 = WorkspaceBuffer<u16>;

/// Persistent CUDA resources for Gemma 4 inference.
///
/// All BF16 language-model weights are uploaded once during construction and
/// remain resident until this value is dropped.
struct CudaWeight {
    shape: Vec<usize>,
    buffer: DeviceBuffer<u16>,
}

struct Gemma4DecodeLayerPlan {
    source: usize,
    sliding: bool,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f32,
    rope_factor: f32,
    input_norm: usize,
    query_projection: usize,
    fused_qkv: bool,
    query_norm: usize,
    key_norm: Option<usize>,
    output_projection: usize,
    post_attention_norm: usize,
    pre_feedforward_norm: usize,
    gate_projection: usize,
    up_projection: usize,
    down_projection: usize,
    post_feedforward_norm: usize,
    ple_gate: Option<usize>,
    ple_projection: Option<usize>,
    ple_norm: Option<usize>,
    layer_scalar: usize,
}

/// Persistent fixed-allocation ring buffer for autoregressive K/V state.
pub struct Gemma4CudaKvCache {
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    kv_heads: usize,
    head_dim: usize,
    capacity: usize,
    start: usize,
    len: usize,
    total_seen: usize,
}

impl Gemma4CudaKvCache {
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Physical cache layout for all decoder layers, including Gemma 4 KV sharing.
pub struct Gemma4CudaCacheTable {
    layers: Vec<Option<Gemma4CudaKvCache>>,
    sources: Vec<usize>,
    position: usize,
    decode_state: DeviceBuffer<usize>,
    decode_graph: Option<Gemma4CudaDecodeGraph>,
    decode_graph_attempted: bool,
}

struct Gemma4CudaDecodeGraph {
    executable: CudaGraphExec,
    scores: DeviceBuffer<f32>,
    ids: DeviceBuffer<f32>,
    _fixed_f32: Vec<DeviceBuffer<f32>>,
    _fixed_bf16: Vec<DeviceBuffer<u16>>,
    top_k: usize,
    repetition_penalty: f32,
}

impl Gemma4CudaCacheTable {
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    #[must_use]
    pub fn physical_cache_count(&self) -> usize {
        self.layers.iter().filter(|cache| cache.is_some()).count()
    }

    #[must_use]
    pub fn shared_layer_count(&self) -> usize {
        self.layers.len() - self.physical_cache_count()
    }

    /// Returns the physical source layer used by `layer`.
    #[must_use]
    pub fn source_layer(&self, layer: usize) -> Option<usize> {
        self.sources.get(layer).copied()
    }
}

pub struct Gemma4CudaState {
    pub(crate) _context: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) cublas: Cublas,
    module: oxide_torch_cuda::kernels::LoadedModule,
    inference: oxide_torch_cuda::kernels::inference::LoadedModule,
    attention: oxide_torch_cuda::kernels::attention::LoadedModule,
    sampling: oxide_torch_cuda::kernels::sampling::LoadedModule,
    weights: Vec<CudaWeight>,
    weight_indices: HashMap<String, usize>,
    decode_plan: Vec<Gemma4DecodeLayerPlan>,
    f32_pool: BufferPool<f32>,
    bf16_pool: BufferPool<u16>,
    seen_tokens: RefCell<Option<DeviceBuffer<u8>>>,
}

impl Gemma4CudaState {
    pub(crate) fn reset_generation_state(&self) {
        *self.seen_tokens.borrow_mut() = None;
    }

    fn trace_fingerprint(
        &self,
        label: &str,
        value: &DeviceBuffer<f32>,
        enabled: bool,
    ) -> Result<()> {
        if !enabled {
            return Ok(());
        }
        let host = value.to_host_vec(&self.stream).map_err(cuda_error)?;
        let rms = (host.iter().map(|item| item * item).sum::<f32>() / host.len() as f32).sqrt();
        let maximum = host.iter().map(|item| item.abs()).fold(0.0_f32, f32::max);
        eprintln!(
            "Gemma4 trace {label}: rms={rms:.8} abs_max={maximum:.8} first={:?}",
            &host[..host.len().min(4)]
        );
        Ok(())
    }

    fn trace_weight_fingerprint(&self, label: &str, name: &str, enabled: bool) -> Result<()> {
        if !enabled {
            return Ok(());
        }
        let weight = self.weight(name)?;
        let mut value = self.output_f32(weight.buffer.len())?;
        // SAFETY: output has exactly the same extent as the BF16 weight.
        unsafe {
            self.inference.bf16_to_f32_scaled(
                &self.stream,
                Self::launch_config(weight.buffer.len())?,
                0,
                1.0,
                &weight.buffer,
                &mut value,
            )
        }
        .map_err(cuda_error)?;
        self.trace_fingerprint(label, &value, true)
    }
    pub(crate) fn load(model: &Gemma4ForCausalLM, device: usize) -> Result<Self> {
        let context = CudaContext::new(device).map_err(cuda_error)?;
        // CUDA Graph capture is unsupported on the legacy default stream.
        // Keep all Gemma allocations, kernels and cuBLAS calls on one
        // dedicated non-blocking stream for both normal execution and replay.
        let stream = context.new_stream().map_err(cuda_error)?;
        let cublas = Cublas::new()?;
        cublas.bind_stream(&stream)?;
        let module = oxide_torch_cuda::load_kernels(&context).map_err(cuda_error)?;
        let inference = oxide_torch_cuda::kernels::inference::LoadedModule::from_parent(&module)
            .map_err(cuda_error)?;
        let attention = oxide_torch_cuda::kernels::attention::LoadedModule::from_parent(&module)
            .map_err(cuda_error)?;
        let sampling = oxide_torch_cuda::kernels::sampling::LoadedModule::from_parent(&module)
            .map_err(cuda_error)?;
        let names: Vec<_> = model
            .checkpoint_weight_names()
            .filter(|name| {
                name.starts_with("model.language_model.")
                    || name == &"model.multi_modal_projector.linear.weight"
            })
            .map(str::to_owned)
            .collect();
        let mut weights = HashMap::with_capacity(names.len());
        for name in names {
            let weight = model.with_checkpoint_view(&name, |dtype, shape, bytes| {
                if dtype != Dtype::BF16 {
                    return Err(Error::Execution(format!(
                        "Gemma 4 CUDA weight `{name}` is {dtype:?}, expected BF16"
                    )));
                }
                let elements = shape.iter().try_fold(1usize, |size, dimension| {
                    size.checked_mul(*dimension)
                        .ok_or_else(|| Error::InvalidShape("Gemma 4 weight shape overflow".into()))
                })?;
                if bytes.len() != elements * 2 || bytes.as_ptr().align_offset(2) != 0 {
                    return Err(Error::Execution(format!(
                        "Gemma 4 weight `{name}` has an invalid BF16 payload"
                    )));
                }
                // SAFETY: alignment and exact 2*elements byte extent were
                // checked; every u16 bit pattern is a valid BF16 carrier. The
                // synchronous copy completes before the mmap borrow ends.
                let host =
                    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u16>(), elements) };
                let buffer = DeviceBuffer::from_host(&stream, host).map_err(cuda_error)?;
                Ok(CudaWeight {
                    shape: shape.to_vec(),
                    buffer,
                })
            })?;
            let canonical_name = name
                .strip_prefix("model.language_model.")
                .or_else(|| name.strip_prefix("language_model."))
                .or_else(|| name.strip_prefix("model."))
                .unwrap_or(name.as_str())
                .to_owned();
            weights.insert(canonical_name, weight);
        }
        // Qwen/tinygrad-style fused QKV projection. Gemma's first physical
        // cache layers produce Q, K and V together; later KV-sharing layers
        // only produce Q and therefore keep the separate projection.
        let physical_layers = model
            .config()
            .num_hidden_layers
            .saturating_sub(model.config().num_kv_shared_layers);
        for layer in 0..physical_layers {
            let prefix = format!("layers.{layer}.self_attn");
            let q_name = format!("{prefix}.q_proj.weight");
            let k_name = format!("{prefix}.k_proj.weight");
            let v_name = format!("{prefix}.v_proj.weight");
            let (input_width, q_width, k_width, v_width, total_elements) = {
                let q = weights
                    .get(&q_name)
                    .ok_or_else(|| Error::Execution(format!("missing CUDA weight `{q_name}`")))?;
                let k = weights
                    .get(&k_name)
                    .ok_or_else(|| Error::Execution(format!("missing CUDA weight `{k_name}`")))?;
                let v = weights
                    .get(&v_name)
                    .ok_or_else(|| Error::Execution(format!("missing CUDA weight `{v_name}`")))?;
                if q.shape.len() != 2
                    || k.shape.len() != 2
                    || v.shape.len() != 2
                    || q.shape[1] != k.shape[1]
                    || q.shape[1] != v.shape[1]
                {
                    return Err(Error::InvalidShape(format!(
                        "layer {layer} QKV projection shapes do not match"
                    )));
                }
                (
                    q.shape[1],
                    q.shape[0],
                    k.shape[0],
                    v.shape[0],
                    q.buffer.len() + k.buffer.len() + v.buffer.len(),
                )
            };
            let mut fused = unsafe {
                DeviceBuffer::uninitialized_async(&stream, total_elements).map_err(cuda_error)?
            };
            let mut offset = 0;
            for name in [&q_name, &k_name, &v_name] {
                let weight = weights
                    .get(name)
                    .ok_or_else(|| Error::Execution(format!("missing CUDA weight `{name}`")))?;
                unsafe {
                    inference.copy_bf16(
                        &stream,
                        Self::launch_config(weight.buffer.len())?,
                        offset,
                        &weight.buffer,
                        &mut fused,
                    )
                }
                .map_err(cuda_error)?;
                offset += weight.buffer.len();
            }
            weights.remove(&q_name);
            weights.remove(&k_name);
            weights.remove(&v_name);
            weights.insert(
                format!("{prefix}.qkv_proj.weight"),
                CudaWeight {
                    shape: vec![q_width + k_width + v_width, input_width],
                    buffer: fused,
                },
            );
        }
        let mut named_weights: Vec<_> = weights.into_iter().collect();
        named_weights.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        let mut weight_indices = HashMap::with_capacity(named_weights.len());
        let mut weights = Vec::with_capacity(named_weights.len());
        for (name, weight) in named_weights {
            weight_indices.insert(name, weights.len());
            weights.push(weight);
        }
        let weight_index = |name: &str| {
            weight_indices
                .get(name)
                .copied()
                .ok_or_else(|| Error::Execution(format!("missing CUDA weight `{name}`")))
        };
        let layer_types = model
            .config()
            .layer_types
            .as_ref()
            .ok_or_else(|| Error::Execution("Gemma 4 layer_types are missing".into()))?;
        let mut decode_plan = Vec::with_capacity(model.config().num_hidden_layers);
        let mut last_sliding = None;
        let mut last_full = None;
        for (layer, layer_type) in layer_types.iter().enumerate() {
            let prefix = format!("layers.{layer}");
            let sliding = layer_type == "sliding_attention";
            let source = if layer < physical_layers {
                if sliding {
                    last_sliding = Some(layer);
                } else {
                    last_full = Some(layer);
                }
                layer
            } else if sliding {
                last_sliding.ok_or_else(|| {
                    Error::Execution(format!("layer {layer} has no sliding KV source"))
                })?
            } else {
                last_full.ok_or_else(|| {
                    Error::Execution(format!("layer {layer} has no global KV source"))
                })?
            };
            let head_dim = if sliding {
                model.config().head_dim
            } else {
                model.config().global_head_dim
            };
            let rope = model
                .config()
                .rope_parameters
                .as_ref()
                .and_then(|parameters| parameters.get(layer_type))
                .ok_or_else(|| {
                    Error::Execution(format!("RoPE parameters for {layer_type} missing"))
                })?;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let rotary_dim = (rope.partial_rotary_factor * head_dim as f32) as usize;
            decode_plan.push(Gemma4DecodeLayerPlan {
                source,
                sliding,
                head_dim,
                rotary_dim,
                rope_theta: rope.rope_theta,
                rope_factor: rope.factor,
                input_norm: weight_index(&format!("{prefix}.input_layernorm.weight"))?,
                query_projection: weight_index(&if source == layer {
                    format!("{prefix}.self_attn.qkv_proj.weight")
                } else {
                    format!("{prefix}.self_attn.q_proj.weight")
                })?,
                fused_qkv: source == layer,
                query_norm: weight_index(&format!("{prefix}.self_attn.q_norm.weight"))?,
                key_norm: (source == layer)
                    .then(|| weight_index(&format!("{prefix}.self_attn.k_norm.weight")))
                    .transpose()?,
                output_projection: weight_index(&format!("{prefix}.self_attn.o_proj.weight"))?,
                post_attention_norm: weight_index(&format!(
                    "{prefix}.post_attention_layernorm.weight"
                ))?,
                pre_feedforward_norm: weight_index(&format!(
                    "{prefix}.pre_feedforward_layernorm.weight"
                ))?,
                gate_projection: weight_index(&format!("{prefix}.mlp.gate_proj.weight"))?,
                up_projection: weight_index(&format!("{prefix}.mlp.up_proj.weight"))?,
                down_projection: weight_index(&format!("{prefix}.mlp.down_proj.weight"))?,
                post_feedforward_norm: weight_index(&format!(
                    "{prefix}.post_feedforward_layernorm.weight"
                ))?,
                ple_gate: (model.config().hidden_size_per_layer_input != 0)
                    .then(|| weight_index(&format!("{prefix}.per_layer_input_gate.weight")))
                    .transpose()?,
                ple_projection: (model.config().hidden_size_per_layer_input != 0)
                    .then(|| weight_index(&format!("{prefix}.per_layer_projection.weight")))
                    .transpose()?,
                ple_norm: (model.config().hidden_size_per_layer_input != 0)
                    .then(|| weight_index(&format!("{prefix}.post_per_layer_input_norm.weight")))
                    .transpose()?,
                layer_scalar: weight_index(&format!("{prefix}.layer_scalar"))?,
            });
        }
        stream.synchronize().map_err(cuda_error)?;
        Ok(Self {
            _context: context,
            stream,
            cublas,
            module,
            inference,
            attention,
            sampling,
            weights,
            weight_indices,
            decode_plan,
            f32_pool: Rc::new(RefCell::new(HashMap::new())),
            bf16_pool: Rc::new(RefCell::new(HashMap::new())),
            seen_tokens: RefCell::new(None),
        })
    }

    /// Number of persistent checkpoint tensors stored on the GPU.
    #[must_use]
    pub fn weight_count(&self) -> usize {
        self.weights.len()
    }

    /// Total bytes occupied by persistent BF16 checkpoint tensors.
    #[must_use]
    pub fn weight_bytes(&self) -> usize {
        self.weights
            .iter()
            .map(|weight| weight.buffer.len() * std::mem::size_of::<u16>())
            .sum()
    }

    /// Waits until all queued Gemma CUDA work has completed. This is exposed
    /// for opt-in profiling; normal inference synchronizes only when logits
    /// are copied to the host.
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(cuda_error)
    }

    fn weight(&self, suffix: &str) -> Result<&CudaWeight> {
        let aliases = [
            format!("model.{suffix}"),
            format!("model.language_model.{suffix}"),
            format!("language_model.{suffix}"),
            suffix.to_owned(),
        ];
        aliases
            .iter()
            .find_map(|name| self.weight_indices.get(name).copied())
            .map(|index| &self.weights[index])
            .ok_or_else(|| Error::Execution(format!("CUDA weight `{suffix}` is missing")))
    }

    fn weight_at(&self, index: usize) -> Result<&CudaWeight> {
        self.weights
            .get(index)
            .ok_or_else(|| Error::Execution("CUDA decode-plan weight index is invalid".into()))
    }

    fn output_f32(&self, len: usize) -> Result<WorkspaceF32> {
        if let Some(buffer) = self.f32_pool.borrow_mut().get_mut(&len).and_then(Vec::pop) {
            return Ok(WorkspaceBuffer {
                buffer: Some(buffer),
                pool: Rc::clone(&self.f32_pool),
            });
        }
        // SAFETY: every caller launches a kernel/cuBLAS operation that writes
        // all elements on `self.stream` before any read or synchronization.
        let buffer =
            unsafe { DeviceBuffer::uninitialized_async(&self.stream, len) }.map_err(cuda_error)?;
        Ok(WorkspaceBuffer {
            buffer: Some(buffer),
            pool: Rc::clone(&self.f32_pool),
        })
    }

    fn output_bf16(&self, len: usize) -> Result<WorkspaceBf16> {
        if let Some(buffer) = self.bf16_pool.borrow_mut().get_mut(&len).and_then(Vec::pop) {
            return Ok(WorkspaceBuffer {
                buffer: Some(buffer),
                pool: Rc::clone(&self.bf16_pool),
            });
        }
        // SAFETY: same invariant as `output_f32`.
        let buffer =
            unsafe { DeviceBuffer::uninitialized_async(&self.stream, len) }.map_err(cuda_error)?;
        Ok(WorkspaceBuffer {
            buffer: Some(buffer),
            pool: Rc::clone(&self.bf16_pool),
        })
    }

    fn launch_config(len: usize) -> Result<LaunchConfig> {
        if len == 0 {
            return Err(Error::InvalidShape("zero-sized Gemma CUDA launch".into()));
        }
        Ok(LaunchConfig::for_num_elems(len as u32))
    }

    fn row_launch_config(rows: usize) -> Result<LaunchConfig> {
        let rows = u32::try_from(rows)
            .map_err(|_| Error::InvalidShape("too many Gemma CUDA rows".into()))?;
        if rows == 0 {
            return Err(Error::InvalidShape(
                "zero-sized Gemma CUDA row launch".into(),
            ));
        }
        Ok(LaunchConfig {
            grid_dim: (rows, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }
}

mod attention;
mod cache;
mod decode;
mod ple;
mod smoke;
mod tensor_ops;

fn apply_logit_softcap(logits: &mut [f32], softcap: Option<f32>) {
    if let Some(cap) = softcap {
        for logit in logits {
            *logit = (*logit / cap).tanh() * cap;
        }
    }
}

fn cuda_error(error: impl std::fmt::Display) -> Error {
    Error::Execution(format!("Gemma 4 CUDA error: {error}"))
}
