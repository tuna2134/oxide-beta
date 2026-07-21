//! Gemma 4 persistent BF16 CUDA state.
#![allow(unsafe_code)]

use crate::cublas::Cublas;
use crate::cuda_graph::CudaGraphExec;
use crate::models::gemma4::{Gemma4ForCausalLM, Gemma4TextConfig};
use crate::{Error, Result};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
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
    module: crate::cuda::kernels::LoadedModule,
    weights: Vec<CudaWeight>,
    weight_indices: HashMap<String, usize>,
    decode_plan: Vec<Gemma4DecodeLayerPlan>,
    f32_pool: BufferPool<f32>,
    bf16_pool: BufferPool<u16>,
    seen_tokens: RefCell<Option<DeviceBuffer<u8>>>,
}

impl Gemma4CudaState {
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
            self.module.gemma_bf16_to_f32_scaled(
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
        let stream = context.default_stream();
        let cublas = Cublas::new()?;
        cublas.bind_stream(&stream)?;
        let module = crate::cuda::kernels::load(&context).map_err(cuda_error)?;
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
                    module.gemma_copy_bf16(
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

    fn embedding(&self, token: u32, hidden_size: usize) -> Result<WorkspaceF32> {
        let embedding = self.weight("embed_tokens.weight")?;
        if embedding.shape.len() != 2 || embedding.shape[1] != hidden_size {
            return Err(Error::InvalidShape(format!(
                "embed_tokens.weight has unexpected shape {:?}",
                embedding.shape
            )));
        }
        let token = token as usize;
        if token >= embedding.shape[0] {
            return Err(Error::Execution(
                "token id exceeds CUDA embedding table".into(),
            ));
        }
        let mut hidden = self.output_f32(hidden_size)?;
        #[allow(clippy::cast_precision_loss)]
        let scale = (hidden_size as f32).sqrt();
        // SAFETY: bounds and output extent were validated above.
        unsafe {
            self.module.gemma_bf16_to_f32_scaled(
                &self.stream,
                Self::launch_config(hidden_size)?,
                token * hidden_size,
                scale,
                &embedding.buffer,
                &mut hidden,
            )
        }
        .map_err(cuda_error)?;
        Ok(hidden)
    }

    fn set_decode_state(&self, token: u32, table: &mut Gemma4CudaCacheTable) -> Result<()> {
        unsafe {
            self.module.gemma_decode_state_set(
                &self.stream,
                Self::launch_config(1)?,
                token as usize,
                table.position,
                &mut table.decode_state,
            )
        }
        .map_err(cuda_error)
    }

    fn embedding_state(
        &self,
        state: &DeviceBuffer<usize>,
        hidden_size: usize,
    ) -> Result<WorkspaceF32> {
        let embedding = self.weight("embed_tokens.weight")?;
        if embedding.shape.len() != 2 || embedding.shape[1] != hidden_size {
            return Err(Error::InvalidShape(
                "decode-state embedding shape mismatch".into(),
            ));
        }
        let mut hidden = self.output_f32(hidden_size)?;
        #[allow(clippy::cast_precision_loss)]
        let scale = (hidden_size as f32).sqrt();
        unsafe {
            self.module.gemma_bf16_row_scaled_state(
                &self.stream,
                Self::launch_config(hidden_size)?,
                hidden_size,
                scale,
                state,
                &embedding.buffer,
                &mut hidden,
            )
        }
        .map_err(cuda_error)?;
        Ok(hidden)
    }

    fn embedding_rows(&self, tokens: &[u32], hidden_size: usize) -> Result<WorkspaceF32> {
        let embedding = self.weight("embed_tokens.weight")?;
        if tokens.is_empty()
            || embedding.shape.len() != 2
            || embedding.shape[1] != hidden_size
            || tokens
                .iter()
                .any(|&token| token as usize >= embedding.shape[0])
        {
            return Err(Error::InvalidShape(
                "invalid batched CUDA embedding input".into(),
            ));
        }
        let token_buffer = DeviceBuffer::from_host(&self.stream, tokens).map_err(cuda_error)?;
        let mut hidden = self.output_f32(tokens.len() * hidden_size)?;
        #[allow(clippy::cast_precision_loss)]
        let scale = (hidden_size as f32).sqrt();
        unsafe {
            self.module.gemma_embedding_rows(
                &self.stream,
                Self::launch_config(hidden.len())?,
                hidden_size,
                scale,
                &token_buffer,
                &embedding.buffer,
                &mut hidden,
            )
        }
        .map_err(cuda_error)?;
        Ok(hidden)
    }

    fn to_bf16(&self, input: &DeviceBuffer<f32>) -> Result<WorkspaceBf16> {
        let mut output = self.output_bf16(input.len())?;
        // SAFETY: input/output have equal lengths and are disjoint.
        unsafe {
            self.module.gemma_f32_to_bf16(
                &self.stream,
                Self::launch_config(input.len())?,
                input,
                &mut output,
            )
        }
        .map_err(cuda_error)?;
        Ok(output)
    }

    fn rms_norm(
        &self,
        input: &DeviceBuffer<f32>,
        weight_name: &str,
        epsilon: f32,
    ) -> Result<WorkspaceF32> {
        let weight = self.weight(weight_name)?;
        let hidden = *weight
            .shape
            .first()
            .filter(|_| weight.shape.len() == 1)
            .ok_or_else(|| Error::InvalidShape(format!("{weight_name} is not one-dimensional")))?;
        if input.len() % hidden != 0 {
            return Err(Error::InvalidShape(format!(
                "{weight_name} does not match its RMSNorm input"
            )));
        }
        let mut output = self.output_f32(input.len())?;
        // SAFETY: the kernel indexes weight by column modulo `hidden`; input
        // consists of complete rows and output has the same extent.
        unsafe {
            self.module.gemma_rms_norm(
                &self.stream,
                Self::row_launch_config(input.len() / hidden)?,
                hidden,
                epsilon,
                input,
                &weight.buffer,
                &mut output,
            )
        }
        .map_err(cuda_error)?;
        Ok(output)
    }

    fn rms_norm_at(
        &self,
        input: &DeviceBuffer<f32>,
        weight_index: usize,
        epsilon: f32,
    ) -> Result<WorkspaceF32> {
        let weight = self.weight_at(weight_index)?;
        let hidden = *weight
            .shape
            .first()
            .filter(|_| weight.shape.len() == 1)
            .ok_or_else(|| Error::InvalidShape("decode-plan RMSNorm weight is invalid".into()))?;
        if input.len() % hidden != 0 {
            return Err(Error::InvalidShape(
                "decode-plan RMSNorm input mismatch".into(),
            ));
        }
        let mut output = self.output_f32(input.len())?;
        unsafe {
            self.module.gemma_rms_norm(
                &self.stream,
                Self::row_launch_config(input.len() / hidden)?,
                hidden,
                epsilon,
                input,
                &weight.buffer,
                &mut output,
            )
        }
        .map_err(cuda_error)?;
        Ok(output)
    }

    fn rms_norm_bf16(
        &self,
        input: &DeviceBuffer<f32>,
        weight_name: &str,
        epsilon: f32,
    ) -> Result<WorkspaceBf16> {
        let weight = self.weight(weight_name)?;
        let hidden = *weight
            .shape
            .first()
            .filter(|_| weight.shape.len() == 1)
            .ok_or_else(|| Error::InvalidShape(format!("{weight_name} is not one-dimensional")))?;
        if input.len() % hidden != 0 {
            return Err(Error::InvalidShape(format!(
                "{weight_name} does not match its RMSNorm input"
            )));
        }
        let mut output = self.output_bf16(input.len())?;
        unsafe {
            self.module.gemma_rms_norm_bf16(
                &self.stream,
                Self::row_launch_config(input.len() / hidden)?,
                hidden,
                epsilon,
                input,
                &weight.buffer,
                &mut output,
            )
        }
        .map_err(cuda_error)?;
        Ok(output)
    }

    fn rms_norm_bf16_at(
        &self,
        input: &DeviceBuffer<f32>,
        weight_index: usize,
        epsilon: f32,
    ) -> Result<WorkspaceBf16> {
        let weight = self.weight_at(weight_index)?;
        let hidden = *weight
            .shape
            .first()
            .filter(|_| weight.shape.len() == 1)
            .ok_or_else(|| Error::InvalidShape("decode-plan RMSNorm weight is invalid".into()))?;
        if input.len() % hidden != 0 {
            return Err(Error::InvalidShape(
                "decode-plan RMSNorm input mismatch".into(),
            ));
        }
        let mut output = self.output_bf16(input.len())?;
        unsafe {
            self.module.gemma_rms_norm_bf16(
                &self.stream,
                Self::row_launch_config(input.len() / hidden)?,
                hidden,
                epsilon,
                input,
                &weight.buffer,
                &mut output,
            )
        }
        .map_err(cuda_error)?;
        Ok(output)
    }

    fn rms_norm_unit(
        &self,
        input: &DeviceBuffer<f32>,
        hidden: usize,
        epsilon: f32,
    ) -> Result<WorkspaceF32> {
        if hidden == 0 || input.len() % hidden != 0 {
            return Err(Error::InvalidShape("unit RMSNorm shape mismatch".into()));
        }
        let mut output = self.output_f32(input.len())?;
        // SAFETY: input consists of complete rows of width `hidden` and
        // output has the same extent.
        unsafe {
            self.module.gemma_rms_norm_unit(
                &self.stream,
                Self::row_launch_config(input.len() / hidden)?,
                hidden,
                epsilon,
                input,
                &mut output,
            )
        }
        .map_err(cuda_error)?;
        Ok(output)
    }

    fn rope(
        &self,
        input: &DeviceBuffer<f32>,
        heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        position: usize,
        theta: f32,
        factor: f32,
    ) -> Result<WorkspaceF32> {
        if heads == 0 || head_dim == 0 || input.len() % (heads * head_dim) != 0 {
            return Err(Error::InvalidShape("RoPE shape mismatch".into()));
        }
        let mut output = self.output_f32(input.len())?;
        // SAFETY: the input contains complete `[heads, head_dim]` rows and
        // output has an identical extent.
        unsafe {
            self.module.gemma_rope(
                &self.stream,
                Self::launch_config(input.len())?,
                heads,
                head_dim,
                rotary_dim,
                position,
                theta,
                factor,
                input,
                &mut output,
            )
        }
        .map_err(cuda_error)?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn rope_state(
        &self,
        input: &DeviceBuffer<f32>,
        heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        theta: f32,
        factor: f32,
        state: &DeviceBuffer<usize>,
    ) -> Result<WorkspaceF32> {
        if heads == 0 || head_dim == 0 || input.len() % (heads * head_dim) != 0 {
            return Err(Error::InvalidShape("state RoPE shape mismatch".into()));
        }
        let mut output = self.output_f32(input.len())?;
        unsafe {
            self.module.gemma_rope_state(
                &self.stream,
                Self::launch_config(input.len())?,
                heads,
                head_dim,
                rotary_dim,
                theta,
                factor,
                state,
                input,
                &mut output,
            )
        }
        .map_err(cuda_error)?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn gqa(
        &self,
        query: &DeviceBuffer<f32>,
        key: &DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        sequence: usize,
        window: usize,
        cache_start: usize,
        cache_capacity: usize,
    ) -> Result<WorkspaceF32> {
        if query.len() != heads * head_dim
            || key.len() != cache_capacity * kv_heads * head_dim
            || value.len() != key.len()
            || heads % kv_heads != 0
            || cache_capacity == 0
            || cache_start >= cache_capacity
        {
            return Err(Error::InvalidShape("GQA shape mismatch".into()));
        }
        let mut output = self.output_f32(query.len())?;
        // SAFETY: all GQA layouts and divisibility constraints were checked.
        let config = if sequence <= 4096 {
            LaunchConfig {
                grid_dim: (heads as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            }
        } else {
            Self::launch_config(query.len())?
        };
        unsafe {
            if sequence <= 4096 {
                self.module.gemma_gqa_decode_block(
                    &self.stream,
                    config,
                    heads,
                    kv_heads,
                    head_dim,
                    sequence,
                    window,
                    cache_start,
                    cache_capacity,
                    query,
                    key,
                    value,
                    &mut output,
                )
            } else {
                self.module.gemma_gqa_decode(
                    &self.stream,
                    config,
                    heads,
                    kv_heads,
                    head_dim,
                    sequence,
                    window,
                    cache_start,
                    cache_capacity,
                    query,
                    key,
                    value,
                    &mut output,
                )
            }
        }
        .map_err(cuda_error)?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn gqa_state(
        &self,
        query: &DeviceBuffer<f32>,
        key: &DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        window: usize,
        cache_capacity: usize,
        use_block: bool,
        state: &DeviceBuffer<usize>,
    ) -> Result<WorkspaceF32> {
        if query.len() != heads * head_dim
            || key.len() != cache_capacity * kv_heads * head_dim
            || value.len() != key.len()
            || heads % kv_heads != 0
            || cache_capacity == 0
        {
            return Err(Error::InvalidShape("state GQA shape mismatch".into()));
        }
        let mut output = self.output_f32(query.len())?;
        unsafe {
            if use_block {
                self.module.gemma_gqa_decode_block_state(
                    &self.stream,
                    LaunchConfig {
                        grid_dim: (heads as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    heads,
                    kv_heads,
                    head_dim,
                    window,
                    cache_capacity,
                    state,
                    query,
                    key,
                    value,
                    &mut output,
                )
            } else {
                self.module.gemma_gqa_decode_state(
                    &self.stream,
                    Self::launch_config(query.len())?,
                    heads,
                    kv_heads,
                    head_dim,
                    window,
                    cache_capacity,
                    state,
                    query,
                    key,
                    value,
                    &mut output,
                )
            }
        }
        .map_err(cuda_error)?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn gqa_prefill(
        &self,
        query: &DeviceBuffer<f32>,
        key: &DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        rows: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        window: usize,
    ) -> Result<WorkspaceF32> {
        if rows == 0
            || rows > 4096
            || query.len() != rows * heads * head_dim
            || key.len() < rows * kv_heads * head_dim
            || value.len() != key.len()
            || heads % kv_heads != 0
        {
            return Err(Error::InvalidShape("prefill GQA shape mismatch".into()));
        }
        let blocks = rows
            .checked_mul(heads)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| Error::InvalidShape("prefill GQA grid overflow".into()))?;
        let mut output = self.output_f32(query.len())?;
        unsafe {
            self.module.gemma_gqa_prefill_block(
                &self.stream,
                LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                rows,
                heads,
                kv_heads,
                head_dim,
                window,
                query,
                key,
                value,
                &mut output,
            )
        }
        .map_err(cuda_error)?;
        Ok(output)
    }

    /// Allocates a reusable ring-buffer KV cache.
    pub fn new_kv_cache(
        &self,
        kv_heads: usize,
        head_dim: usize,
        capacity: usize,
    ) -> Result<Gemma4CudaKvCache> {
        let elements = capacity
            .checked_mul(kv_heads)
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or_else(|| Error::InvalidShape("KV cache size overflow".into()))?;
        if elements == 0 {
            return Err(Error::InvalidShape("zero-sized KV cache".into()));
        }
        Ok(Gemma4CudaKvCache {
            key: DeviceBuffer::zeroed(&self.stream, elements).map_err(cuda_error)?,
            value: DeviceBuffer::zeroed(&self.stream, elements).map_err(cuda_error)?,
            kv_heads,
            head_dim,
            capacity,
            start: 0,
            len: 0,
            total_seen: 0,
        })
    }

    /// Allocates the complete 35-layer cache table with KV sharing.
    pub fn new_cache_table(
        &self,
        config: &Gemma4TextConfig,
        max_sequence: usize,
    ) -> Result<Gemma4CudaCacheTable> {
        if max_sequence == 0 {
            return Err(Error::InvalidShape("zero max_sequence".into()));
        }
        let layer_types = config
            .layer_types
            .as_ref()
            .ok_or_else(|| Error::Execution("Gemma 4 layer_types are missing".into()))?;
        let first_shared = config
            .num_hidden_layers
            .saturating_sub(config.num_kv_shared_layers);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let mut sources = Vec::with_capacity(config.num_hidden_layers);
        let mut last_sliding = None;
        let mut last_full = None;
        for (layer, layer_type) in layer_types.iter().enumerate() {
            let sliding = layer_type == "sliding_attention";
            if layer < first_shared {
                let head_dim = if sliding {
                    config.head_dim
                } else {
                    config.global_head_dim
                };
                let capacity = if sliding {
                    config.sliding_window.min(max_sequence)
                } else {
                    max_sequence
                };
                let kv_heads = if !sliding && config.attention_k_eq_v {
                    config
                        .num_global_key_value_heads
                        .unwrap_or(config.num_key_value_heads)
                } else {
                    config.num_key_value_heads
                };
                layers.push(Some(self.new_kv_cache(kv_heads, head_dim, capacity)?));
                sources.push(layer);
                if sliding {
                    last_sliding = Some(layer);
                } else {
                    last_full = Some(layer);
                }
            } else {
                let source = if sliding { last_sliding } else { last_full }.ok_or_else(|| {
                    Error::Execution(format!(
                        "shared layer {layer} has no preceding {layer_type} KV source"
                    ))
                })?;
                layers.push(None);
                sources.push(source);
            }
        }
        Ok(Gemma4CudaCacheTable {
            layers,
            sources,
            position: 0,
            decode_state: DeviceBuffer::zeroed(&self.stream, 2).map_err(cuda_error)?,
            decode_graph: None,
            decode_graph_attempted: false,
        })
    }

    fn append_kv(
        &self,
        cache: &mut Gemma4CudaKvCache,
        key: &DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
    ) -> Result<()> {
        let width = cache.kv_heads * cache.head_dim;
        if key.len() != width || value.len() != width {
            return Err(Error::InvalidShape("KV cache append shape mismatch".into()));
        }
        let position = if cache.len < cache.capacity {
            (cache.start + cache.len) % cache.capacity
        } else {
            cache.start
        };
        let offset = position * width;
        // SAFETY: one complete cache position is in bounds and allocations
        // are disjoint.
        unsafe {
            self.module.gemma_cache_write(
                &self.stream,
                Self::launch_config(width)?,
                offset,
                key,
                &mut cache.key,
            )
        }
        .map_err(cuda_error)?;
        // SAFETY: same invariant for the separate value allocation.
        unsafe {
            self.module.gemma_cache_write(
                &self.stream,
                Self::launch_config(width)?,
                offset,
                value,
                &mut cache.value,
            )
        }
        .map_err(cuda_error)?;
        if cache.len < cache.capacity {
            cache.len += 1;
        } else {
            cache.start = (cache.start + 1) % cache.capacity;
        }
        cache.total_seen += 1;
        Ok(())
    }

    fn append_kv_state(
        &self,
        cache: &mut Gemma4CudaKvCache,
        key: &DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        state: &DeviceBuffer<usize>,
    ) -> Result<()> {
        let width = cache.kv_heads * cache.head_dim;
        if key.len() != width || value.len() != width {
            return Err(Error::InvalidShape(
                "state KV cache append shape mismatch".into(),
            ));
        }
        unsafe {
            self.module.gemma_cache_write_state(
                &self.stream,
                Self::launch_config(width)?,
                width,
                cache.capacity,
                state,
                key,
                &mut cache.key,
            )
        }
        .map_err(cuda_error)?;
        unsafe {
            self.module.gemma_cache_write_state(
                &self.stream,
                Self::launch_config(width)?,
                width,
                cache.capacity,
                state,
                value,
                &mut cache.value,
            )
        }
        .map_err(cuda_error)
    }

    fn linear(
        &self,
        input: &DeviceBuffer<f32>,
        rows: usize,
        weight_name: &str,
    ) -> Result<WorkspaceF32> {
        let weight = self.weight(weight_name)?;
        if weight.shape.len() != 2 {
            return Err(Error::InvalidShape(format!(
                "{weight_name} is not a matrix"
            )));
        }
        let input_width = weight.shape[1];
        if input.len() != rows * input_width {
            return Err(Error::InvalidShape(format!(
                "{weight_name} input has {} elements, expected {}",
                input.len(),
                rows * input_width
            )));
        }
        let input_bf16 = self.to_bf16(input)?;
        self.linear_bf16(&input_bf16, rows, weight_name)
    }

    fn linear_bf16(
        &self,
        input: &DeviceBuffer<u16>,
        rows: usize,
        weight_name: &str,
    ) -> Result<WorkspaceF32> {
        let weight = self.weight(weight_name)?;
        if weight.shape.len() != 2 {
            return Err(Error::InvalidShape(format!(
                "{weight_name} is not a matrix"
            )));
        }
        let (output_width, input_width) = (weight.shape[0], weight.shape[1]);
        if input.len() != rows * input_width {
            return Err(Error::InvalidShape(format!(
                "{weight_name} BF16 input has {} elements, expected {}",
                input.len(),
                rows * input_width
            )));
        }
        let mut output = self.output_f32(rows * output_width)?;
        self.cublas.linear_bf16_f32(
            &self.stream,
            rows,
            output_width,
            input_width,
            input,
            &weight.buffer,
            &mut output,
        )?;
        Ok(output)
    }

    fn linear_bf16_at(
        &self,
        input: &DeviceBuffer<u16>,
        rows: usize,
        weight_index: usize,
    ) -> Result<WorkspaceF32> {
        let weight = self.weight_at(weight_index)?;
        if weight.shape.len() != 2 {
            return Err(Error::InvalidShape(
                "decode-plan linear weight is not a matrix".into(),
            ));
        }
        let (output_width, input_width) = (weight.shape[0], weight.shape[1]);
        if input.len() != rows * input_width {
            return Err(Error::InvalidShape(
                "decode-plan BF16 linear input mismatch".into(),
            ));
        }
        let mut output = self.output_f32(rows * output_width)?;
        self.cublas.linear_bf16_f32(
            &self.stream,
            rows,
            output_width,
            input_width,
            input,
            &weight.buffer,
            &mut output,
        )?;
        Ok(output)
    }

    fn linear_at(
        &self,
        input: &DeviceBuffer<f32>,
        rows: usize,
        weight_index: usize,
    ) -> Result<WorkspaceF32> {
        let input = self.to_bf16(input)?;
        self.linear_bf16_at(&input, rows, weight_index)
    }

    fn linear_qkv_bf16(
        &self,
        input: &DeviceBuffer<u16>,
        rows: usize,
        prefix: &str,
        query_width: usize,
        key_width: usize,
        value_width: usize,
    ) -> Result<(WorkspaceF32, WorkspaceF32, WorkspaceF32)> {
        let fused_name = format!("{prefix}.self_attn.qkv_proj.weight");
        let fused = self.linear_bf16(input, rows, &fused_name)?;
        let row_width = query_width + key_width + value_width;
        if fused.len() != rows * row_width {
            return Err(Error::InvalidShape(format!(
                "{fused_name} output has an unexpected extent"
            )));
        }
        let query = self.slice_rows(&fused, rows, row_width, 0, query_width)?;
        let key = self.slice_rows(&fused, rows, row_width, query_width, key_width)?;
        let value = self.slice_rows(
            &fused,
            rows,
            row_width,
            query_width + key_width,
            value_width,
        )?;
        Ok((query, key, value))
    }

    fn linear_qkv_bf16_at(
        &self,
        input: &DeviceBuffer<u16>,
        rows: usize,
        weight_index: usize,
        query_width: usize,
        key_width: usize,
        value_width: usize,
    ) -> Result<(WorkspaceF32, WorkspaceF32, WorkspaceF32)> {
        let fused = self.linear_bf16_at(input, rows, weight_index)?;
        let row_width = query_width + key_width + value_width;
        if fused.len() != rows * row_width {
            return Err(Error::InvalidShape(
                "decode-plan fused QKV output mismatch".into(),
            ));
        }
        let query = self.slice_rows(&fused, rows, row_width, 0, query_width)?;
        let key = self.slice_rows(&fused, rows, row_width, query_width, key_width)?;
        let value = self.slice_rows(
            &fused,
            rows,
            row_width,
            query_width + key_width,
            value_width,
        )?;
        Ok((query, key, value))
    }

    fn gelu_mul(&self, gate: &DeviceBuffer<f32>, up: &DeviceBuffer<f32>) -> Result<WorkspaceF32> {
        if gate.len() != up.len() {
            return Err(Error::InvalidShape("GELU gate/up length mismatch".into()));
        }
        let mut output = self.output_f32(gate.len())?;
        // SAFETY: both inputs and output have equal extents and are disjoint.
        unsafe {
            self.module.gemma_gelu_mul(
                &self.stream,
                Self::launch_config(gate.len())?,
                gate,
                up,
                &mut output,
            )
        }
        .map_err(cuda_error)?;
        Ok(output)
    }

    fn add(&self, left: &DeviceBuffer<f32>, right: &DeviceBuffer<f32>) -> Result<WorkspaceF32> {
        if left.len() != right.len() {
            return Err(Error::InvalidShape("residual length mismatch".into()));
        }
        let mut output = self.output_f32(left.len())?;
        // SAFETY: both inputs and output have equal extents and are disjoint.
        unsafe {
            self.module.gemma_add(
                &self.stream,
                Self::launch_config(left.len())?,
                left,
                right,
                &mut output,
            )
        }
        .map_err(cuda_error)?;
        Ok(output)
    }

    fn mul(&self, left: &DeviceBuffer<f32>, right: &DeviceBuffer<f32>) -> Result<WorkspaceF32> {
        if left.len() != right.len() {
            return Err(Error::InvalidShape("elementwise multiply mismatch".into()));
        }
        let mut output = self.output_f32(left.len())?;
        // SAFETY: equal extents and three disjoint allocations.
        unsafe {
            self.module.gemma_mul(
                &self.stream,
                Self::launch_config(left.len())?,
                left,
                right,
                &mut output,
            )
        }
        .map_err(cuda_error)?;
        Ok(output)
    }

    fn scale(&self, input: &DeviceBuffer<f32>, scale: f32) -> Result<WorkspaceF32> {
        let mut output = self.output_f32(input.len())?;
        // SAFETY: input/output extents match and allocations are disjoint.
        unsafe {
            self.module.gemma_scale(
                &self.stream,
                Self::launch_config(input.len())?,
                scale,
                input,
                &mut output,
            )
        }
        .map_err(cuda_error)?;
        Ok(output)
    }

    fn scale_by_weight_at(
        &self,
        input: &DeviceBuffer<f32>,
        weight_index: usize,
    ) -> Result<WorkspaceF32> {
        let weight = self.weight_at(weight_index)?;
        if weight.buffer.len() != 1 {
            return Err(Error::InvalidShape(
                "decode-plan scalar weight is invalid".into(),
            ));
        }
        let mut output = self.output_f32(input.len())?;
        unsafe {
            self.module.gemma_mul_bf16_scalar(
                &self.stream,
                Self::launch_config(input.len())?,
                input,
                &weight.buffer,
                &mut output,
            )
        }
        .map_err(cuda_error)?;
        Ok(output)
    }

    fn slice(&self, input: &DeviceBuffer<f32>, offset: usize, len: usize) -> Result<WorkspaceF32> {
        if offset.checked_add(len).is_none_or(|end| end > input.len()) {
            return Err(Error::InvalidShape("CUDA slice is out of bounds".into()));
        }
        let mut output = self.output_f32(len)?;
        // SAFETY: the source range was checked and output has `len` elements.
        unsafe {
            self.module.gemma_slice(
                &self.stream,
                Self::launch_config(len)?,
                offset,
                input,
                &mut output,
            )
        }
        .map_err(cuda_error)?;
        Ok(output)
    }

    fn slice_rows(
        &self,
        input: &DeviceBuffer<f32>,
        rows: usize,
        input_width: usize,
        column_offset: usize,
        output_width: usize,
    ) -> Result<WorkspaceF32> {
        if rows == 0
            || input.len() != rows * input_width
            || column_offset
                .checked_add(output_width)
                .is_none_or(|end| end > input_width)
        {
            return Err(Error::InvalidShape(
                "CUDA row slice is out of bounds".into(),
            ));
        }
        let mut output = self.output_f32(rows * output_width)?;
        unsafe {
            self.module.gemma_slice_rows(
                &self.stream,
                Self::launch_config(output.len())?,
                input_width,
                output_width,
                column_offset,
                input,
                &mut output,
            )
        }
        .map_err(cuda_error)?;
        Ok(output)
    }

    fn gelu(&self, input: &DeviceBuffer<f32>) -> Result<WorkspaceF32> {
        let mut output = self.output_f32(input.len())?;
        // SAFETY: input/output extents match and allocations are disjoint.
        unsafe {
            self.module.gemma_gelu(
                &self.stream,
                Self::launch_config(input.len())?,
                input,
                &mut output,
            )
        }
        .map_err(cuda_error)?;
        Ok(output)
    }

    fn packed_ple(
        &self,
        token: u32,
        embedding: &DeviceBuffer<f32>,
        config: &Gemma4TextConfig,
    ) -> Result<Option<WorkspaceF32>> {
        let dimension = config.hidden_size_per_layer_input;
        if dimension == 0 {
            return Ok(None);
        }
        let packed = config.num_hidden_layers * dimension;
        let token_weight = self.weight("embed_tokens_per_layer.weight")?;
        if token_weight.shape.as_slice() != [config.vocab_size_per_layer_input, packed] {
            return Err(Error::InvalidShape(format!(
                "embed_tokens_per_layer.weight has shape {:?}",
                token_weight.shape
            )));
        }
        let token = token as usize;
        if token >= token_weight.shape[0] {
            return Err(Error::Execution("PLE token exceeds vocabulary".into()));
        }
        let mut token_ple = self.output_f32(packed)?;
        #[allow(clippy::cast_precision_loss)]
        let token_scale = (dimension as f32).sqrt();
        // SAFETY: token row and packed output extent were validated.
        unsafe {
            self.module.gemma_bf16_to_f32_scaled(
                &self.stream,
                Self::launch_config(packed)?,
                token * packed,
                token_scale,
                &token_weight.buffer,
                &mut token_ple,
            )
        }
        .map_err(cuda_error)?;
        let context = self.linear(embedding, 1, "per_layer_model_projection.weight")?;
        #[allow(clippy::cast_precision_loss)]
        let context = self.scale(&context, 1.0 / (config.hidden_size as f32).sqrt())?;
        let context = self.rms_norm(
            &context,
            "per_layer_projection_norm.weight",
            config.rms_norm_eps,
        )?;
        let combined = self.add(&context, &token_ple)?;
        Ok(Some(
            self.scale(&combined, core::f32::consts::FRAC_1_SQRT_2)?,
        ))
    }

    fn packed_ple_state(
        &self,
        state: &DeviceBuffer<usize>,
        embedding: &DeviceBuffer<f32>,
        config: &Gemma4TextConfig,
    ) -> Result<Option<WorkspaceF32>> {
        let dimension = config.hidden_size_per_layer_input;
        if dimension == 0 {
            return Ok(None);
        }
        let packed = config.num_hidden_layers * dimension;
        let token_weight = self.weight("embed_tokens_per_layer.weight")?;
        if token_weight.shape.as_slice() != [config.vocab_size_per_layer_input, packed] {
            return Err(Error::InvalidShape(
                "decode-state PLE embedding shape mismatch".into(),
            ));
        }
        let mut token_ple = self.output_f32(packed)?;
        #[allow(clippy::cast_precision_loss)]
        let token_scale = (dimension as f32).sqrt();
        unsafe {
            self.module.gemma_bf16_row_scaled_state(
                &self.stream,
                Self::launch_config(packed)?,
                packed,
                token_scale,
                state,
                &token_weight.buffer,
                &mut token_ple,
            )
        }
        .map_err(cuda_error)?;
        let context = self.linear(embedding, 1, "per_layer_model_projection.weight")?;
        #[allow(clippy::cast_precision_loss)]
        let context = self.scale(&context, 1.0 / (config.hidden_size as f32).sqrt())?;
        let context = self.rms_norm(
            &context,
            "per_layer_projection_norm.weight",
            config.rms_norm_eps,
        )?;
        let combined = self.add(&context, &token_ple)?;
        Ok(Some(
            self.scale(&combined, core::f32::consts::FRAC_1_SQRT_2)?,
        ))
    }

    fn packed_ple_rows(
        &self,
        tokens: &[u32],
        embedding: &DeviceBuffer<f32>,
        config: &Gemma4TextConfig,
    ) -> Result<Option<WorkspaceF32>> {
        let rows = tokens.len();
        let dimension = config.hidden_size_per_layer_input;
        if dimension == 0 {
            return Ok(None);
        }
        let packed = config.num_hidden_layers * dimension;
        let token_weight = self.weight("embed_tokens_per_layer.weight")?;
        if token_weight.shape.as_slice() != [config.vocab_size_per_layer_input, packed]
            || tokens
                .iter()
                .any(|&token| token as usize >= token_weight.shape[0])
        {
            return Err(Error::InvalidShape("invalid batched PLE input".into()));
        }
        let token_buffer = DeviceBuffer::from_host(&self.stream, tokens).map_err(cuda_error)?;
        let mut token_ple = self.output_f32(rows * packed)?;
        #[allow(clippy::cast_precision_loss)]
        let token_scale = (dimension as f32).sqrt();
        unsafe {
            self.module.gemma_embedding_rows(
                &self.stream,
                Self::launch_config(token_ple.len())?,
                packed,
                token_scale,
                &token_buffer,
                &token_weight.buffer,
                &mut token_ple,
            )
        }
        .map_err(cuda_error)?;
        let context = self.linear(embedding, rows, "per_layer_model_projection.weight")?;
        #[allow(clippy::cast_precision_loss)]
        let context = self.scale(&context, 1.0 / (config.hidden_size as f32).sqrt())?;
        let context = self.rms_norm(
            &context,
            "per_layer_projection_norm.weight",
            config.rms_norm_eps,
        )?;
        let combined = self.add(&context, &token_ple)?;
        Ok(Some(
            self.scale(&combined, core::f32::consts::FRAC_1_SQRT_2)?,
        ))
    }

    fn apply_ple(
        &self,
        hidden: &DeviceBuffer<f32>,
        packed_ple: &DeviceBuffer<f32>,
        layer: usize,
        config: &Gemma4TextConfig,
    ) -> Result<WorkspaceF32> {
        let dimension = config.hidden_size_per_layer_input;
        let per_layer = self.slice(packed_ple, layer * dimension, dimension)?;
        let plan = self
            .decode_plan
            .get(layer)
            .ok_or_else(|| Error::Execution("Gemma decode layer is missing".into()))?;
        let gate = self.linear_at(
            hidden,
            1,
            plan.ple_gate
                .ok_or_else(|| Error::Execution("Gemma PLE gate is missing".into()))?,
        )?;
        let gate = self.gelu(&gate)?;
        let gated = self.mul(&gate, &per_layer)?;
        let projected = self.linear_at(
            &gated,
            1,
            plan.ple_projection
                .ok_or_else(|| Error::Execution("Gemma PLE projection is missing".into()))?,
        )?;
        let projected = self.rms_norm_at(
            &projected,
            plan.ple_norm
                .ok_or_else(|| Error::Execution("Gemma PLE norm is missing".into()))?,
            config.rms_norm_eps,
        )?;
        self.add(hidden, &projected)
    }

    fn apply_ple_rows(
        &self,
        hidden: &DeviceBuffer<f32>,
        packed_ple: &DeviceBuffer<f32>,
        rows: usize,
        layer: usize,
        config: &Gemma4TextConfig,
    ) -> Result<WorkspaceF32> {
        let dimension = config.hidden_size_per_layer_input;
        let packed = config.num_hidden_layers * dimension;
        let per_layer = self.slice_rows(packed_ple, rows, packed, layer * dimension, dimension)?;
        let plan = self
            .decode_plan
            .get(layer)
            .ok_or_else(|| Error::Execution("Gemma decode layer is missing".into()))?;
        let gate = self.linear_at(
            hidden,
            rows,
            plan.ple_gate
                .ok_or_else(|| Error::Execution("Gemma PLE gate is missing".into()))?,
        )?;
        let gate = self.gelu(&gate)?;
        let gated = self.mul(&gate, &per_layer)?;
        let projected = self.linear_at(
            &gated,
            rows,
            plan.ple_projection
                .ok_or_else(|| Error::Execution("Gemma PLE projection is missing".into()))?,
        )?;
        let projected = self.rms_norm_at(
            &projected,
            plan.ple_norm
                .ok_or_else(|| Error::Execution("Gemma PLE norm is missing".into()))?,
            config.rms_norm_eps,
        )?;
        self.add(hidden, &projected)
    }

    /// Executes tied-embedding LM-head projection for one token.
    ///
    /// This is also a runtime validation of the persistent BF16 store and the
    /// dynamically loaded cuBLAS BF16 GEMM ABI.
    pub fn embedding_logits(&self, token: u32, hidden_size: usize) -> Result<Vec<f32>> {
        let embedding = self.weight("embed_tokens.weight")?;
        if embedding.shape.len() != 2 || embedding.shape[1] != hidden_size {
            return Err(Error::InvalidShape(format!(
                "embed_tokens.weight has unexpected shape {:?}",
                embedding.shape
            )));
        }
        let hidden = self.embedding(token, hidden_size)?;
        let hidden_bf16 = self.to_bf16(&hidden)?;
        let mut logits = self.output_f32(embedding.shape[0])?;
        self.cublas.linear_bf16_f32(
            &self.stream,
            1,
            embedding.shape[0],
            hidden_size,
            &hidden_bf16,
            &embedding.buffer,
            &mut logits,
        )?;
        logits.to_host_vec(&self.stream).map_err(cuda_error)
    }

    /// Runs the complete dense MLP residual branch for one decoder layer.
    pub fn decoder_mlp_smoke(
        &self,
        token: u32,
        layer: usize,
        hidden_size: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>> {
        let hidden = self.embedding(token, hidden_size)?;
        let prefix = format!("layers.{layer}");
        let normalized = self.rms_norm(
            &hidden,
            &format!("{prefix}.pre_feedforward_layernorm.weight"),
            epsilon,
        )?;
        let gate = self.linear(&normalized, 1, &format!("{prefix}.mlp.gate_proj.weight"))?;
        let up = self.linear(&normalized, 1, &format!("{prefix}.mlp.up_proj.weight"))?;
        let activated = self.gelu_mul(&gate, &up)?;
        let down = self.linear(&activated, 1, &format!("{prefix}.mlp.down_proj.weight"))?;
        let down = self.rms_norm(
            &down,
            &format!("{prefix}.post_feedforward_layernorm.weight"),
            epsilon,
        )?;
        let output = self.add(&hidden, &down)?;
        output.to_host_vec(&self.stream).map_err(cuda_error)
    }

    /// Runs a one-token attention residual branch for a decoder layer.
    #[allow(clippy::too_many_arguments)]
    pub fn decoder_attention_smoke(
        &self,
        token: u32,
        layer: usize,
        hidden_size: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        epsilon: f32,
        window: usize,
    ) -> Result<Vec<f32>> {
        let hidden = self.embedding(token, hidden_size)?;
        let prefix = format!("layers.{layer}");
        let normalized = self.rms_norm(
            &hidden,
            &format!("{prefix}.input_layernorm.weight"),
            epsilon,
        )?;
        let (query, key, value) = if self
            .weight_indices
            .contains_key(&format!("{prefix}.self_attn.qkv_proj.weight"))
        {
            let normalized = self.to_bf16(&normalized)?;
            self.linear_qkv_bf16(
                &normalized,
                1,
                &prefix,
                heads * head_dim,
                kv_heads * head_dim,
                kv_heads * head_dim,
            )?
        } else {
            (
                self.linear(&normalized, 1, &format!("{prefix}.self_attn.q_proj.weight"))?,
                self.linear(&normalized, 1, &format!("{prefix}.self_attn.k_proj.weight"))?,
                self.linear(&normalized, 1, &format!("{prefix}.self_attn.v_proj.weight"))?,
            )
        };
        if query.len() != heads * head_dim || key.len() != kv_heads * head_dim {
            return Err(Error::InvalidShape(format!(
                "layer {layer} attention projection shape mismatch"
            )));
        }
        let query = self.rms_norm(
            &query,
            &format!("{prefix}.self_attn.q_norm.weight"),
            epsilon,
        )?;
        let key = self.rms_norm(&key, &format!("{prefix}.self_attn.k_norm.weight"), epsilon)?;
        let value = self.rms_norm_unit(&value, head_dim, epsilon)?;
        let query = self.rope(&query, heads, head_dim, head_dim, 0, 10_000.0, 1.0)?;
        let key = self.rope(&key, kv_heads, head_dim, head_dim, 0, 10_000.0, 1.0)?;
        let attended = self.gqa(
            &query, &key, &value, heads, kv_heads, head_dim, 1, window, 0, 1,
        )?;
        let projected = self.linear(&attended, 1, &format!("{prefix}.self_attn.o_proj.weight"))?;
        let projected = self.rms_norm(
            &projected,
            &format!("{prefix}.post_attention_layernorm.weight"),
            epsilon,
        )?;
        let output = self.add(&hidden, &projected)?;
        output.to_host_vec(&self.stream).map_err(cuda_error)
    }

    /// Appends one token and evaluates attention against the persistent cache.
    #[allow(clippy::too_many_arguments)]
    pub fn cached_attention_smoke(
        &self,
        token: u32,
        layer: usize,
        hidden_size: usize,
        heads: usize,
        epsilon: f32,
        cache: &mut Gemma4CudaKvCache,
    ) -> Result<Vec<f32>> {
        let hidden = self.embedding(token, hidden_size)?;
        let prefix = format!("layers.{layer}");
        let normalized = self.rms_norm(
            &hidden,
            &format!("{prefix}.input_layernorm.weight"),
            epsilon,
        )?;
        let (query, key, value) = if self
            .weight_indices
            .contains_key(&format!("{prefix}.self_attn.qkv_proj.weight"))
        {
            let normalized = self.to_bf16(&normalized)?;
            self.linear_qkv_bf16(
                &normalized,
                1,
                &prefix,
                heads * cache.head_dim,
                cache.kv_heads * cache.head_dim,
                cache.kv_heads * cache.head_dim,
            )?
        } else {
            (
                self.linear(&normalized, 1, &format!("{prefix}.self_attn.q_proj.weight"))?,
                self.linear(&normalized, 1, &format!("{prefix}.self_attn.k_proj.weight"))?,
                self.linear(&normalized, 1, &format!("{prefix}.self_attn.v_proj.weight"))?,
            )
        };
        let query = self.rms_norm(
            &query,
            &format!("{prefix}.self_attn.q_norm.weight"),
            epsilon,
        )?;
        let key = self.rms_norm(&key, &format!("{prefix}.self_attn.k_norm.weight"), epsilon)?;
        let value = self.rms_norm_unit(&value, cache.head_dim, epsilon)?;
        let absolute_position = cache.total_seen;
        let query = self.rope(
            &query,
            heads,
            cache.head_dim,
            cache.head_dim,
            absolute_position,
            10_000.0,
            1.0,
        )?;
        let key = self.rope(
            &key,
            cache.kv_heads,
            cache.head_dim,
            cache.head_dim,
            absolute_position,
            10_000.0,
            1.0,
        )?;
        self.append_kv(cache, &key, &value)?;
        let attended = self.gqa(
            &query,
            &cache.key,
            &cache.value,
            heads,
            cache.kv_heads,
            cache.head_dim,
            cache.len,
            cache.capacity,
            cache.start,
            cache.capacity,
        )?;
        attended.to_host_vec(&self.stream).map_err(cuda_error)
    }

    fn decoder_mlp(
        &self,
        hidden: &DeviceBuffer<f32>,
        layer: usize,
        epsilon: f32,
    ) -> Result<WorkspaceF32> {
        let plan = self
            .decode_plan
            .get(layer)
            .ok_or_else(|| Error::Execution("Gemma decode layer is missing".into()))?;
        let normalized = self.rms_norm_bf16_at(hidden, plan.pre_feedforward_norm, epsilon)?;
        let gate = self.linear_bf16_at(&normalized, 1, plan.gate_projection)?;
        let up = self.linear_bf16_at(&normalized, 1, plan.up_projection)?;
        let activated = self.gelu_mul(&gate, &up)?;
        let down = self.linear_at(&activated, 1, plan.down_projection)?;
        let down = self.rms_norm_at(&down, plan.post_feedforward_norm, epsilon)?;
        self.add(hidden, &down)
    }

    fn decoder_mlp_rows(
        &self,
        hidden: &DeviceBuffer<f32>,
        rows: usize,
        layer: usize,
        epsilon: f32,
    ) -> Result<WorkspaceF32> {
        let plan = self
            .decode_plan
            .get(layer)
            .ok_or_else(|| Error::Execution("Gemma decode layer is missing".into()))?;
        let normalized = self.rms_norm_bf16_at(hidden, plan.pre_feedforward_norm, epsilon)?;
        let gate = self.linear_bf16_at(&normalized, rows, plan.gate_projection)?;
        let up = self.linear_bf16_at(&normalized, rows, plan.up_projection)?;
        let activated = self.gelu_mul(&gate, &up)?;
        let down = self.linear_at(&activated, rows, plan.down_projection)?;
        let down = self.rms_norm_at(&down, plan.post_feedforward_norm, epsilon)?;
        self.add(hidden, &down)
    }

    fn decoder_attention(
        &self,
        hidden: &DeviceBuffer<f32>,
        layer: usize,
        config: &Gemma4TextConfig,
        table: &mut Gemma4CudaCacheTable,
    ) -> Result<WorkspaceF32> {
        let plan = self
            .decode_plan
            .get(layer)
            .ok_or_else(|| Error::Execution("Gemma decode layer is missing".into()))?;
        let trace =
            std::env::var_os("GEMMA4_TRACE_LAYERS").is_some() && table.position == 0 && layer == 0;
        let normalized_f32 = if trace {
            Some(self.rms_norm_at(hidden, plan.input_norm, config.rms_norm_eps)?)
        } else {
            None
        };
        let normalized = if let Some(value) = &normalized_f32 {
            self.to_bf16(value)?
        } else {
            self.rms_norm_bf16_at(hidden, plan.input_norm, config.rms_norm_eps)?
        };
        if trace {
            self.trace_weight_fingerprint(
                "layer.0.input_norm.weight",
                &format!("layers.{layer}.input_layernorm.weight"),
                true,
            )?;
        }
        if let Some(value) = &normalized_f32 {
            self.trace_fingerprint("layer.0.input_norm", value, trace)?;
        }
        let source = plan.source;
        debug_assert_eq!(source, table.sources[layer]);
        let cache_kv_heads = table.layers[source]
            .as_ref()
            .ok_or_else(|| Error::Execution(format!("source cache {source} is missing")))?
            .kv_heads;
        let (query, physical_key, physical_value) = if plan.fused_qkv {
            let (query, key, value) = self.linear_qkv_bf16_at(
                &normalized,
                1,
                plan.query_projection,
                config.num_attention_heads * plan.head_dim,
                cache_kv_heads * plan.head_dim,
                cache_kv_heads * plan.head_dim,
            )?;
            (query, Some(key), Some(value))
        } else {
            (
                self.linear_bf16_at(&normalized, 1, plan.query_projection)?,
                None,
                None,
            )
        };
        let query = self.rms_norm_at(&query, plan.query_norm, config.rms_norm_eps)?;
        self.trace_fingerprint("layer.0.query_norm", &query, trace)?;
        let query = self.rope_state(
            &query,
            config.num_attention_heads,
            plan.head_dim,
            plan.rotary_dim,
            plan.rope_theta,
            plan.rope_factor,
            &table.decode_state,
        )?;
        if plan.fused_qkv {
            let key = physical_key.ok_or_else(|| {
                Error::Execution(format!("layer {layer} fused key projection is missing"))
            })?;
            let value = physical_value.ok_or_else(|| {
                Error::Execution(format!("layer {layer} fused value projection is missing"))
            })?;
            let key_norm = plan.key_norm.ok_or_else(|| {
                Error::Execution(format!(
                    "layer {layer} key norm is missing from decode plan"
                ))
            })?;
            let key = self.rms_norm_at(&key, key_norm, config.rms_norm_eps)?;
            let cache = table.layers[layer]
                .as_mut()
                .ok_or_else(|| Error::Execution(format!("layer {layer} cache is missing")))?;
            let key = self.rope_state(
                &key,
                cache.kv_heads,
                plan.head_dim,
                plan.rotary_dim,
                plan.rope_theta,
                plan.rope_factor,
                &table.decode_state,
            )?;
            let value = self.rms_norm_unit(&value, plan.head_dim, config.rms_norm_eps)?;
            self.trace_fingerprint("layer.0.value_norm", &value, trace)?;
            self.append_kv_state(cache, &key, &value, &table.decode_state)?;
        }
        let cache = table.layers[source]
            .as_ref()
            .ok_or_else(|| Error::Execution(format!("source cache {source} is missing")))?;
        let attended = self.gqa_state(
            &query,
            &cache.key,
            &cache.value,
            config.num_attention_heads,
            cache.kv_heads,
            plan.head_dim,
            if plan.sliding {
                config.sliding_window
            } else {
                0
            },
            cache.capacity,
            cache.capacity <= 4096,
            &table.decode_state,
        )?;
        self.trace_fingerprint("layer.0.attended", &attended, trace)?;
        let projected = self.linear_at(&attended, 1, plan.output_projection)?;
        self.trace_fingerprint("layer.0.attention_raw", &projected, trace)?;
        let projected =
            self.rms_norm_at(&projected, plan.post_attention_norm, config.rms_norm_eps)?;
        if trace {
            self.trace_weight_fingerprint(
                "layer.0.attention_norm.weight",
                &format!("layers.{layer}.post_attention_layernorm.weight"),
                true,
            )?;
        }
        self.trace_fingerprint("layer.0.attention_norm", &projected, trace)?;
        let output = self.add(hidden, &projected)?;
        self.trace_fingerprint("layer.0.attention_residual", &output, trace)?;
        Ok(output)
    }

    fn decoder_attention_rows(
        &self,
        hidden: &DeviceBuffer<f32>,
        rows: usize,
        layer: usize,
        config: &Gemma4TextConfig,
        table: &mut Gemma4CudaCacheTable,
    ) -> Result<WorkspaceF32> {
        let prefix = format!("layers.{layer}");
        let normalized = self.rms_norm_bf16(
            hidden,
            &format!("{prefix}.input_layernorm.weight"),
            config.rms_norm_eps,
        )?;
        let layer_type = &config
            .layer_types
            .as_ref()
            .ok_or_else(|| Error::Execution("Gemma 4 layer_types are missing".into()))?[layer];
        let sliding = layer_type == "sliding_attention";
        let head_dim = if sliding {
            config.head_dim
        } else {
            config.global_head_dim
        };
        let rope = config
            .rope_parameters
            .as_ref()
            .and_then(|parameters| parameters.get(layer_type))
            .ok_or_else(|| Error::Execution(format!("RoPE parameters for {layer_type} missing")))?;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let rotary_dim = (rope.partial_rotary_factor * head_dim as f32) as usize;
        let source = table.sources[layer];
        let cache_kv_heads = table.layers[source]
            .as_ref()
            .ok_or_else(|| Error::Execution(format!("source cache {source} is missing")))?
            .kv_heads;
        let (query, physical_key, physical_value) = if source == layer {
            let (query, key, value) = self.linear_qkv_bf16(
                &normalized,
                rows,
                &prefix,
                config.num_attention_heads * head_dim,
                cache_kv_heads * head_dim,
                cache_kv_heads * head_dim,
            )?;
            (query, Some(key), Some(value))
        } else {
            (
                self.linear_bf16(
                    &normalized,
                    rows,
                    &format!("{prefix}.self_attn.q_proj.weight"),
                )?,
                None,
                None,
            )
        };
        let query = self.rms_norm(
            &query,
            &format!("{prefix}.self_attn.q_norm.weight"),
            config.rms_norm_eps,
        )?;
        let query = self.rope(
            &query,
            config.num_attention_heads,
            head_dim,
            rotary_dim,
            0,
            rope.rope_theta,
            rope.factor,
        )?;
        if source == layer {
            let key = physical_key.ok_or_else(|| {
                Error::Execution(format!("layer {layer} fused key projection is missing"))
            })?;
            let value = physical_value.ok_or_else(|| {
                Error::Execution(format!("layer {layer} fused value projection is missing"))
            })?;
            let key = self.rms_norm(
                &key,
                &format!("{prefix}.self_attn.k_norm.weight"),
                config.rms_norm_eps,
            )?;
            let cache = table.layers[layer]
                .as_mut()
                .ok_or_else(|| Error::Execution(format!("layer {layer} cache is missing")))?;
            let key = self.rope(
                &key,
                cache.kv_heads,
                head_dim,
                rotary_dim,
                0,
                rope.rope_theta,
                rope.factor,
            )?;
            let value = self.rms_norm_unit(&value, head_dim, config.rms_norm_eps)?;
            let width = cache.kv_heads * head_dim;
            if cache.len != 0 || cache.start != 0 || rows > cache.capacity {
                return Err(Error::Execution(
                    "batched prefill requires an empty contiguous KV cache".into(),
                ));
            }
            unsafe {
                self.module.gemma_cache_write(
                    &self.stream,
                    Self::launch_config(rows * width)?,
                    0,
                    &key,
                    &mut cache.key,
                )
            }
            .map_err(cuda_error)?;
            unsafe {
                self.module.gemma_cache_write(
                    &self.stream,
                    Self::launch_config(rows * width)?,
                    0,
                    &value,
                    &mut cache.value,
                )
            }
            .map_err(cuda_error)?;
            cache.len = rows;
            cache.total_seen = rows;
        }
        let cache = table.layers[source]
            .as_ref()
            .ok_or_else(|| Error::Execution(format!("source cache {source} is missing")))?;
        let attended = self.gqa_prefill(
            &query,
            &cache.key,
            &cache.value,
            rows,
            config.num_attention_heads,
            cache.kv_heads,
            head_dim,
            if sliding { config.sliding_window } else { 0 },
        )?;
        let projected = self.linear(
            &attended,
            rows,
            &format!("{prefix}.self_attn.o_proj.weight"),
        )?;
        let projected = self.rms_norm(
            &projected,
            &format!("{prefix}.post_attention_layernorm.weight"),
            config.rms_norm_eps,
        )?;
        self.add(hidden, &projected)
    }

    fn forward_token(
        &self,
        token: u32,
        config: &Gemma4TextConfig,
        table: &mut Gemma4CudaCacheTable,
    ) -> Result<WorkspaceF32> {
        if token as usize >= config.vocab_size {
            return Err(Error::Execution(
                "token id exceeds CUDA embedding vocabulary".into(),
            ));
        }
        self.set_decode_state(token, table)?;
        self.forward_token_state(config, table, true)
    }

    fn forward_token_state(
        &self,
        config: &Gemma4TextConfig,
        table: &mut Gemma4CudaCacheTable,
        advance_host_state: bool,
    ) -> Result<WorkspaceF32> {
        let embedding = self.embedding_state(&table.decode_state, config.hidden_size)?;
        let packed_ple = self.packed_ple_state(&table.decode_state, &embedding, config)?;
        let mut hidden = embedding;
        let trace = std::env::var_os("GEMMA4_TRACE_LAYERS").is_some() && table.position == 0;
        self.trace_fingerprint("embedding", &hidden, trace)?;
        for layer in 0..config.num_hidden_layers {
            hidden = self.decoder_attention(&hidden, layer, config, table)?;
            self.trace_fingerprint(&format!("layer.{layer}.attention"), &hidden, trace)?;
            hidden = self.decoder_mlp(&hidden, layer, config.rms_norm_eps)?;
            self.trace_fingerprint(&format!("layer.{layer}.mlp"), &hidden, trace)?;
            if let Some(ple) = &packed_ple {
                hidden = self.apply_ple(&hidden, ple, layer, config)?;
                self.trace_fingerprint(&format!("layer.{layer}.ple"), &hidden, trace)?;
            }
            let scalar = self
                .decode_plan
                .get(layer)
                .ok_or_else(|| Error::Execution("Gemma decode layer is missing".into()))?
                .layer_scalar;
            hidden = self.scale_by_weight_at(&hidden, scalar)?;
            self.trace_fingerprint(&format!("layer.{layer}.scaled"), &hidden, trace)?;
            self.trace_fingerprint(&format!("layer.{layer}"), &hidden, trace)?;
        }
        if advance_host_state {
            table.position += 1;
            for cache in table.layers.iter_mut().flatten() {
                cache.total_seen += 1;
                cache.len = cache.total_seen.min(cache.capacity);
                cache.start = if cache.total_seen > cache.capacity {
                    cache.total_seen % cache.capacity
                } else {
                    0
                };
            }
        }
        Ok(hidden)
    }

    fn forward_prompt_rows(
        &self,
        tokens: &[u32],
        config: &Gemma4TextConfig,
        table: &mut Gemma4CudaCacheTable,
    ) -> Result<WorkspaceF32> {
        if tokens.is_empty() || tokens.len() > 4096 || table.position != 0 {
            return Err(Error::Execution(
                "batched prefill requires 1..=4096 tokens and an empty cache table".into(),
            ));
        }
        if table
            .layers
            .iter()
            .flatten()
            .any(|cache| cache.capacity < tokens.len() || !cache.is_empty())
        {
            return Err(Error::Execution(
                "prompt does not fit the contiguous batched-prefill cache".into(),
            ));
        }
        let rows = tokens.len();
        let embedding = self.embedding_rows(tokens, config.hidden_size)?;
        let packed_ple = self.packed_ple_rows(tokens, &embedding, config)?;
        let mut hidden = embedding;
        for layer in 0..config.num_hidden_layers {
            hidden = self.decoder_attention_rows(&hidden, rows, layer, config, table)?;
            hidden = self.decoder_mlp_rows(&hidden, rows, layer, config.rms_norm_eps)?;
            if let Some(ple) = &packed_ple {
                hidden = self.apply_ple_rows(&hidden, ple, rows, layer, config)?;
            }
            let scalar = self
                .decode_plan
                .get(layer)
                .ok_or_else(|| Error::Execution("Gemma decode layer is missing".into()))?
                .layer_scalar;
            hidden = self.scale_by_weight_at(&hidden, scalar)?;
        }
        table.position = rows;
        Ok(hidden)
    }

    /// Prefills a complete prompt with matrix-shaped projections and returns
    /// the final-token logits. The empty-cache restriction keeps causal KV
    /// layout simple and preserves the one-token path as a safe fallback.
    pub fn prefill_prompt(
        &self,
        tokens: &[u32],
        config: &Gemma4TextConfig,
        table: &mut Gemma4CudaCacheTable,
    ) -> Result<Vec<f32>> {
        let hidden = self.forward_prompt_rows(tokens, config, table)?;
        let last = self.slice(
            &hidden,
            (tokens.len() - 1) * config.hidden_size,
            config.hidden_size,
        )?;
        let hidden = self.rms_norm(&last, "norm.weight", config.rms_norm_eps)?;
        let embedding = self.weight("embed_tokens.weight")?;
        let hidden_bf16 = self.to_bf16(&hidden)?;
        let mut logits = self.output_f32(config.vocab_size)?;
        self.cublas.linear_bf16_f32(
            &self.stream,
            1,
            config.vocab_size,
            config.hidden_size,
            &hidden_bf16,
            &embedding.buffer,
            &mut logits,
        )?;
        let mut logits = logits.to_host_vec(&self.stream).map_err(cuda_error)?;
        apply_logit_softcap(&mut logits, config.final_logit_softcapping);
        Ok(logits)
    }

    /// Prefills one token into the persistent KV cache without evaluating the
    /// vocabulary-sized LM head. Intermediate prompt tokens do not need
    /// logits, so this avoids a 262,144-way GEMM and D2H copy per token.
    pub fn prefill_token(
        &self,
        token: u32,
        config: &Gemma4TextConfig,
        table: &mut Gemma4CudaCacheTable,
    ) -> Result<()> {
        let _hidden = self.forward_token(token, config, table)?;
        Ok(())
    }

    /// Runs one autoregressive token through all decoder layers and evaluates
    /// the tied LM head. Returned logits are copied to the host once.
    pub fn decode_token(
        &self,
        token: u32,
        config: &Gemma4TextConfig,
        table: &mut Gemma4CudaCacheTable,
    ) -> Result<Vec<f32>> {
        let logits = self.decode_logits(token, config, table)?;
        let mut logits = logits.to_host_vec(&self.stream).map_err(cuda_error)?;
        apply_logit_softcap(&mut logits, config.final_logit_softcapping);
        Ok(logits)
    }

    fn decode_logits(
        &self,
        token: u32,
        config: &Gemma4TextConfig,
        table: &mut Gemma4CudaCacheTable,
    ) -> Result<WorkspaceF32> {
        let hidden = self.forward_token(token, config, table)?;
        let hidden = self.rms_norm(&hidden, "norm.weight", config.rms_norm_eps)?;
        let embedding = self.weight("embed_tokens.weight")?;
        let hidden_bf16 = self.to_bf16(&hidden)?;
        let mut logits = self.output_f32(config.vocab_size)?;
        self.cublas.linear_bf16_f32(
            &self.stream,
            1,
            config.vocab_size,
            config.hidden_size,
            &hidden_bf16,
            &embedding.buffer,
            &mut logits,
        )?;
        Ok(logits)
    }

    fn decode_logits_state(
        &self,
        config: &Gemma4TextConfig,
        table: &mut Gemma4CudaCacheTable,
    ) -> Result<WorkspaceF32> {
        let hidden = self.forward_token_state(config, table, false)?;
        let hidden = self.rms_norm(&hidden, "norm.weight", config.rms_norm_eps)?;
        let embedding = self.weight("embed_tokens.weight")?;
        let hidden_bf16 = self.to_bf16(&hidden)?;
        let mut logits = self.output_f32(config.vocab_size)?;
        self.cublas.linear_bf16_f32(
            &self.stream,
            1,
            config.vocab_size,
            config.hidden_size,
            &hidden_bf16,
            &embedding.buffer,
            &mut logits,
        )?;
        Ok(logits)
    }

    fn build_decode_graph(
        &self,
        config: &Gemma4TextConfig,
        table: &mut Gemma4CudaCacheTable,
        top_k: usize,
        repetition_penalty: f32,
        seen: &DeviceBuffer<u8>,
    ) -> Result<Gemma4CudaDecodeGraph> {
        if table
            .layers
            .iter()
            .flatten()
            .any(|cache| cache.capacity > 4096)
        {
            return Err(Error::Execution(
                "CUDA Graph decode currently requires KV capacities <= 4096".into(),
            ));
        }
        let blocks = config.vocab_size.div_ceil(256);
        let mut scores = self.output_f32(top_k)?.into_inner();
        let mut ids = self.output_f32(top_k)?.into_inner();
        let executable = CudaGraphExec::capture(&self.stream, || {
            let logits = self.decode_logits_state(config, table)?;
            let candidates = blocks * top_k;
            let mut stage_scores = self.output_f32(candidates)?;
            let mut stage_ids = self.output_f32(candidates)?;
            unsafe {
                self.module.gemma_topk_stage1(
                    &self.stream,
                    LaunchConfig {
                        grid_dim: (blocks as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    top_k,
                    repetition_penalty,
                    &logits,
                    seen,
                    &mut stage_scores,
                    &mut stage_ids,
                )
            }
            .map_err(cuda_error)?;
            unsafe {
                self.module.gemma_topk_stage2(
                    &self.stream,
                    Self::launch_config(1)?,
                    top_k,
                    &stage_scores,
                    &stage_ids,
                    &mut scores,
                    &mut ids,
                )
            }
            .map_err(cuda_error)
        })?;

        // Every temporary pointer referenced by the graph is now back in its
        // pool. Drain the pools so regular workspace allocation cannot reuse
        // or overwrite graph-owned storage between replays.
        let fixed_f32 = self
            .f32_pool
            .borrow_mut()
            .values_mut()
            .flat_map(|buffers| buffers.drain(..))
            .collect();
        let fixed_bf16 = self
            .bf16_pool
            .borrow_mut()
            .values_mut()
            .flat_map(|buffers| buffers.drain(..))
            .collect();
        Ok(Gemma4CudaDecodeGraph {
            executable,
            scores,
            ids,
            _fixed_f32: fixed_f32,
            _fixed_bf16: fixed_bf16,
            top_k,
            repetition_penalty,
        })
    }

    fn replay_decode_graph(
        &self,
        token: u32,
        config: &Gemma4TextConfig,
        table: &mut Gemma4CudaCacheTable,
        mark_seen: bool,
        seen: &mut DeviceBuffer<u8>,
    ) -> Result<Vec<(u32, f32)>> {
        let graph = table
            .decode_graph
            .as_ref()
            .ok_or_else(|| Error::Execution("CUDA decode graph is missing".into()))?;
        unsafe {
            self.module.gemma_decode_state_update(
                &self.stream,
                Self::launch_config(1)?,
                token as usize,
                table.position,
                mark_seen,
                &mut table.decode_state,
                seen,
            )
        }
        .map_err(cuda_error)?;
        graph.executable.launch(&self.stream)?;
        let mut scores = graph.scores.to_host_vec(&self.stream).map_err(cuda_error)?;
        let ids = graph.ids.to_host_vec(&self.stream).map_err(cuda_error)?;
        apply_logit_softcap(&mut scores, config.final_logit_softcapping);
        table.position += 1;
        for cache in table.layers.iter_mut().flatten() {
            cache.total_seen += 1;
            cache.len = cache.total_seen.min(cache.capacity);
            cache.start = if cache.total_seen > cache.capacity {
                cache.total_seen % cache.capacity
            } else {
                0
            };
        }
        Ok(ids
            .into_iter()
            .zip(scores)
            .map(|(id, score)| (id as u32, score))
            .collect())
    }

    /// Runs decode and transfers only the global top-k candidates to the
    /// host. Repetition penalty is applied before selection using a persistent
    /// device-side seen-token bitmap.
    pub fn decode_topk(
        &self,
        token: u32,
        config: &Gemma4TextConfig,
        table: &mut Gemma4CudaCacheTable,
        top_k: usize,
        repetition_penalty: f32,
        mark_seen: bool,
    ) -> Result<Vec<(u32, f32)>> {
        if token as usize >= config.vocab_size {
            return Err(Error::Execution(
                "token id exceeds CUDA decode vocabulary".into(),
            ));
        }
        let top_k = top_k.clamp(1, 64).min(config.vocab_size);
        if self.seen_tokens.borrow().is_none() {
            *self.seen_tokens.borrow_mut() =
                Some(DeviceBuffer::zeroed(&self.stream, config.vocab_size).map_err(cuda_error)?);
        }
        let mut seen = self.seen_tokens.borrow_mut();
        let seen = seen
            .as_mut()
            .ok_or_else(|| Error::Execution("CUDA seen-token state is missing".into()))?;
        let graph_compatible = table.decode_graph.as_ref().is_some_and(|graph| {
            graph.top_k == top_k
                && graph.repetition_penalty.to_bits() == repetition_penalty.to_bits()
        });
        if graph_compatible {
            return self.replay_decode_graph(token, config, table, mark_seen, seen);
        }
        if table.decode_graph.is_some() {
            table.decode_graph = None;
            table.decode_graph_attempted = false;
        }
        if mark_seen {
            unsafe {
                self.module.gemma_mark_seen(
                    &self.stream,
                    Self::launch_config(1)?,
                    token as usize,
                    seen,
                )
            }
            .map_err(cuda_error)?;
        }
        let logits = self.decode_logits(token, config, table)?;
        let blocks = config.vocab_size.div_ceil(256);
        let candidates = blocks * top_k;
        let mut stage_scores = self.output_f32(candidates)?;
        let mut stage_ids = self.output_f32(candidates)?;
        unsafe {
            self.module.gemma_topk_stage1(
                &self.stream,
                LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                top_k,
                repetition_penalty,
                &logits,
                seen,
                &mut stage_scores,
                &mut stage_ids,
            )
        }
        .map_err(cuda_error)?;
        let mut final_scores = self.output_f32(top_k)?;
        let mut final_ids = self.output_f32(top_k)?;
        unsafe {
            self.module.gemma_topk_stage2(
                &self.stream,
                Self::launch_config(1)?,
                top_k,
                &stage_scores,
                &stage_ids,
                &mut final_scores,
                &mut final_ids,
            )
        }
        .map_err(cuda_error)?;
        let mut scores = final_scores.to_host_vec(&self.stream).map_err(cuda_error)?;
        let ids = final_ids.to_host_vec(&self.stream).map_err(cuda_error)?;
        apply_logit_softcap(&mut scores, config.final_logit_softcapping);
        let candidates: Vec<_> = ids
            .into_iter()
            .zip(scores)
            .map(|(id, score)| (id as u32, score))
            .collect();
        // Return the complete warm-up allocation set before capture. This
        // prevents `cuMemAllocAsync` graph nodes and gives every captured
        // operation a stable, already allocated pointer.
        drop(final_ids);
        drop(final_scores);
        drop(stage_ids);
        drop(stage_scores);
        drop(logits);
        if std::env::var_os("GEMMA4_DISABLE_CUDA_GRAPH").is_none()
            && table.decode_graph.is_none()
            && !table.decode_graph_attempted
            && table
                .layers
                .iter()
                .flatten()
                .all(|cache| cache.capacity <= 4096)
        {
            table.decode_graph_attempted = true;
            match self.build_decode_graph(config, table, top_k, repetition_penalty, seen) {
                Ok(graph) => {
                    table.decode_graph = Some(graph);
                    if std::env::var_os("GEMMA4_PROFILE").is_some() {
                        eprintln!("Gemma4 CUDA Graph: captured decode+top-k");
                    }
                }
                Err(error) => {
                    eprintln!("Gemma4 CUDA Graph disabled: {error}");
                }
            }
        }
        Ok(candidates)
    }
}

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
