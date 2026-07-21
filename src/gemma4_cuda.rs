//! Gemma 4 persistent BF16 CUDA state.
#![allow(unsafe_code)]

use crate::cublas::Cublas;
use crate::models::gemma4::{Gemma4ForCausalLM, Gemma4TextConfig};
use crate::{Error, Result};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use safetensors::Dtype;
use std::collections::HashMap;
use std::sync::Arc;

/// Persistent CUDA resources for Gemma 4 inference.
///
/// All BF16 language-model weights are uploaded once during construction and
/// remain resident until this value is dropped.
struct CudaWeight {
    shape: Vec<usize>,
    buffer: DeviceBuffer<u16>,
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
    weights: HashMap<String, CudaWeight>,
}

impl Gemma4CudaState {
    pub(crate) fn load(model: &Gemma4ForCausalLM, device: usize) -> Result<Self> {
        let context = CudaContext::new(device).map_err(cuda_error)?;
        let stream = context.default_stream();
        let cublas = Cublas::new()?;
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
            weights.insert(name, weight);
        }
        stream.synchronize().map_err(cuda_error)?;
        Ok(Self {
            _context: context,
            stream,
            cublas,
            module,
            weights,
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
            .values()
            .map(|weight| weight.buffer.len() * std::mem::size_of::<u16>())
            .sum()
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
            .find_map(|name| self.weights.get(name))
            .ok_or_else(|| Error::Execution(format!("CUDA weight `{suffix}` is missing")))
    }

    fn output_f32(&self, len: usize) -> Result<DeviceBuffer<f32>> {
        // SAFETY: every caller launches a kernel/cuBLAS operation that writes
        // all elements on `self.stream` before any read or synchronization.
        unsafe { DeviceBuffer::uninitialized_async(&self.stream, len) }.map_err(cuda_error)
    }

    fn output_bf16(&self, len: usize) -> Result<DeviceBuffer<u16>> {
        // SAFETY: same invariant as `output_f32`.
        unsafe { DeviceBuffer::uninitialized_async(&self.stream, len) }.map_err(cuda_error)
    }

    fn launch_config(len: usize) -> Result<LaunchConfig> {
        if len == 0 {
            return Err(Error::InvalidShape("zero-sized Gemma CUDA launch".into()));
        }
        Ok(LaunchConfig::for_num_elems(len as u32))
    }

    fn embedding(&self, token: u32, hidden_size: usize) -> Result<DeviceBuffer<f32>> {
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

    fn to_bf16(&self, input: &DeviceBuffer<f32>) -> Result<DeviceBuffer<u16>> {
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
    ) -> Result<DeviceBuffer<f32>> {
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
                Self::launch_config(input.len())?,
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
    ) -> Result<DeviceBuffer<f32>> {
        if hidden == 0 || input.len() % hidden != 0 {
            return Err(Error::InvalidShape("unit RMSNorm shape mismatch".into()));
        }
        let mut output = self.output_f32(input.len())?;
        // SAFETY: input consists of complete rows of width `hidden` and
        // output has the same extent.
        unsafe {
            self.module.gemma_rms_norm_unit(
                &self.stream,
                Self::launch_config(input.len())?,
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
        position: usize,
        theta: f32,
    ) -> Result<DeviceBuffer<f32>> {
        if heads == 0 || input.len() != heads * head_dim {
            return Err(Error::InvalidShape("RoPE shape mismatch".into()));
        }
        let mut output = self.output_f32(input.len())?;
        // SAFETY: the input is one complete `[heads, head_dim]` token and
        // rotary_dim equals head_dim. Output has an identical extent.
        unsafe {
            self.module.gemma_rope(
                &self.stream,
                Self::launch_config(input.len())?,
                heads,
                head_dim,
                head_dim,
                position,
                theta,
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
    ) -> Result<DeviceBuffer<f32>> {
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
        unsafe {
            self.module.gemma_gqa_decode(
                &self.stream,
                Self::launch_config(query.len())?,
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

    fn linear(
        &self,
        input: &DeviceBuffer<f32>,
        rows: usize,
        weight_name: &str,
    ) -> Result<DeviceBuffer<f32>> {
        let weight = self.weight(weight_name)?;
        if weight.shape.len() != 2 {
            return Err(Error::InvalidShape(format!(
                "{weight_name} is not a matrix"
            )));
        }
        let (output_width, input_width) = (weight.shape[0], weight.shape[1]);
        if input.len() != rows * input_width {
            return Err(Error::InvalidShape(format!(
                "{weight_name} input has {} elements, expected {}",
                input.len(),
                rows * input_width
            )));
        }
        let input_bf16 = self.to_bf16(input)?;
        let mut output = self.output_f32(rows * output_width)?;
        self.cublas.linear_bf16_f32(
            &self.stream,
            rows,
            output_width,
            input_width,
            &input_bf16,
            &weight.buffer,
            &mut output,
        )?;
        Ok(output)
    }

    fn gelu_mul(
        &self,
        gate: &DeviceBuffer<f32>,
        up: &DeviceBuffer<f32>,
    ) -> Result<DeviceBuffer<f32>> {
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

    fn add(
        &self,
        left: &DeviceBuffer<f32>,
        right: &DeviceBuffer<f32>,
    ) -> Result<DeviceBuffer<f32>> {
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

    fn mul(
        &self,
        left: &DeviceBuffer<f32>,
        right: &DeviceBuffer<f32>,
    ) -> Result<DeviceBuffer<f32>> {
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

    fn scale(&self, input: &DeviceBuffer<f32>, scale: f32) -> Result<DeviceBuffer<f32>> {
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

    fn slice(
        &self,
        input: &DeviceBuffer<f32>,
        offset: usize,
        len: usize,
    ) -> Result<DeviceBuffer<f32>> {
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

    fn gelu(&self, input: &DeviceBuffer<f32>) -> Result<DeviceBuffer<f32>> {
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
    ) -> Result<Option<DeviceBuffer<f32>>> {
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

    fn apply_ple(
        &self,
        hidden: &DeviceBuffer<f32>,
        packed_ple: &DeviceBuffer<f32>,
        layer: usize,
        config: &Gemma4TextConfig,
    ) -> Result<DeviceBuffer<f32>> {
        let dimension = config.hidden_size_per_layer_input;
        let per_layer = self.slice(packed_ple, layer * dimension, dimension)?;
        let prefix = format!("layers.{layer}");
        let gate = self.linear(hidden, 1, &format!("{prefix}.per_layer_input_gate.weight"))?;
        let gate = self.gelu(&gate)?;
        let gated = self.mul(&gate, &per_layer)?;
        let projected = self.linear(&gated, 1, &format!("{prefix}.per_layer_projection.weight"))?;
        let projected = self.rms_norm(
            &projected,
            &format!("{prefix}.post_per_layer_input_norm.weight"),
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
        let query = self.linear(&normalized, 1, &format!("{prefix}.self_attn.q_proj.weight"))?;
        let key = self.linear(&normalized, 1, &format!("{prefix}.self_attn.k_proj.weight"))?;
        let value = self.linear(&normalized, 1, &format!("{prefix}.self_attn.v_proj.weight"))?;
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
        let query = self.rope(&query, heads, head_dim, 0, 10_000.0)?;
        let key = self.rope(&key, kv_heads, head_dim, 0, 10_000.0)?;
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
        let query = self.linear(&normalized, 1, &format!("{prefix}.self_attn.q_proj.weight"))?;
        let key = self.linear(&normalized, 1, &format!("{prefix}.self_attn.k_proj.weight"))?;
        let value = self.linear(&normalized, 1, &format!("{prefix}.self_attn.v_proj.weight"))?;
        let query = self.rms_norm(
            &query,
            &format!("{prefix}.self_attn.q_norm.weight"),
            epsilon,
        )?;
        let key = self.rms_norm(&key, &format!("{prefix}.self_attn.k_norm.weight"), epsilon)?;
        let value = self.rms_norm_unit(&value, cache.head_dim, epsilon)?;
        let absolute_position = cache.total_seen;
        let query = self.rope(&query, heads, cache.head_dim, absolute_position, 10_000.0)?;
        let key = self.rope(
            &key,
            cache.kv_heads,
            cache.head_dim,
            absolute_position,
            10_000.0,
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
    ) -> Result<DeviceBuffer<f32>> {
        let prefix = format!("layers.{layer}");
        let normalized = self.rms_norm(
            hidden,
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
        self.add(hidden, &down)
    }

    fn decoder_attention(
        &self,
        hidden: &DeviceBuffer<f32>,
        layer: usize,
        config: &Gemma4TextConfig,
        table: &mut Gemma4CudaCacheTable,
    ) -> Result<DeviceBuffer<f32>> {
        let prefix = format!("layers.{layer}");
        let normalized = self.rms_norm(
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
        let query = self.linear(&normalized, 1, &format!("{prefix}.self_attn.q_proj.weight"))?;
        let query = self.rms_norm(
            &query,
            &format!("{prefix}.self_attn.q_norm.weight"),
            config.rms_norm_eps,
        )?;
        let query = self.rope(
            &query,
            config.num_attention_heads,
            head_dim,
            table.position,
            10_000.0,
        )?;
        let source = table.sources[layer];
        if source == layer {
            let key = self.linear(&normalized, 1, &format!("{prefix}.self_attn.k_proj.weight"))?;
            let value =
                self.linear(&normalized, 1, &format!("{prefix}.self_attn.v_proj.weight"))?;
            let key = self.rms_norm(
                &key,
                &format!("{prefix}.self_attn.k_norm.weight"),
                config.rms_norm_eps,
            )?;
            let cache = table.layers[layer]
                .as_mut()
                .ok_or_else(|| Error::Execution(format!("layer {layer} cache is missing")))?;
            let key = self.rope(&key, cache.kv_heads, head_dim, table.position, 10_000.0)?;
            let value = self.rms_norm_unit(&value, head_dim, config.rms_norm_eps)?;
            self.append_kv(cache, &key, &value)?;
        }
        let cache = table.layers[source]
            .as_ref()
            .ok_or_else(|| Error::Execution(format!("source cache {source} is missing")))?;
        let attended = self.gqa(
            &query,
            &cache.key,
            &cache.value,
            config.num_attention_heads,
            cache.kv_heads,
            head_dim,
            cache.len,
            if sliding { config.sliding_window } else { 0 },
            cache.start,
            cache.capacity,
        )?;
        let projected = self.linear(&attended, 1, &format!("{prefix}.self_attn.o_proj.weight"))?;
        let projected = self.rms_norm(
            &projected,
            &format!("{prefix}.post_attention_layernorm.weight"),
            config.rms_norm_eps,
        )?;
        self.add(hidden, &projected)
    }

    /// Runs one autoregressive token through all 35 decoder layers.
    ///
    /// The supplied cache table is updated in place and reused by subsequent
    /// calls. Returned logits are copied once after the tied LM head.
    pub fn decode_token(
        &self,
        token: u32,
        config: &Gemma4TextConfig,
        table: &mut Gemma4CudaCacheTable,
    ) -> Result<Vec<f32>> {
        let embedding = self.embedding(token, config.hidden_size)?;
        let packed_ple = self.packed_ple(token, &embedding, config)?;
        let mut hidden = embedding;
        for layer in 0..config.num_hidden_layers {
            hidden = self.decoder_attention(&hidden, layer, config, table)?;
            hidden = self.decoder_mlp(&hidden, layer, config.rms_norm_eps)?;
            if let Some(ple) = &packed_ple {
                hidden = self.apply_ple(&hidden, ple, layer, config)?;
            }
        }
        hidden = self.rms_norm(&hidden, "norm.weight", config.rms_norm_eps)?;
        table.position += 1;
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
        if let Some(cap) = config.final_logit_softcapping {
            for logit in &mut logits {
                *logit = (*logit / cap).tanh() * cap;
            }
        }
        Ok(logits)
    }
}

fn cuda_error(error: impl std::fmt::Display) -> Error {
    Error::Execution(format!("Gemma 4 CUDA error: {error}"))
}
