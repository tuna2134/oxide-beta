use super::*;

impl Gemma4CudaState {
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

    pub(super) fn append_kv(
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
            self.inference.cache_write_f32(
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
            self.inference.cache_write_f32(
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

    pub(super) fn append_kv_state(
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
            self.inference.cache_write_f32_state(
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
            self.inference.cache_write_f32_state(
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
}
