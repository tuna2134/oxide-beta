use super::*;

impl Gemma4CudaState {
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
}
