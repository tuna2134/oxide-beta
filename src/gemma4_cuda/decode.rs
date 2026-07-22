use super::*;

impl Gemma4CudaState {
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
