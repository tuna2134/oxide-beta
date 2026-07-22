use super::*;

impl Gemma4CudaState {
    pub(super) fn embedding(&self, token: u32, hidden_size: usize) -> Result<WorkspaceF32> {
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
            self.inference.bf16_to_f32_scaled(
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

    pub(super) fn set_decode_state(
        &self,
        token: u32,
        table: &mut Gemma4CudaCacheTable,
    ) -> Result<()> {
        unsafe {
            self.inference.decode_state_set(
                &self.stream,
                Self::launch_config(1)?,
                token as usize,
                table.position,
                &mut table.decode_state,
            )
        }
        .map_err(cuda_error)
    }

    pub(super) fn embedding_state(
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
            self.inference.bf16_row_scaled_state(
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

    pub(super) fn embedding_rows(
        &self,
        tokens: &[u32],
        hidden_size: usize,
    ) -> Result<WorkspaceF32> {
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
            self.inference.embedding_rows_bf16(
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

    pub(super) fn to_bf16(&self, input: &DeviceBuffer<f32>) -> Result<WorkspaceBf16> {
        let mut output = self.output_bf16(input.len())?;
        // SAFETY: input/output have equal lengths and are disjoint.
        unsafe {
            self.inference.f32_to_bf16(
                &self.stream,
                Self::launch_config(input.len())?,
                input,
                &mut output,
            )
        }
        .map_err(cuda_error)?;
        Ok(output)
    }

    pub(super) fn rms_norm(
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
            self.inference.rms_norm(
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

    pub(super) fn rms_norm_at(
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
            self.inference.rms_norm(
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

    pub(super) fn rms_norm_bf16(
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
            self.inference.rms_norm_bf16(
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

    pub(super) fn rms_norm_bf16_at(
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
            self.inference.rms_norm_bf16(
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

    pub(super) fn rms_norm_unit(
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
            self.inference.rms_norm_unit(
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

    pub(super) fn rope(
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
            self.attention.rope(
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
    pub(super) fn rope_state(
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
            self.attention.rope_state(
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
    pub(super) fn gqa(
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
                self.attention.gqa_decode_block(
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
                self.attention.gqa_decode(
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
    pub(super) fn gqa_state(
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
                self.attention.gqa_decode_block_state(
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
                self.attention.gqa_decode_state(
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
    pub(super) fn gqa_prefill(
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
            self.attention.gqa_prefill_block(
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
}
