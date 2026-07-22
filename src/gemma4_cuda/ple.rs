use super::*;

impl Gemma4CudaState {
    pub(super) fn packed_ple(
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

    pub(super) fn packed_ple_state(
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

    pub(super) fn packed_ple_rows(
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

    pub(super) fn apply_ple(
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
}

pub(super) fn apply_ple_rows(
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
