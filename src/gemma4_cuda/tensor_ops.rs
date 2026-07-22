use super::*;

impl Gemma4CudaState {
    pub(super) fn linear(
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

    pub(super) fn linear_bf16(
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
}

pub(super) fn linear_bf16_at(
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

pub(super) fn linear_at(
    &self,
    input: &DeviceBuffer<f32>,
    rows: usize,
    weight_index: usize,
) -> Result<WorkspaceF32> {
    let input = self.to_bf16(input)?;
    self.linear_bf16_at(&input, rows, weight_index)
}

pub(super) fn linear_qkv_bf16(
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

pub(super) fn linear_qkv_bf16_at(
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

pub(super) fn gelu_mul(
    &self,
    gate: &DeviceBuffer<f32>,
    up: &DeviceBuffer<f32>,
) -> Result<WorkspaceF32> {
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

pub(super) fn add(
    &self,
    left: &DeviceBuffer<f32>,
    right: &DeviceBuffer<f32>,
) -> Result<WorkspaceF32> {
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

pub(super) fn mul(
    &self,
    left: &DeviceBuffer<f32>,
    right: &DeviceBuffer<f32>,
) -> Result<WorkspaceF32> {
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

pub(super) fn scale(&self, input: &DeviceBuffer<f32>, scale: f32) -> Result<WorkspaceF32> {
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

pub(super) fn scale_by_weight_at(
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

pub(super) fn slice(
    &self,
    input: &DeviceBuffer<f32>,
    offset: usize,
    len: usize,
) -> Result<WorkspaceF32> {
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

pub(super) fn slice_rows(
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

pub(super) fn gelu(&self, input: &DeviceBuffer<f32>) -> Result<WorkspaceF32> {
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
