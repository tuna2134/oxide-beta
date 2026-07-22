use oxide_torch::{Device, Error, Result, Tensor};

/// Tensor inputs accepted by BERT modules.
#[derive(Clone, Debug)]
pub struct BertInput {
    /// Token ids shaped `[batch, sequence]`.
    pub input_ids: Tensor,
    /// Optional attention mask shaped `[batch, sequence]`.
    pub attention_mask: Option<Tensor>,
    /// Optional token-type ids shaped `[batch, sequence]`.
    pub token_type_ids: Option<Tensor>,
}

impl BertInput {
    /// Moves every present input tensor to `device`.
    #[must_use]
    pub fn to(&self, device: Device) -> Self {
        Self {
            input_ids: self.input_ids.to(device),
            attention_mask: self.attention_mask.as_ref().map(|value| value.to(device)),
            token_type_ids: self.token_type_ids.as_ref().map(|value| value.to(device)),
        }
    }

    /// Builds tensor inputs from rectangular token batches.
    ///
    /// # Errors
    ///
    /// Returns an error when a batch is empty, ragged, or has a shape that
    /// differs from `input_ids`.
    #[allow(clippy::cast_precision_loss)]
    pub fn from_ids(
        input_ids: &[Vec<u32>],
        attention_mask: Option<&[Vec<u8>]>,
        token_type_ids: Option<&[Vec<u32>]>,
    ) -> Result<Self> {
        let (batch, sequence) = rectangle(input_ids, "input_ids")?;
        let input_ids = Tensor::from_vec(
            input_ids.iter().flatten().map(|&id| id as f32).collect(),
            vec![batch, sequence],
        )?;
        let attention_mask = attention_mask
            .map(|values| {
                require_shape(rectangle(values, "attention_mask")?, batch, sequence)?;
                Tensor::from_vec(
                    values
                        .iter()
                        .flatten()
                        .map(|&value| f32::from(value))
                        .collect(),
                    vec![batch, sequence],
                )
            })
            .transpose()?;
        let token_type_ids = token_type_ids
            .map(|values| {
                require_shape(rectangle(values, "token_type_ids")?, batch, sequence)?;
                Tensor::from_vec(
                    values.iter().flatten().map(|&id| id as f32).collect(),
                    vec![batch, sequence],
                )
            })
            .transpose()?;
        Ok(Self {
            input_ids,
            attention_mask,
            token_type_ids,
        })
    }
}

fn rectangle<T>(values: &[Vec<T>], name: &str) -> Result<(usize, usize)> {
    let sequence = values.first().map_or(0, Vec::len);
    if sequence == 0 || values.iter().any(|row| row.len() != sequence) {
        return Err(Error::InvalidShape(format!(
            "BERT {name} must be a non-empty rectangular batch"
        )));
    }
    Ok((values.len(), sequence))
}

fn require_shape(shape: (usize, usize), batch: usize, sequence: usize) -> Result<()> {
    if shape != (batch, sequence) {
        return Err(Error::InvalidShape(
            "BERT auxiliary input shape differs from input_ids".into(),
        ));
    }
    Ok(())
}
