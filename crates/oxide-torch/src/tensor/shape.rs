use crate::{Error, Result};

pub(super) fn validate_numel(data_len: usize, shape: &[usize]) -> Result<()> {
    let expected = checked_numel(shape)?;
    if data_len != expected {
        return Err(Error::InvalidShape(format!(
            "shape {shape:?} contains {expected} elements, but data contains {data_len}"
        )));
    }
    Ok(())
}

pub(super) fn checked_numel(shape: &[usize]) -> Result<usize> {
    if shape.is_empty() {
        return Err(Error::InvalidShape(
            "scalar shapes are not implemented".into(),
        ));
    }
    shape.iter().try_fold(1usize, |total, &dimension| {
        total
            .checked_mul(dimension)
            .ok_or_else(|| Error::InvalidShape("element count overflow".into()))
    })
}
