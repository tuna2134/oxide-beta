//! Differentiable loss functions.

use crate::{Result, Tensor};

/// Mean sparse categorical cross-entropy over `[batch, classes]` logits.
///
/// # Errors
///
/// Returns an error for invalid target/logit shapes, devices, or class values.
pub fn cross_entropy(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    logits.cross_entropy(targets)
}
