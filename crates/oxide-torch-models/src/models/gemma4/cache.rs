use oxide_torch::{Error, Result};

/// Per-layer autoregressive key/value state in `[sequence, kv_head, head_dim]` order.
#[derive(Clone, Debug, Default)]
pub struct KvCache {
    key: Vec<f32>,
    value: Vec<f32>,
    kv_heads: usize,
    head_dim: usize,
    sequence_len: usize,
}

impl KvCache {
    /// Appends one or more positions, retaining only `window` latest positions when set.
    ///
    /// # Errors
    ///
    /// Returns an error when K/V shapes are inconsistent.
    pub fn append(
        &mut self,
        key: &[f32],
        value: &[f32],
        kv_heads: usize,
        positions: usize,
        head_dim: usize,
        window: Option<usize>,
    ) -> Result<()> {
        let added = kv_heads
            .checked_mul(positions)
            .and_then(|size| size.checked_mul(head_dim))
            .ok_or_else(|| Error::InvalidShape("KV cache shape overflow".into()))?;
        if key.len() != added || value.len() != added || kv_heads == 0 || head_dim == 0 {
            return Err(Error::InvalidShape("invalid KV cache update".into()));
        }
        if self.sequence_len != 0 && (self.kv_heads != kv_heads || self.head_dim != head_dim) {
            return Err(Error::InvalidShape("KV cache dimensions changed".into()));
        }
        self.kv_heads = kv_heads;
        self.head_dim = head_dim;
        self.key.extend_from_slice(key);
        self.value.extend_from_slice(value);
        self.sequence_len += positions;
        if let Some(window) = window {
            if self.sequence_len > window {
                let discard_positions = self.sequence_len - window;
                let discard = discard_positions * kv_heads * head_dim;
                self.key.drain(..discard);
                self.value.drain(..discard);
                self.sequence_len = window;
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn sequence_len(&self) -> usize {
        self.sequence_len
    }

    #[must_use]
    pub fn key(&self) -> &[f32] {
        &self.key
    }

    #[must_use]
    pub fn value(&self) -> &[f32] {
        &self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_sliding_window() {
        let mut cache = KvCache::default();
        cache
            .append(&[1.0, 2.0], &[3.0, 4.0], 1, 2, 1, Some(3))
            .unwrap();
        cache
            .append(&[5.0, 6.0], &[7.0, 8.0], 1, 2, 1, Some(3))
            .unwrap();
        assert_eq!(cache.sequence_len(), 3);
        assert_eq!(cache.key(), &[2.0, 5.0, 6.0]);
        assert_eq!(cache.value(), &[4.0, 7.0, 8.0]);
    }
}
