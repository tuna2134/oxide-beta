use super::GenerationConfig;
use oxide_torch::{Error, Result};

/// Samples one token using temperature, top-k, and nucleus filtering.
///
/// # Errors
///
/// Returns an error for empty/non-finite logits or invalid parameters.
pub fn sample_token(logits: &[f32], config: &GenerationConfig, random: &mut u64) -> Result<u32> {
    if logits.is_empty() || config.temperature <= 0.0 || !(0.0..=1.0).contains(&config.top_p) {
        return Err(Error::Execution("invalid sampling input".into()));
    }
    let mut candidates: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .filter_map(|(index, &logit)| {
            logit
                .is_finite()
                .then_some((index, logit / config.temperature))
        })
        .collect();
    if candidates.is_empty() {
        return Err(Error::Execution("all logits are non-finite".into()));
    }
    let keep = config.top_k.max(1).min(candidates.len());
    if keep < candidates.len() {
        candidates.select_nth_unstable_by(keep - 1, |left, right| right.1.total_cmp(&left.1));
        candidates.truncate(keep);
    }
    candidates.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
    let maximum = candidates[0].1;
    let mut total = 0.0;
    for (_, score) in &mut candidates {
        *score = (*score - maximum).exp();
        total += *score;
    }
    if total == 0.0 || !total.is_finite() {
        return Err(Error::Execution(
            "sampling probability normalization failed".into(),
        ));
    }
    for (_, probability) in &mut candidates {
        *probability /= total;
    }
    if config.top_p < 1.0 {
        let mut cumulative = 0.0;
        let mut keep = 0;
        for (_, probability) in &candidates {
            cumulative += *probability;
            keep += 1;
            if cumulative >= config.top_p {
                break;
            }
        }
        candidates.truncate(keep.max(1));
        total = candidates.iter().map(|(_, probability)| probability).sum();
    } else {
        total = 1.0;
    }
    if *random == 0 {
        *random = 0x4d59_5df4_d0f3_3173;
    }
    *random ^= *random << 13;
    *random ^= *random >> 7;
    *random ^= *random << 17;
    #[allow(clippy::cast_precision_loss)]
    let unit = ((*random >> 40) as u32) as f32 / 16_777_216.0;
    let mut threshold = unit * total;
    for (index, probability) in &candidates {
        if threshold <= *probability {
            return u32::try_from(*index).map_err(|_| Error::Execution("token id overflow".into()));
        }
        threshold -= *probability;
    }
    let fallback = candidates
        .last()
        .ok_or_else(|| Error::Execution("sampling removed all candidates".into()))?;
    u32::try_from(fallback.0).map_err(|_| Error::Execution("token id overflow".into()))
}

/// Samples from a preselected, descending top-k candidate list. This is used
/// by the CUDA backend so only a few `(token, logit)` pairs cross D2H.
///
/// # Errors
///
/// Returns an error for invalid sampling parameters, empty candidates, or
/// non-finite candidate probabilities.
pub fn sample_topk_candidates(
    candidates: &[(u32, f32)],
    config: &GenerationConfig,
    random: &mut u64,
) -> Result<u32> {
    if candidates.is_empty() || config.temperature <= 0.0 || !(0.0..=1.0).contains(&config.top_p) {
        return Err(Error::Execution("invalid candidate sampling input".into()));
    }
    let mut candidates: Vec<(u32, f32)> = candidates
        .iter()
        .filter_map(|&(id, logit)| {
            logit
                .is_finite()
                .then_some((id, logit / config.temperature))
        })
        .collect();
    if candidates.is_empty() {
        return Err(Error::Execution(
            "all candidate logits are non-finite".into(),
        ));
    }
    candidates.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
    let maximum = candidates[0].1;
    let mut total = 0.0;
    for (_, score) in &mut candidates {
        *score = (*score - maximum).exp();
        total += *score;
    }
    if total == 0.0 || !total.is_finite() {
        return Err(Error::Execution(
            "candidate probability normalization failed".into(),
        ));
    }
    for (_, probability) in &mut candidates {
        *probability /= total;
    }
    if config.top_p < 1.0 {
        let mut cumulative = 0.0;
        let mut keep = 0;
        for (_, probability) in &candidates {
            cumulative += *probability;
            keep += 1;
            if cumulative >= config.top_p {
                break;
            }
        }
        candidates.truncate(keep.max(1));
        total = candidates.iter().map(|(_, probability)| probability).sum();
    } else {
        total = 1.0;
    }
    if *random == 0 {
        *random = 0x4d59_5df4_d0f3_3173;
    }
    *random ^= *random << 13;
    *random ^= *random >> 7;
    *random ^= *random << 17;
    #[allow(clippy::cast_precision_loss)]
    let unit = ((*random >> 40) as u32) as f32 / 16_777_216.0;
    let mut threshold = unit * total;
    for &(id, probability) in &candidates {
        if threshold <= probability {
            return Ok(id);
        }
        threshold -= probability;
    }
    candidates
        .last()
        .map(|candidate| candidate.0)
        .ok_or_else(|| Error::Execution("candidate sampling removed all tokens".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn respects_top_k_one() {
        let config = GenerationConfig {
            top_k: 1,
            ..GenerationConfig::default()
        };
        let mut random = config.seed;
        assert_eq!(
            sample_token(&[0.0, 5.0, 1.0], &config, &mut random).unwrap(),
            1
        );
    }
}
