use super::{Gemma4ForCausalLM, GenerationConfig, sample_token, sample_topk_candidates};
use oxide_torch::{Device, Error, Result};
use std::collections::HashSet;

impl GenerationConfig {
    fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(Error::Execution(
                "generation temperature must be finite and greater than zero".into(),
            ));
        }
        if !self.top_p.is_finite() || !(0.0..=1.0).contains(&self.top_p) {
            return Err(Error::Execution(
                "generation top_p must be finite and between zero and one".into(),
            ));
        }
        if self.top_k == 0 {
            return Err(Error::Execution(
                "generation top_k must be greater than zero".into(),
            ));
        }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty < 1.0 {
            return Err(Error::Execution(
                "generation repetition_penalty must be finite and at least one".into(),
            ));
        }
        if self.stop_token_sequences.iter().any(Vec::is_empty) {
            return Err(Error::Execution(
                "generation stop-token sequences cannot be empty".into(),
            ));
        }
        Ok(())
    }
}

impl Gemma4ForCausalLM {
    /// Autoregressively generates tokens, mirroring Transformers' `generate`.
    ///
    /// The returned sequence contains `input_ids` followed by generated IDs.
    /// CUDA weights are initialized lazily on the first call and reused by
    /// later calls and cloned model handles.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid generation parameters, empty input, a
    /// non-CUDA model, or a CUDA inference failure.
    pub fn generate(&self, input_ids: &[u32], generation: &GenerationConfig) -> Result<Vec<u32>> {
        self.generate_with_callback(input_ids, generation, |_| Ok(()))
    }

    /// Autoregressively generates tokens and reports each token immediately.
    ///
    /// The callback runs after a token is appended to the output, including a
    /// terminating token. Use [`Gemma4Tokenizer::decode_stream`](super::Gemma4Tokenizer::decode_stream)
    /// in the callback when text chunks rather than token IDs are required.
    /// The returned sequence has the same layout as [`Self::generate`].
    ///
    /// # Errors
    ///
    /// Returns generation errors as well as errors returned by `on_token`.
    pub fn generate_stream(
        &self,
        input_ids: &[u32],
        generation: &GenerationConfig,
        on_token: impl FnMut(u32) -> Result<()>,
    ) -> Result<Vec<u32>> {
        self.generate_with_callback(input_ids, generation, on_token)
    }

    fn generate_with_callback(
        &self,
        input_ids: &[u32],
        generation: &GenerationConfig,
        mut on_token: impl FnMut(u32) -> Result<()>,
    ) -> Result<Vec<u32>> {
        generation.validate()?;
        if input_ids.is_empty() {
            return Err(Error::Execution(
                "Gemma 4 generation requires at least one input token".into(),
            ));
        }
        if !matches!(self.device, Device::Cuda(_)) {
            return Err(Error::Execution(
                "Gemma 4 generate currently requires Device::Cuda".into(),
            ));
        }
        let maximum_sequence = input_ids
            .len()
            .checked_add(generation.max_new_tokens)
            .ok_or_else(|| Error::Execution("Gemma 4 sequence length overflow".into()))?;

        if self.cuda.borrow().is_none() {
            let state = self.prepare_cuda()?;
            *self.cuda.borrow_mut() = Some(state);
        }
        let cuda = self.cuda.borrow();
        let cuda = cuda
            .as_ref()
            .ok_or_else(|| Error::Execution("Gemma 4 CUDA state is missing".into()))?;
        cuda.reset_generation_state();

        let mut table = cuda.new_cache_table(&self.config, maximum_sequence)?;
        let mut logits = cuda.prefill_prompt(input_ids, &self.config, &mut table)?;
        let mut random = generation.seed;
        let mut generated = Vec::with_capacity(generation.max_new_tokens);
        let mut candidates: Option<Vec<(u32, f32)>> = None;

        for _ in 0..generation.max_new_tokens {
            if candidates.is_none() && generation.repetition_penalty > 1.0 {
                apply_repetition_penalty(&mut logits, &generated, generation.repetition_penalty);
            }
            let next = match &candidates {
                Some(values) => sample_topk_candidates(values, generation, &mut random)?,
                None => sample_token(&logits, generation, &mut random)?,
            };
            generated.push(next);
            on_token(next)?;
            if generation.eos_token_ids.contains(&next)
                || generation
                    .stop_token_sequences
                    .iter()
                    .any(|stop| generated.ends_with(stop))
            {
                break;
            }
            candidates = Some(cuda.decode_topk(
                next,
                &self.config,
                &mut table,
                generation.top_k,
                generation.repetition_penalty,
                true,
            )?);
        }

        let mut sequences = Vec::with_capacity(input_ids.len() + generated.len());
        sequences.extend_from_slice(input_ids);
        sequences.extend_from_slice(&generated);
        Ok(sequences)
    }
}

fn apply_repetition_penalty(logits: &mut [f32], generated: &[u32], penalty: f32) {
    let mut seen = HashSet::with_capacity(generated.len());
    for &token in generated {
        if !seen.insert(token) {
            continue;
        }
        if let Some(logit) = logits.get_mut(token as usize) {
            if *logit >= 0.0 {
                *logit /= penalty;
            } else {
                *logit *= penalty;
            }
        }
    }
}
