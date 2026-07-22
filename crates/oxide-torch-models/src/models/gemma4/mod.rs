//! Gemma 4 text model, tokenizer, generation, and checkpoint adapter.
//!
//! Public items are re-exported here so existing `gemma4::...` paths remain
//! stable while each implementation lives in a responsibility-focused module.

mod cache;
mod config;
#[cfg(feature = "cuda")]
mod generation;
mod model;
mod sampling;
mod tokenizer;

pub use cache::KvCache;
pub use config::{Gemma4RopeParameters, Gemma4TextConfig, GenerationConfig};
pub use model::Gemma4ForCausalLM;
pub use oxide_torch::nn::grouped_query_attention;
pub use sampling::{sample_token, sample_topk_candidates};
pub use tokenizer::Gemma4Tokenizer;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouped_query_attention_obeys_causality() {
        let output =
            grouped_query_attention(&[1.0, 1.0], &[1.0, 1.0], &[2.0, 6.0], 2, 2, 1, 1, 1, None)
                .unwrap();
        assert!((output[0] - 2.0).abs() < 1e-6);
        assert!((output[1] - 4.0).abs() < 1e-6);
    }
}
