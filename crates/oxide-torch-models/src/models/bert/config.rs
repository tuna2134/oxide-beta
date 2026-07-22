use oxide_torch::{Error, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct BertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f32,
    pub initializer_range: f32,
    pub num_labels: usize,
}

impl Default for BertConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30_522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3_072,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            initializer_range: 0.02,
            num_labels: 2,
        }
    }
}

impl BertConfig {
    pub(super) fn validate(&self) -> Result<()> {
        if self.hidden_size == 0
            || self.num_attention_heads == 0
            || self.hidden_size % self.num_attention_heads != 0
            || self.intermediate_size == 0
            || self.vocab_size == 0
            || self.max_position_embeddings == 0
            || self.type_vocab_size == 0
        {
            return Err(Error::InvalidShape(
                "invalid BERT dimensions or attention head count".into(),
            ));
        }
        Ok(())
    }
}
