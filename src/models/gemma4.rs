//! Gemma 4 text-model configuration and Hugging Face checkpoint adapter.
//!
//! The layout follows `transformers.models.gemma4.Gemma4TextModel`: scaled
//! token embeddings, 5:1 local/global decoder layers, GQA, PLE, gated MLP,
//! final `RMSNorm`, and a tied or independent LM head.

use crate::safetensors::{LoadedTensor, SafeTensorLoader};
use crate::{Device, Error, Result, Tensor};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

fn default_vocab_size() -> usize {
    262_144
}
fn default_hidden_size() -> usize {
    2_304
}
fn default_intermediate_size() -> usize {
    9_216
}
fn default_layers() -> usize {
    30
}
fn default_heads() -> usize {
    8
}
fn default_kv_heads() -> usize {
    4
}
fn default_head_dim() -> usize {
    256
}
fn default_global_head_dim() -> usize {
    512
}
fn default_max_positions() -> usize {
    131_072
}
fn default_sliding_window() -> usize {
    512
}
fn default_eps() -> f32 {
    1e-6
}
fn default_ple_vocab() -> usize {
    262_144
}
fn default_ple_dim() -> usize {
    256
}
fn default_true() -> bool {
    true
}

/// Hugging Face compatible `Gemma4TextConfig` subset used by the Rust model.
#[derive(Clone, Debug, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct Gemma4TextConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub num_global_key_value_heads: Option<usize>,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_global_head_dim")]
    pub global_head_dim: usize,
    #[serde(default = "default_max_positions")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default = "default_ple_vocab")]
    pub vocab_size_per_layer_input: usize,
    #[serde(default = "default_ple_dim")]
    pub hidden_size_per_layer_input: usize,
    #[serde(default)]
    pub attention_k_eq_v: bool,
    #[serde(default)]
    pub num_kv_shared_layers: usize,
    #[serde(default)]
    pub enable_moe_block: bool,
    #[serde(default)]
    pub use_double_wide_mlp: bool,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
}

impl Gemma4TextConfig {
    fn finish(mut self) -> Result<Self> {
        if self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.num_attention_heads % self.num_key_value_heads != 0
            || self.hidden_size == 0
            || self.num_hidden_layers == 0
        {
            return Err(Error::InvalidShape(
                "invalid Gemma 4 attention configuration".into(),
            ));
        }
        if self.layer_types.is_none() {
            self.layer_types = Some(
                (0..self.num_hidden_layers)
                    .map(|index| {
                        if (index + 1) % 6 == 0 || index + 1 == self.num_hidden_layers {
                            "full_attention".to_owned()
                        } else {
                            "sliding_attention".to_owned()
                        }
                    })
                    .collect(),
            );
        }
        if self
            .layer_types
            .as_ref()
            .is_some_and(|types| types.len() != self.num_hidden_layers)
        {
            return Err(Error::InvalidShape(
                "Gemma 4 layer_types length does not match num_hidden_layers".into(),
            ));
        }
        Ok(self)
    }
}

/// A lazily loaded Gemma 4 causal-language-model checkpoint.
///
/// `SafeTensors` shards remain on disk and individual weights are decoded only
/// when requested. This avoids converting an entire BF16 checkpoint to an
/// additional f32 copy during model construction.
#[derive(Clone, Debug)]
pub struct Gemma4ForCausalLM {
    config: Gemma4TextConfig,
    weights: SafeTensorLoader,
    device: Device,
    root: PathBuf,
}

impl Gemma4ForCausalLM {
    /// Loads `config.json` plus a single or sharded `SafeTensors` checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid config or incomplete checkpoint.
    pub fn from_pretrained(directory: impl AsRef<Path>, device: Device) -> Result<Self> {
        let root = directory.as_ref().to_owned();
        let config_path = root.join("config.json");
        let bytes = fs::read(&config_path).map_err(|error| {
            Error::Execution(format!("failed to read {}: {error}", config_path.display()))
        })?;
        let value: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|error| Error::Execution(format!("invalid Gemma 4 config: {error}")))?;
        let config = if let Some(text) = value.get("text_config") {
            serde_json::from_value::<Gemma4TextConfig>(text.clone())
        } else {
            serde_json::from_value::<Gemma4TextConfig>(value)
        }
        .map_err(|error| Error::Execution(format!("invalid Gemma4TextConfig: {error}")))?
        .finish()?;
        let weights = SafeTensorLoader::open(&root)?;
        let model = Self {
            config,
            weights,
            device,
            root,
        };
        model.validate_checkpoint()?;
        Ok(model)
    }

    #[must_use]
    pub fn config(&self) -> &Gemma4TextConfig {
        &self.config
    }

    #[must_use]
    pub fn device(&self) -> Device {
        self.device
    }

    #[must_use]
    pub fn checkpoint_directory(&self) -> &Path {
        &self.root
    }

    /// Loads a Hugging Face parameter, accepting common multimodal prefixes.
    ///
    /// # Errors
    ///
    /// Returns an error if the parameter is missing or cannot be decoded.
    pub fn weight(&self, suffix: &str) -> Result<LoadedTensor> {
        let model = format!("model.{suffix}");
        let model_language = format!("model.language_model.{suffix}");
        let language = format!("language_model.{suffix}");
        let bare = suffix.to_owned();
        self.weights.load_any([
            model.as_str(),
            model_language.as_str(),
            language.as_str(),
            bare.as_str(),
        ])
    }

    /// Converts a named `SafeTensors` parameter into an oxide-torch `Tensor`.
    ///
    /// # Errors
    ///
    /// Returns an error if loading, conversion, or device transfer fails.
    pub fn weight_tensor(&self, suffix: &str) -> Result<Tensor> {
        self.weight(suffix)?.into_tensor(self.device)
    }

    /// Performs the scaled token-embedding lookup used at the model input.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid token, embedding shape, or device.
    pub fn embed(&self, input_ids: &[u32]) -> Result<Tensor> {
        let embedding = self.weight("embed_tokens.weight")?;
        expect_shape(
            &embedding,
            &[self.config.vocab_size, self.config.hidden_size],
            "embed_tokens.weight",
        )?;
        let mut output = Vec::with_capacity(input_ids.len() * self.config.hidden_size);
        #[allow(clippy::cast_precision_loss)]
        let scale = (self.config.hidden_size as f32).sqrt();
        for &token in input_ids {
            let token =
                usize::try_from(token).map_err(|_| Error::Execution("token id overflow".into()))?;
            if token >= self.config.vocab_size {
                return Err(Error::Execution(format!(
                    "token id {token} exceeds vocabulary"
                )));
            }
            output.extend(
                embedding.data
                    [token * self.config.hidden_size..(token + 1) * self.config.hidden_size]
                    .iter()
                    .map(|value| value * scale),
            );
        }
        Ok(
            Tensor::from_vec(output, vec![1, input_ids.len(), self.config.hidden_size])?
                .to(self.device),
        )
    }

    fn validate_checkpoint(&self) -> Result<()> {
        self.require_weight("embed_tokens.weight")?;
        self.require_weight("norm.weight")?;
        for layer in 0..self.config.num_hidden_layers {
            let prefix = format!("layers.{layer}");
            for norm in [
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ] {
                self.require_weight(&format!("{prefix}.{norm}.weight"))?;
            }
        }
        if !self.config.tie_word_embeddings {
            self.require_weight("lm_head.weight")?;
        }
        Ok(())
    }

    fn require_weight(&self, suffix: &str) -> Result<()> {
        let model = format!("model.{suffix}");
        let model_language = format!("model.language_model.{suffix}");
        let language = format!("language_model.{suffix}");
        if self.weights.contains_any([
            model.as_str(),
            model_language.as_str(),
            language.as_str(),
            suffix,
        ]) {
            Ok(())
        } else {
            Err(Error::Execution(format!(
                "required Gemma 4 tensor `{suffix}` is missing"
            )))
        }
    }
}

fn expect_shape(tensor: &LoadedTensor, expected: &[usize], name: &str) -> Result<()> {
    if tensor.shape == expected {
        Ok(())
    } else {
        Err(Error::InvalidShape(format!(
            "Gemma 4 tensor {name} has shape {:?}, expected {expected:?}",
            tensor.shape
        )))
    }
}
