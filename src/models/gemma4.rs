//! Gemma 4 text-model configuration and Hugging Face checkpoint adapter.
//!
//! The layout follows `transformers.models.gemma4.Gemma4TextModel`: scaled
//! token embeddings, 5:1 local/global decoder layers, GQA, PLE, gated MLP,
//! final `RMSNorm`, and a tied or independent LM head.

use crate::safetensors::{LoadedTensor, SafeTensorLoader, TensorMetadata};
use crate::{Device, Error, Result, Tensor};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

/// Generation parameters matching `google/gemma-4-E2B-it` defaults.
#[derive(Clone, Debug)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub eos_token_ids: Vec<u32>,
    pub seed: u64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            temperature: 1.0,
            top_k: 64,
            top_p: 0.95,
            eos_token_ids: vec![1, 50, 106],
            seed: 0x4d59_5df4_d0f3_3173,
        }
    }
}

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
    candidates.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
    candidates.truncate(config.top_k.max(1).min(candidates.len()));
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

/// Reference grouped-query causal attention used to validate the CUDA kernel.
/// Layouts are Q=`[q, heads, dim]`, K/V=`[kv, kv_heads, dim]`.
///
/// # Errors
///
/// Returns an error for incompatible shapes or an invalid sliding window.
#[allow(clippy::too_many_arguments)]
pub fn grouped_query_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    q_len: usize,
    kv_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    sliding_window: Option<usize>,
) -> Result<Vec<f32>> {
    if heads == 0
        || kv_heads == 0
        || heads % kv_heads != 0
        || q_len > kv_len
        || query.len() != q_len * heads * head_dim
        || key.len() != kv_len * kv_heads * head_dim
        || value.len() != key.len()
        || sliding_window == Some(0)
    {
        return Err(Error::InvalidShape(
            "invalid grouped-query attention shape".into(),
        ));
    }
    let groups = heads / kv_heads;
    let query_offset = kv_len - q_len;
    let mut output = vec![0.0; query.len()];
    let mut scores = vec![0.0; kv_len];
    for q in 0..q_len {
        let absolute_q = query_offset + q;
        let start = sliding_window.map_or(0, |window| (absolute_q + 1).saturating_sub(window));
        for head in 0..heads {
            let kv_head = head / groups;
            let q_base = (q * heads + head) * head_dim;
            let mut maximum = f32::NEG_INFINITY;
            for (position, score_slot) in scores
                .iter_mut()
                .enumerate()
                .take(absolute_q + 1)
                .skip(start)
            {
                let k_base = (position * kv_heads + kv_head) * head_dim;
                let score = query[q_base..q_base + head_dim]
                    .iter()
                    .zip(&key[k_base..k_base + head_dim])
                    .map(|(a, b)| a * b)
                    .sum();
                *score_slot = score;
                maximum = maximum.max(score);
            }
            let normalizer: f32 = (start..=absolute_q)
                .map(|position| {
                    scores[position] = (scores[position] - maximum).exp();
                    scores[position]
                })
                .sum();
            let out_base = q_base;
            for (position, score) in scores.iter().enumerate().take(absolute_q + 1).skip(start) {
                let probability = score / normalizer;
                let v_base = (position * kv_heads + kv_head) * head_dim;
                for dimension in 0..head_dim {
                    output[out_base + dimension] += probability * value[v_base + dimension];
                }
            }
        }
    }
    Ok(output)
}

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

fn default_rope_theta() -> f32 {
    10_000.0
}

fn default_rope_factor() -> f32 {
    1.0
}

/// Per-layer-type rotary embedding parameters from Hugging Face config.
#[derive(Clone, Debug, Deserialize)]
pub struct Gemma4RopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_type: String,
    #[serde(default = "default_rope_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default = "default_rope_factor")]
    pub factor: f32,
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
    #[serde(default)]
    pub rope_parameters: Option<HashMap<String, Gemma4RopeParameters>>,
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
        if self.rope_parameters.is_none() {
            self.rope_parameters = Some(HashMap::from([
                (
                    "sliding_attention".to_owned(),
                    Gemma4RopeParameters {
                        rope_theta: 10_000.0,
                        rope_type: "default".to_owned(),
                        partial_rotary_factor: 1.0,
                        factor: 1.0,
                    },
                ),
                (
                    "full_attention".to_owned(),
                    Gemma4RopeParameters {
                        rope_theta: 1_000_000.0,
                        rope_type: "proportional".to_owned(),
                        partial_rotary_factor: 0.25,
                        factor: 1.0,
                    },
                ),
            ]));
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

/// Hugging Face `tokenizer.json` adapter for Gemma 4 prompts and output.
#[derive(Clone)]
pub struct Gemma4Tokenizer {
    inner: Tokenizer,
}

impl std::fmt::Debug for Gemma4Tokenizer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Gemma4Tokenizer")
            .finish_non_exhaustive()
    }
}

impl Gemma4Tokenizer {
    /// Loads `tokenizer.json` from a Hugging Face model directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the tokenizer is missing or malformed.
    pub fn from_pretrained(directory: impl AsRef<Path>) -> Result<Self> {
        let path = directory.as_ref().join("tokenizer.json");
        let inner = Tokenizer::from_file(&path).map_err(|error| {
            Error::Execution(format!("failed to load {}: {error}", path.display()))
        })?;
        Ok(Self { inner })
    }

    /// Encodes plain text using the checkpoint's normalizer, pre-tokenizer,
    /// model, and post-processor.
    ///
    /// # Errors
    ///
    /// Returns an error when tokenization fails.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.inner
            .encode(text, add_special_tokens)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| Error::Execution(format!("Gemma 4 tokenization failed: {error}")))
    }

    /// Formats a single-turn Gemma instruction prompt and tokenizes it.
    ///
    /// # Errors
    ///
    /// Returns an error when tokenization fails.
    pub fn encode_user_turn(&self, prompt: &str) -> Result<Vec<u32>> {
        self.encode(
            &format!("<|turn>user\n{}<turn|>\n<|turn>model\n", prompt.trim()),
            true,
        )
    }

    /// Decodes token IDs with optional special-token removal.
    ///
    /// # Errors
    ///
    /// Returns an error when the tokenizer decoder rejects the sequence.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|error| Error::Execution(format!("Gemma 4 decoding failed: {error}")))
    }

    #[must_use]
    pub fn vocabulary_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

impl Gemma4ForCausalLM {
    /// Uploads every BF16 language-model weight to a persistent CUDA store.
    ///
    /// # Errors
    ///
    /// Returns an error when CUDA/cuBLAS initialization or a weight upload
    /// fails. The model must have been created for the same CUDA device.
    #[cfg(feature = "cuda")]
    pub fn prepare_cuda(&self) -> Result<crate::gemma4_cuda::Gemma4CudaState> {
        let Device::Cuda(device) = self.device else {
            return Err(Error::Execution(
                "prepare_cuda requires a model on Device::Cuda".into(),
            ));
        };
        crate::gemma4_cuda::Gemma4CudaState::load(self, device)
    }

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

    #[cfg(feature = "cuda")]
    pub(crate) fn checkpoint_weight_names(&self) -> impl Iterator<Item = &str> {
        self.weights.names()
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn with_checkpoint_view<R>(
        &self,
        name: &str,
        visitor: impl FnOnce(safetensors::Dtype, &[usize], &[u8]) -> Result<R>,
    ) -> Result<R> {
        self.weights.with_view(name, visitor)
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

    /// Returns checkpoint metadata without decoding the weight payload.
    ///
    /// # Errors
    ///
    /// Returns an error if no compatible parameter alias exists.
    pub fn weight_metadata(&self, suffix: &str) -> Result<TensorMetadata> {
        let names = Self::weight_aliases(suffix);
        for name in &names {
            if self.weights.contains(name) {
                return self.weights.metadata(name);
            }
        }
        Err(Error::Execution(format!(
            "required Gemma 4 tensor `{suffix}` is missing"
        )))
    }

    /// Borrows the original `SafeTensors` payload for a model weight.
    ///
    /// # Errors
    ///
    /// Returns an error if no alias exists or `visitor` rejects the tensor.
    pub fn with_weight_view<R>(
        &self,
        suffix: &str,
        visitor: impl FnOnce(safetensors::Dtype, &[usize], &[u8]) -> Result<R>,
    ) -> Result<R> {
        let names = Self::weight_aliases(suffix);
        for name in &names {
            if self.weights.contains(name) {
                return self.weights.with_view(name, visitor);
            }
        }
        Err(Error::Execution(format!(
            "required Gemma 4 tensor `{suffix}` is missing"
        )))
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
        let names = Self::weight_aliases(suffix);
        if self.weights.contains_any(names.iter().map(String::as_str)) {
            Ok(())
        } else {
            Err(Error::Execution(format!(
                "required Gemma 4 tensor `{suffix}` is missing"
            )))
        }
    }

    fn weight_aliases(suffix: &str) -> [String; 4] {
        [
            format!("model.{suffix}"),
            format!("model.language_model.{suffix}"),
            format!("language_model.{suffix}"),
            suffix.to_owned(),
        ]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_cache_applies_sliding_window() {
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

    #[test]
    fn grouped_query_attention_obeys_causality() {
        let output =
            grouped_query_attention(&[1.0, 1.0], &[1.0, 1.0], &[2.0, 6.0], 2, 2, 1, 1, 1, None)
                .unwrap();
        assert!((output[0] - 2.0).abs() < 1e-6);
        assert!((output[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn sampling_respects_top_k_one() {
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
