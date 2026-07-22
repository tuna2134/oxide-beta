use super::Gemma4TextConfig;
use oxide_torch::safetensors::{LoadedTensor, SafeTensorLoader, TensorMetadata};
use oxide_torch::{Device, Error, Result, Tensor};
use std::fs;
use std::path::{Path, PathBuf};
#[cfg(feature = "cuda")]
use std::{cell::RefCell, rc::Rc};

/// A lazily loaded Gemma 4 causal-language-model checkpoint.
///
/// `SafeTensors` shards remain on disk and individual weights are decoded only
/// when requested. This avoids converting an entire BF16 checkpoint to an
/// additional f32 copy during model construction.
#[derive(Clone)]
pub struct Gemma4ForCausalLM {
    pub(super) config: Gemma4TextConfig,
    weights: SafeTensorLoader,
    pub(super) device: Device,
    root: PathBuf,
    #[cfg(feature = "cuda")]
    pub(super) cuda: Rc<RefCell<Option<crate::gemma4_cuda::Gemma4CudaState>>>,
}

impl std::fmt::Debug for Gemma4ForCausalLM {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Gemma4ForCausalLM")
            .field("config", &self.config)
            .field("device", &self.device)
            .field("root", &self.root)
            .finish_non_exhaustive()
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
            #[cfg(feature = "cuda")]
            cuda: Rc::new(RefCell::new(None)),
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
