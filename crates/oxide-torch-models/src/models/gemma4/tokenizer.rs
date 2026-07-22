use oxide_torch::{Error, Result};
use std::path::Path;
use tokenizers::Tokenizer;

/// Hugging Face `tokenizer.json` adapter for Gemma 4 prompts and output.
#[derive(Clone)]
pub struct Gemma4Tokenizer {
    inner: Tokenizer,
}

/// Stateful decoder for text emitted one token at a time.
pub struct Gemma4DecodeStream<'tokenizer> {
    step: Box<dyn FnMut(u32) -> Result<Option<String>> + 'tokenizer>,
}

impl Gemma4DecodeStream<'_> {
    /// Decodes one token, returning a chunk once it forms valid text.
    ///
    /// # Errors
    ///
    /// Returns an error if the token does not extend the decoder state with a
    /// valid prefix.
    pub fn step(&mut self, token_id: u32) -> Result<Option<String>> {
        (self.step)(token_id)
    }
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

    /// Creates a stateful decoder suitable for `Gemma4ForCausalLM::generate_stream`.
    ///
    /// Unlike decoding each token independently, this preserves whitespace and
    /// buffers incomplete multi-token Unicode sequences.
    #[must_use]
    pub fn decode_stream(&self, skip_special_tokens: bool) -> Gemma4DecodeStream<'_> {
        let mut stream = self.inner.decode_stream(skip_special_tokens);
        Gemma4DecodeStream {
            step: Box::new(move |token_id| {
                stream.step(token_id).map_err(|error| {
                    Error::Execution(format!("Gemma 4 streaming decode failed: {error}"))
                })
            }),
        }
    }

    #[must_use]
    pub fn vocabulary_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}
