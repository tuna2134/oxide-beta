use oxide_torch::{Error, Result};
use std::path::Path;
use tokenizers::Tokenizer;

pub struct BertTokenizer {
    inner: Tokenizer,
}

impl BertTokenizer {
    /// Loads `tokenizer.json` from a Hugging Face model directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the tokenizer is missing or invalid.
    pub fn from_pretrained(directory: impl AsRef<Path>) -> Result<Self> {
        let path = directory.as_ref().join("tokenizer.json");
        let inner = Tokenizer::from_file(&path).map_err(|error| {
            Error::Execution(format!(
                "failed to load BERT tokenizer {}: {error}",
                path.display()
            ))
        })?;
        Ok(Self { inner })
    }

    /// Encodes text and inserts BERT's configured special tokens.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization fails.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.inner
            .encode(text, true)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| Error::Execution(format!("BERT tokenization failed: {error}")))
    }
}
