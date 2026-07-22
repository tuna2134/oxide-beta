use super::config::BertConfig;
use super::math::{Linear, gelu, layer_norm};
use oxide_torch::nn::{Parameter, Trainable};
use oxide_torch::safetensors::{LoadedTensor, SafeTensorLoader};
use oxide_torch::{Device, Error, Result, Tensor};
use std::fs;
use std::path::Path;

#[derive(Clone, Debug)]
struct Norm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

#[derive(Clone, Debug)]
struct BertLayer {
    query: Linear,
    key: Linear,
    value: Linear,
    attention_output: Linear,
    attention_norm: Norm,
    intermediate: Linear,
    output: Linear,
    output_norm: Norm,
}

#[derive(Clone, Debug)]
pub struct BertModel {
    config: BertConfig,
    word_embeddings: Vec<f32>,
    position_embeddings: Vec<f32>,
    token_type_embeddings: Vec<f32>,
    embedding_norm: Norm,
    layers: Vec<BertLayer>,
    pooler: Option<Linear>,
}

#[derive(Clone, Debug)]
pub struct BertModelOutput {
    pub last_hidden_state: Tensor,
    pub pooler_output: Tensor,
}

impl BertModel {
    /// Loads a standard Hugging Face BERT `SafeTensors` checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported devices, invalid configuration, or missing weights.
    pub fn from_pretrained(directory: impl AsRef<Path>, device: Device) -> Result<Self> {
        if device != Device::Cpu {
            return Err(Error::Execution(
                "BERT currently supports the portable CPU backend".into(),
            ));
        }
        let directory = directory.as_ref();
        let config_path = directory.join("config.json");
        let bytes = fs::read(&config_path).map_err(|source| {
            Error::io(format!("failed to read {}", config_path.display()), source)
        })?;
        let config: BertConfig = serde_json::from_slice(&bytes)
            .map_err(|source| Error::json("invalid BERT config", source))?;
        config.validate()?;
        let loader = SafeTensorLoader::open(directory)?;
        Self::load(config, &loader)
    }

    fn load(config: BertConfig, loader: &SafeTensorLoader) -> Result<Self> {
        let prefix = if loader.contains("bert.embeddings.word_embeddings.weight") {
            "bert."
        } else {
            ""
        };
        let load = |suffix: &str| loader.load(&format!("{prefix}{suffix}"));
        let norm = |base: &str| -> Result<Norm> {
            Ok(Norm {
                weight: vector(load(&format!("{base}.weight"))?)?,
                bias: vector(load(&format!("{base}.bias"))?)?,
            })
        };
        let linear = |base: &str, input: usize, output: usize| -> Result<Linear> {
            linear_from(
                load(&format!("{base}.weight"))?,
                load(&format!("{base}.bias"))?,
                input,
                output,
            )
        };
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let base = format!("encoder.layer.{index}");
            layers.push(BertLayer {
                query: linear(
                    &format!("{base}.attention.self.query"),
                    config.hidden_size,
                    config.hidden_size,
                )?,
                key: linear(
                    &format!("{base}.attention.self.key"),
                    config.hidden_size,
                    config.hidden_size,
                )?,
                value: linear(
                    &format!("{base}.attention.self.value"),
                    config.hidden_size,
                    config.hidden_size,
                )?,
                attention_output: linear(
                    &format!("{base}.attention.output.dense"),
                    config.hidden_size,
                    config.hidden_size,
                )?,
                attention_norm: norm(&format!("{base}.attention.output.LayerNorm"))?,
                intermediate: linear(
                    &format!("{base}.intermediate.dense"),
                    config.hidden_size,
                    config.intermediate_size,
                )?,
                output: linear(
                    &format!("{base}.output.dense"),
                    config.intermediate_size,
                    config.hidden_size,
                )?,
                output_norm: norm(&format!("{base}.output.LayerNorm"))?,
            });
        }
        let pooler_name = format!("{prefix}pooler.dense.weight");
        let pooler = loader
            .contains(&pooler_name)
            .then(|| {
                linear_from(
                    loader.load(&pooler_name)?,
                    loader.load(&format!("{prefix}pooler.dense.bias"))?,
                    config.hidden_size,
                    config.hidden_size,
                )
            })
            .transpose()?;
        Ok(Self {
            word_embeddings: matrix(
                load("embeddings.word_embeddings.weight")?,
                config.vocab_size,
                config.hidden_size,
            )?,
            position_embeddings: matrix(
                load("embeddings.position_embeddings.weight")?,
                config.max_position_embeddings,
                config.hidden_size,
            )?,
            token_type_embeddings: matrix(
                load("embeddings.token_type_embeddings.weight")?,
                config.type_vocab_size,
                config.hidden_size,
            )?,
            embedding_norm: norm("embeddings.LayerNorm")?,
            layers,
            pooler,
            config,
        })
    }

    #[must_use]
    pub fn config(&self) -> &BertConfig {
        &self.config
    }

    /// Runs the BERT encoder and pooler for a rectangular token batch.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed ids/masks or incompatible model weights.
    pub fn forward(
        &self,
        input_ids: &[Vec<u32>],
        attention_mask: Option<&[Vec<u8>]>,
        token_type_ids: Option<&[Vec<u32>]>,
    ) -> Result<BertModelOutput> {
        let batch = input_ids.len();
        let sequence = input_ids.first().map_or(0, Vec::len);
        if batch == 0 || sequence == 0 || input_ids.iter().any(|row| row.len() != sequence) {
            return Err(Error::InvalidShape(
                "BERT input_ids must be a non-empty rectangle".into(),
            ));
        }
        if sequence > self.config.max_position_embeddings {
            return Err(Error::InvalidShape(
                "BERT sequence exceeds max_position_embeddings".into(),
            ));
        }
        let mask = attention_mask.map_or_else(
            || Ok(vec![vec![1; sequence]; batch]),
            |value| validate_mask(value, batch, sequence),
        )?;
        let types = token_type_ids.map_or_else(
            || Ok(vec![vec![0; sequence]; batch]),
            |value| validate_types(value, batch, sequence),
        )?;
        let hidden = self.config.hidden_size;
        let rows = batch * sequence;
        let mut states = vec![0.0; rows * hidden];
        for b in 0..batch {
            for position in 0..sequence {
                let token = usize::try_from(input_ids[b][position])
                    .map_err(|_| Error::InvalidShape("token id overflow".into()))?;
                let kind = usize::try_from(types[b][position])
                    .map_err(|_| Error::InvalidShape("token type id overflow".into()))?;
                if token >= self.config.vocab_size || kind >= self.config.type_vocab_size {
                    return Err(Error::InvalidShape(
                        "BERT token id is outside its embedding table".into(),
                    ));
                }
                let destination = (b * sequence + position) * hidden;
                for column in 0..hidden {
                    states[destination + column] = self.word_embeddings[token * hidden + column]
                        + self.position_embeddings[position * hidden + column]
                        + self.token_type_embeddings[kind * hidden + column];
                }
            }
        }
        layer_norm(
            &mut states,
            rows,
            hidden,
            &self.embedding_norm.weight,
            &self.embedding_norm.bias,
            self.config.layer_norm_eps,
        );
        for layer in &self.layers {
            states = layer.forward(&states, &mask, batch, sequence, &self.config)?;
        }
        let mut pooled = vec![0.0; batch * hidden];
        for b in 0..batch {
            pooled[b * hidden..(b + 1) * hidden]
                .copy_from_slice(&states[(b * sequence) * hidden..(b * sequence + 1) * hidden]);
        }
        if let Some(pooler) = &self.pooler {
            pooled = pooler.apply(&pooled, batch)?;
            for value in &mut pooled {
                *value = value.tanh();
            }
        }
        Ok(BertModelOutput {
            last_hidden_state: Tensor::from_vec(states, vec![batch, sequence, hidden])?,
            pooler_output: Tensor::from_vec(pooled, vec![batch, hidden])?,
        })
    }
}

impl BertLayer {
    #[allow(clippy::cast_precision_loss)]
    fn forward(
        &self,
        states: &[f32],
        mask: &[Vec<u8>],
        batch: usize,
        sequence: usize,
        config: &BertConfig,
    ) -> Result<Vec<f32>> {
        let hidden = config.hidden_size;
        let heads = config.num_attention_heads;
        let head_dim = hidden / heads;
        let rows = batch * sequence;
        let query = self.query.apply(states, rows)?;
        let key = self.key.apply(states, rows)?;
        let value = self.value.apply(states, rows)?;
        let mut context = vec![0.0; states.len()];
        let scale = (head_dim as f32).sqrt().recip();
        for b in 0..batch {
            for head in 0..heads {
                for q in 0..sequence {
                    let mut scores = vec![f32::NEG_INFINITY; sequence];
                    for k in 0..sequence {
                        if mask[b][k] != 0 {
                            let mut score = 0.0;
                            for d in 0..head_dim {
                                score += query[((b * sequence + q) * hidden) + head * head_dim + d]
                                    * key[((b * sequence + k) * hidden) + head * head_dim + d];
                            }
                            scores[k] = score * scale;
                        }
                    }
                    let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut normalizer = 0.0;
                    for score in &mut scores {
                        *score = (*score - maximum).exp();
                        normalizer += *score;
                    }
                    for k in 0..sequence {
                        let probability = scores[k] / normalizer;
                        for d in 0..head_dim {
                            context[((b * sequence + q) * hidden) + head * head_dim + d] +=
                                probability
                                    * value[((b * sequence + k) * hidden) + head * head_dim + d];
                        }
                    }
                }
            }
        }
        let mut attention = self.attention_output.apply(&context, rows)?;
        attention
            .iter_mut()
            .zip(states)
            .for_each(|(output, residual)| *output += residual);
        layer_norm(
            &mut attention,
            rows,
            hidden,
            &self.attention_norm.weight,
            &self.attention_norm.bias,
            config.layer_norm_eps,
        );
        let mut intermediate = self.intermediate.apply(&attention, rows)?;
        for value in &mut intermediate {
            *value = gelu(*value);
        }
        let mut output = self.output.apply(&intermediate, rows)?;
        output
            .iter_mut()
            .zip(&attention)
            .for_each(|(value, residual)| *value += residual);
        layer_norm(
            &mut output,
            rows,
            hidden,
            &self.output_norm.weight,
            &self.output_norm.bias,
            config.layer_norm_eps,
        );
        Ok(output)
    }
}

#[derive(Clone, Debug)]
pub struct BertForSequenceClassification {
    bert: BertModel,
    classifier: Parameter,
}

impl BertForSequenceClassification {
    /// Loads BERT and an optional `classifier.weight` sequence head.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid configuration or checkpoint tensors.
    pub fn from_pretrained(directory: impl AsRef<Path>, device: Device) -> Result<Self> {
        let directory = directory.as_ref();
        let bert = BertModel::from_pretrained(directory, device)?;
        let loader = SafeTensorLoader::open(directory)?;
        let num_labels = bert.config.num_labels;
        let hidden_size = bert.config.hidden_size;
        let weight = if loader.contains("classifier.weight") {
            let checkpoint_weight = loader.load("classifier.weight")?;
            let checkpoint_bias = loader.load("classifier.bias")?;
            classifier_parameters(checkpoint_weight, checkpoint_bias, num_labels, hidden_size)?
        } else {
            initialized_classifier(hidden_size + 1, num_labels)
        };
        Ok(Self {
            bert,
            classifier: Parameter::new(Tensor::from_vec(
                weight,
                vec![hidden_size + 1, num_labels],
            )?),
        })
    }

    /// Produces `[batch, num_labels]` logits.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid inputs or an incompatible classifier.
    pub fn forward(
        &self,
        input_ids: &[Vec<u32>],
        attention_mask: Option<&[Vec<u8>]>,
        token_type_ids: Option<&[Vec<u32>]>,
    ) -> Result<Tensor> {
        let pooled = self
            .bert
            .forward(input_ids, attention_mask, token_type_ids)?
            .pooler_output;
        let batch = pooled.shape()[0];
        let hidden = pooled.shape()[1];
        let pooled_values = pooled.to_vec()?;
        let mut augmented = Vec::with_capacity(batch * (hidden + 1));
        for row in pooled_values.chunks_exact(hidden) {
            augmented.extend_from_slice(row);
            augmented.push(1.0);
        }
        Tensor::from_vec(augmented, vec![batch, hidden + 1])?.matmul(self.classifier.value())
    }
}

impl Trainable for BertForSequenceClassification {
    fn visit_parameters(&self, visitor: &mut dyn FnMut(&Parameter)) {
        visitor(&self.classifier);
    }
    fn visit_parameters_mut(&mut self, visitor: &mut dyn FnMut(&mut Parameter)) {
        visitor(&mut self.classifier);
    }
}

fn linear_from(
    weight: LoadedTensor,
    bias: LoadedTensor,
    input: usize,
    output: usize,
) -> Result<Linear> {
    Ok(Linear {
        weight: matrix(weight, output, input)?,
        bias: vector_len(bias, output)?,
        input,
        output,
    })
}
fn matrix(tensor: LoadedTensor, rows: usize, columns: usize) -> Result<Vec<f32>> {
    if tensor.shape != [rows, columns] {
        return Err(Error::InvalidShape(format!(
            "BERT weight shape {:?}, expected [{rows}, {columns}]",
            tensor.shape
        )));
    }
    Ok(tensor.data)
}
fn vector(tensor: LoadedTensor) -> Result<Vec<f32>> {
    if tensor.shape.len() != 1 {
        return Err(Error::InvalidShape("expected BERT vector".into()));
    }
    Ok(tensor.data)
}
fn vector_len(tensor: LoadedTensor, len: usize) -> Result<Vec<f32>> {
    let value = vector(tensor)?;
    if value.len() != len {
        return Err(Error::InvalidShape("invalid BERT bias length".into()));
    }
    Ok(value)
}
fn validate_mask(value: &[Vec<u8>], batch: usize, sequence: usize) -> Result<Vec<Vec<u8>>> {
    if value.len() != batch || value.iter().any(|row| row.len() != sequence) {
        return Err(Error::InvalidShape("invalid BERT attention mask".into()));
    }
    Ok(value.to_vec())
}
fn validate_types(value: &[Vec<u32>], batch: usize, sequence: usize) -> Result<Vec<Vec<u32>>> {
    if value.len() != batch || value.iter().any(|row| row.len() != sequence) {
        return Err(Error::InvalidShape("invalid BERT token types".into()));
    }
    Ok(value.to_vec())
}
#[allow(clippy::cast_precision_loss)]
fn initialized_classifier(hidden: usize, labels: usize) -> Vec<f32> {
    (0..hidden * labels)
        .map(|index| ((index * 17 % 101) as f32 - 50.0) * 0.0002)
        .collect()
}
fn classifier_parameters(
    weight: LoadedTensor,
    bias: LoadedTensor,
    labels: usize,
    hidden: usize,
) -> Result<Vec<f32>> {
    let source = matrix(weight, labels, hidden)?;
    let bias = vector_len(bias, labels)?;
    let mut output = vec![0.0; (hidden + 1) * labels];
    for label in 0..labels {
        for column in 0..hidden {
            output[column * labels + label] = source[label * hidden + column];
        }
        output[hidden * labels + label] = bias[label];
    }
    Ok(output)
}
