#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::semicolon_if_nothing_returned,
    clippy::similar_names
)]

use super::config::BertConfig;

use super::input::BertInput;
use oxide_torch::nn::{Module, Parameter, Trainable};
use oxide_torch::safetensors::{LoadedTensor, SafeTensorLoader};
use oxide_torch::transformer as ops;
use oxide_torch::{Device, Error, Result, Tensor};
use safetensors::tensor::TensorView;
use safetensors::{Dtype, serialize_to_file};
use std::{fs, path::Path};

#[derive(Clone, Debug)]
struct Linear {
    w: Parameter,
    b: Parameter,
    out: usize,
}
impl Linear {
    fn load(
        loader: &SafeTensorLoader,
        name: &str,
        input: usize,
        out: usize,
        device: Device,
    ) -> Result<Self> {
        Ok(Self {
            w: param(
                loader.load(&format!("{name}.weight"))?,
                vec![out, input],
                device,
            )?,
            b: param(loader.load(&format!("{name}.bias"))?, vec![out], device)?,
            out,
        })
    }
    fn f(&self, x: &Tensor) -> Result<Tensor> {
        ops::linear(x, self.w.value(), self.b.value(), self.out)
    }
    fn initialized(input: usize, out: usize, scale: f32, device: Device) -> Result<Self> {
        let mut state = 0x4d59_5df4_d0f3_3173_u64;
        let weights = (0..out * input)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let unit = (state >> 40) as f32 / 16_777_216.0;
                (unit * 2.0 - 1.0) * scale
            })
            .collect();
        Ok(Self {
            w: Parameter::new(Tensor::from_vec(weights, vec![out, input])?.to(device)),
            b: Parameter::new(Tensor::zeros(vec![out])?.to(device)),
            out,
        })
    }
    fn visit(&self, v: &mut dyn FnMut(&Parameter)) {
        v(&self.w);
        v(&self.b)
    }
    fn visit_mut(&mut self, v: &mut dyn FnMut(&mut Parameter)) {
        v(&mut self.w);
        v(&mut self.b)
    }
}
#[derive(Clone, Debug)]
struct Norm {
    w: Parameter,
    b: Parameter,
    eps: f32,
}
impl Norm {
    fn load(l: &SafeTensorLoader, n: &str, h: usize, e: f32, device: Device) -> Result<Self> {
        Ok(Self {
            w: param(
                l.load_any([&*format!("{n}.weight"), &*format!("{n}.gamma")])?,
                vec![h],
                device,
            )?,
            b: param(
                l.load_any([&*format!("{n}.bias"), &*format!("{n}.beta")])?,
                vec![h],
                device,
            )?,
            eps: e,
        })
    }
    fn f(&self, x: &Tensor) -> Result<Tensor> {
        ops::layer_norm(x, self.w.value(), self.b.value(), self.eps)
    }
    fn visit(&self, v: &mut dyn FnMut(&Parameter)) {
        v(&self.w);
        v(&self.b)
    }
    fn visit_mut(&mut self, v: &mut dyn FnMut(&mut Parameter)) {
        v(&mut self.w);
        v(&mut self.b)
    }
}
#[derive(Clone, Debug)]
struct Layer {
    q: Linear,
    k: Linear,
    v: Linear,
    attn: Linear,
    an: Norm,
    up: Linear,
    down: Linear,
    on: Norm,
    heads: usize,
}
impl Layer {
    fn f(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let c = ops::scaled_dot_product_attention(
            x,
            mask,
            self.q.w.value(),
            self.q.b.value(),
            self.k.w.value(),
            self.k.b.value(),
            self.v.w.value(),
            self.v.b.value(),
            self.heads,
        )?;
        let a = self.an.f(&self.attn.f(&c)?.add(x)?)?;
        let m = ops::gelu(&self.up.f(&a)?)?;
        self.on.f(&self.down.f(&m)?.add(&a)?)
    }
    fn visit(&self, v: &mut dyn FnMut(&Parameter)) {
        for l in [&self.q, &self.k, &self.v, &self.attn, &self.up, &self.down] {
            l.visit(v)
        }
        self.an.visit(v);
        self.on.visit(v)
    }
    fn visit_mut(&mut self, v: &mut dyn FnMut(&mut Parameter)) {
        for l in [
            &mut self.q,
            &mut self.k,
            &mut self.v,
            &mut self.attn,
            &mut self.up,
            &mut self.down,
        ] {
            l.visit_mut(v)
        }
        self.an.visit_mut(v);
        self.on.visit_mut(v)
    }
}

#[derive(Clone, Debug)]
pub struct BertModel {
    config: BertConfig,
    word: Parameter,
    pos: Parameter,
    types: Parameter,
    en: Norm,
    layers: Vec<Layer>,
    pool: Linear,
}
#[derive(Clone, Debug)]
pub struct BertModelOutput {
    pub last_hidden_state: Tensor,
    pub pooler_output: Tensor,
}
impl BertModel {
    /// Loads a Hugging Face-compatible BERT checkpoint on the selected device.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid configuration, missing or malformed
    /// tensors, or an unavailable device backend.
    pub fn from_pretrained(dir: impl AsRef<Path>, device: Device) -> Result<Self> {
        let p = dir.as_ref().join("config.json");
        let c: BertConfig = serde_json::from_slice(
            &fs::read(&p).map_err(|e| Error::io(format!("failed to read {}", p.display()), e))?,
        )
        .map_err(|e| Error::json("invalid BERT config", e))?;
        c.validate()?;
        Self::load(c, &SafeTensorLoader::open(dir)?, device)
    }

    /// Returns the loaded model configuration.
    #[must_use]
    pub const fn config(&self) -> &BertConfig {
        &self.config
    }
    fn load(c: BertConfig, l: &SafeTensorLoader, device: Device) -> Result<Self> {
        let p = if l.contains("bert.embeddings.word_embeddings.weight") {
            "bert."
        } else {
            ""
        };
        let n = |s: &str| format!("{p}{s}");
        let h = c.hidden_size;
        let mut layers = Vec::new();
        for i in 0..c.num_hidden_layers {
            let b = n(&format!("encoder.layer.{i}"));
            layers.push(Layer {
                q: Linear::load(l, &format!("{b}.attention.self.query"), h, h, device)?,
                k: Linear::load(l, &format!("{b}.attention.self.key"), h, h, device)?,
                v: Linear::load(l, &format!("{b}.attention.self.value"), h, h, device)?,
                attn: Linear::load(l, &format!("{b}.attention.output.dense"), h, h, device)?,
                an: Norm::load(
                    l,
                    &format!("{b}.attention.output.LayerNorm"),
                    h,
                    c.layer_norm_eps,
                    device,
                )?,
                up: Linear::load(
                    l,
                    &format!("{b}.intermediate.dense"),
                    h,
                    c.intermediate_size,
                    device,
                )?,
                down: Linear::load(
                    l,
                    &format!("{b}.output.dense"),
                    c.intermediate_size,
                    h,
                    device,
                )?,
                on: Norm::load(
                    l,
                    &format!("{b}.output.LayerNorm"),
                    h,
                    c.layer_norm_eps,
                    device,
                )?,
                heads: c.num_attention_heads,
            });
        }
        Ok(Self {
            word: param(
                l.load(&n("embeddings.word_embeddings.weight"))?,
                vec![c.vocab_size, h],
                device,
            )?,
            pos: param(
                l.load(&n("embeddings.position_embeddings.weight"))?,
                vec![c.max_position_embeddings, h],
                device,
            )?,
            types: param(
                l.load(&n("embeddings.token_type_embeddings.weight"))?,
                vec![c.type_vocab_size, h],
                device,
            )?,
            en: Norm::load(l, &n("embeddings.LayerNorm"), h, c.layer_norm_eps, device)?,
            layers,
            pool: Linear::load(l, &n("pooler.dense"), h, h, device)?,
            config: c,
        })
    }
    fn visit(&self, v: &mut dyn FnMut(&Parameter)) {
        v(&self.word);
        v(&self.pos);
        v(&self.types);
        self.en.visit(v);
        for x in &self.layers {
            x.visit(v)
        }
        self.pool.visit(v)
    }
    fn visit_mut(&mut self, v: &mut dyn FnMut(&mut Parameter)) {
        v(&mut self.word);
        v(&mut self.pos);
        v(&mut self.types);
        self.en.visit_mut(v);
        for x in &mut self.layers {
            x.visit_mut(v)
        }
        self.pool.visit_mut(v)
    }
}
impl Module<BertInput> for BertModel {
    type Output = BertModelOutput;
    fn forward(&self, i: &BertInput) -> Result<Self::Output> {
        let shape = i.input_ids.shape();
        if shape.len() != 2 {
            return Err(Error::InvalidShape("BERT ids must be rank 2".into()));
        }
        let (b, s, h) = (shape[0], shape[1], self.config.hidden_size);
        let device = i.input_ids.device();
        let mask = i
            .attention_mask
            .clone()
            .unwrap_or(Tensor::ones(vec![b, s])?.to(device));
        let types = i
            .token_type_ids
            .clone()
            .unwrap_or(Tensor::zeros(vec![b, s])?.to(device));
        let positions = Tensor::from_vec(
            (0..b).flat_map(|_| (0..s).map(|x| x as f32)).collect(),
            vec![b, s],
        )?
        .to(device);
        let mut x = ops::embedding(&i.input_ids, self.word.value(), h)?
            .add(&ops::embedding(&positions, self.pos.value(), h)?)?
            .add(&ops::embedding(&types, self.types.value(), h)?)?;
        x = self.en.f(&x)?;
        for layer in &self.layers {
            x = layer.f(&x, &mask)?;
        }
        let pooled = ops::tanh(&self.pool.f(&ops::select_first(&x, h)?)?)?;
        Ok(BertModelOutput {
            last_hidden_state: x,
            pooler_output: pooled,
        })
    }
}
#[derive(Clone, Debug)]
pub struct BertForSequenceClassification {
    bert: BertModel,
    classifier: Linear,
}
impl BertForSequenceClassification {
    /// Loads BERT and its sequence-classification head.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid configuration, missing or malformed
    /// tensors, or an unavailable device backend.
    pub fn from_pretrained(d: impl AsRef<Path>, dev: Device) -> Result<Self> {
        let d = d.as_ref();
        let bert = BertModel::from_pretrained(d, dev)?;
        let l = SafeTensorLoader::open(d)?;
        let classifier = if l.contains("classifier.weight") || l.contains("classifier.bias") {
            Linear::load(
                &l,
                "classifier",
                bert.config.hidden_size,
                bert.config.num_labels,
                dev,
            )?
        } else {
            Linear::initialized(
                bert.config.hidden_size,
                bert.config.num_labels,
                bert.config.initializer_range,
                dev,
            )?
        };
        Ok(Self { bert, classifier })
    }

    /// Saves the complete encoder and classification head as a SafeTensors checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an error if parameters cannot be copied to the host or files cannot be written.
    pub fn save_pretrained(&self, directory: impl AsRef<Path>) -> Result<()> {
        let directory = directory.as_ref();
        fs::create_dir_all(directory).map_err(|error| {
            Error::io(format!("failed to create {}", directory.display()), error)
        })?;
        let mut tensors = Vec::new();
        self.push_named_parameters(&mut tensors)?;
        let views = tensors
            .iter()
            .map(|(name, shape, bytes)| {
                Ok((
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        serialize_to_file(views, None, &directory.join("model.safetensors"))?;
        let config = serde_json::to_vec_pretty(&self.bert.config)
            .map_err(|error| Error::json("failed to serialize BERT config", error))?;
        fs::write(directory.join("config.json"), config).map_err(|error| {
            Error::io(
                format!("failed to write {}/config.json", directory.display()),
                error,
            )
        })
    }

    fn push_named_parameters(
        &self,
        tensors: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
    ) -> Result<()> {
        let mut push = |name: String, parameter: &Parameter| -> Result<()> {
            let values = parameter.value().to_vec()?;
            tensors.push((
                name,
                parameter.value().shape().to_vec(),
                values.into_iter().flat_map(f32::to_le_bytes).collect(),
            ));
            Ok(())
        };
        push(
            "bert.embeddings.word_embeddings.weight".into(),
            &self.bert.word,
        )?;
        push(
            "bert.embeddings.position_embeddings.weight".into(),
            &self.bert.pos,
        )?;
        push(
            "bert.embeddings.token_type_embeddings.weight".into(),
            &self.bert.types,
        )?;
        push("bert.embeddings.LayerNorm.weight".into(), &self.bert.en.w)?;
        push("bert.embeddings.LayerNorm.bias".into(), &self.bert.en.b)?;
        for (index, layer) in self.bert.layers.iter().enumerate() {
            let base = format!("bert.encoder.layer.{index}");
            for (suffix, linear) in [
                ("attention.self.query", &layer.q),
                ("attention.self.key", &layer.k),
                ("attention.self.value", &layer.v),
                ("attention.output.dense", &layer.attn),
                ("intermediate.dense", &layer.up),
                ("output.dense", &layer.down),
            ] {
                push(format!("{base}.{suffix}.weight"), &linear.w)?;
                push(format!("{base}.{suffix}.bias"), &linear.b)?;
            }
            push(
                format!("{base}.attention.output.LayerNorm.weight"),
                &layer.an.w,
            )?;
            push(
                format!("{base}.attention.output.LayerNorm.bias"),
                &layer.an.b,
            )?;
            push(format!("{base}.output.LayerNorm.weight"), &layer.on.w)?;
            push(format!("{base}.output.LayerNorm.bias"), &layer.on.b)?;
        }
        push("bert.pooler.dense.weight".into(), &self.bert.pool.w)?;
        push("bert.pooler.dense.bias".into(), &self.bert.pool.b)?;
        push("classifier.weight".into(), &self.classifier.w)?;
        push("classifier.bias".into(), &self.classifier.b)
    }
}
impl Module<BertInput> for BertForSequenceClassification {
    type Output = Tensor;
    fn forward(&self, i: &BertInput) -> Result<Tensor> {
        self.classifier.f(&self.bert.forward(i)?.pooler_output)
    }
}
impl Trainable for BertForSequenceClassification {
    fn visit_parameters(&self, v: &mut dyn FnMut(&Parameter)) {
        self.bert.visit(v);
        self.classifier.visit(v)
    }
    fn visit_parameters_mut(&mut self, v: &mut dyn FnMut(&mut Parameter)) {
        self.bert.visit_mut(v);
        self.classifier.visit_mut(v)
    }
}
fn param(t: LoadedTensor, shape: Vec<usize>, device: Device) -> Result<Parameter> {
    if t.shape != shape {
        return Err(Error::InvalidShape(format!(
            "weight shape {:?}, expected {shape:?}",
            t.shape
        )));
    }
    Ok(Parameter::new(Tensor::from_vec(t.data, shape)?.to(device)))
}
