//! Fine-tunes the complete BERT encoder and sequence-classification head.

use oxide_torch::loss::cross_entropy;
use oxide_torch::nn::Module;
use oxide_torch::optim::{AdamW, Optimizer};
use oxide_torch::{Device, Error, Result, Tensor};
use oxide_torch_models::bert::{BertForSequenceClassification, BertInput, BertTokenizer};

fn main() -> Result<()> {
    let directory = std::env::args_os().nth(1).ok_or_else(|| {
        Error::Execution(
            "usage: cargo run -p oxide-torch-models --example bert_finetune -- MODEL_DIR".into(),
        )
    })?;
    let tokenizer = BertTokenizer::from_pretrained(&directory)?;
    let mut model = BertForSequenceClassification::from_pretrained(&directory, Device::Cpu)?;
    let samples = [
        ("I loved this movie.", 1.0),
        ("A delightful and moving story.", 1.0),
        ("I disliked this movie.", 0.0),
        ("A dull and frustrating story.", 0.0),
    ];
    let mut encoded = samples
        .iter()
        .map(|(text, _)| tokenizer.encode(text))
        .collect::<Result<Vec<_>>>()?;
    let sequence = encoded.iter().map(Vec::len).max().unwrap_or(1);
    let masks: Vec<Vec<u8>> = encoded
        .iter()
        .map(|tokens| {
            let mut mask = vec![1; tokens.len()];
            mask.resize(sequence, 0);
            mask
        })
        .collect();
    for tokens in &mut encoded {
        tokens.resize(sequence, 0);
    }
    let labels = Tensor::from_vec(
        samples.iter().map(|(_, label)| *label).collect(),
        vec![samples.len()],
    )?;
    let input = BertInput::from_ids(&encoded, Some(&masks), None)?;
    let mut optimizer = AdamW::new(2e-4, 0.01)?;

    for epoch in 1..=5 {
        optimizer.zero_grad(&model)?;
        let logits = model.forward(&input)?;
        let loss = cross_entropy(&logits, &labels)?;
        loss.backward()?;
        optimizer.step(&mut model)?;
        println!("epoch={epoch} loss={:.5}", loss.item()?);
    }
    Ok(())
}
