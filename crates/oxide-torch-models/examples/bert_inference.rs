use oxide_torch::{Device, Error, Result};
use oxide_torch_models::bert::{BertForSequenceClassification, BertTokenizer};

fn main() -> Result<()> {
    let directory = std::env::args_os().nth(1).ok_or_else(|| {
        Error::Execution(
            "usage: cargo run -p oxide-torch-models --example bert_inference -- MODEL_DIR [TEXT]"
                .into(),
        )
    })?;
    let text = std::env::args().skip(2).collect::<Vec<_>>().join(" ");
    let tokenizer = BertTokenizer::from_pretrained(&directory)?;
    let model = BertForSequenceClassification::from_pretrained(&directory, Device::Cpu)?;
    let input_ids = tokenizer.encode(if text.is_empty() {
        "This library is pleasant to use."
    } else {
        &text
    })?;
    let logits = model.forward(&[input_ids], None, None)?.to_vec()?;
    let prediction = logits
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map_or(0, |(index, _)| index);
    println!("prediction={prediction} logits={logits:?}");
    Ok(())
}
