mod bert_support;

use oxide_torch::nn::Module;
use oxide_torch::{Error, Result};
use oxide_torch_models::bert::{BertForSequenceClassification, BertInput, BertTokenizer};

fn main() -> Result<()> {
    let directory = std::env::args_os().nth(1).ok_or_else(|| {
        Error::Execution(
            "usage: cargo run -p oxide-torch-models --example bert_inference -- \
             MODEL_DIR [--device cpu|cuda[:INDEX]] [TEXT]"
                .into(),
        )
    })?;
    let mut arguments = std::env::args().skip(2).collect::<Vec<_>>();
    let device = bert_support::take_device(&mut arguments)?;
    let text = arguments.join(" ");
    let tokenizer = BertTokenizer::from_pretrained(&directory)?;
    let model = BertForSequenceClassification::from_pretrained(&directory, device)?;
    let input_ids = tokenizer.encode(if text.is_empty() {
        "This library is pleasant to use."
    } else {
        &text
    })?;
    let input = BertInput::from_ids(&[input_ids], None, None)?.to(device);
    let logits = model.forward(&input)?.to_vec()?;
    let prediction = logits
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map_or(0, |(index, _)| index);
    println!("device={device:?} prediction={prediction} logits={logits:?}");
    Ok(())
}
