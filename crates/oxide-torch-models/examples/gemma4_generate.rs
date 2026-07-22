#[cfg(feature = "cuda")]
use oxide_torch::Device;
#[cfg(feature = "cuda")]
use oxide_torch_models::gemma4::{Gemma4ForCausalLM, Gemma4Tokenizer, GenerationConfig};

#[cfg(feature = "cuda")]
fn main() -> oxide_torch::Result<()> {
    let directory = std::env::args_os().nth(1).ok_or_else(|| {
        oxide_torch::Error::Execution("usage: gemma4_generate MODEL_DIR [PROMPT ...]".into())
    })?;
    let prompt = std::env::args().skip(2).collect::<Vec<_>>().join(" ");
    let tokenizer = Gemma4Tokenizer::from_pretrained(&directory)?;
    let input = tokenizer.encode_user_turn(if prompt.is_empty() {
        "こんにちは。自己紹介してください。"
    } else {
        &prompt
    })?;
    let model = Gemma4ForCausalLM::from_pretrained(&directory, Device::Cuda(0))?;
    let output = model.generate(
        &input,
        &GenerationConfig {
            max_new_tokens: 256,
            ..GenerationConfig::default()
        },
    )?;
    println!("{}", tokenizer.decode(&output[input.len()..], true)?);
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("gemma4_generate requires --features cuda");
}
