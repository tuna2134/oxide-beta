#[cfg(feature = "cuda")]
use oxide_torch::Device;
#[cfg(feature = "cuda")]
use oxide_torch_models::gemma4::{Gemma4ForCausalLM, Gemma4Tokenizer, GenerationConfig};

#[cfg(feature = "cuda")]
fn main() -> oxide_torch::Result<()> {
    use std::io::Write;

    let directory = std::env::args_os().nth(1).ok_or_else(|| {
        oxide_torch::Error::Execution("usage: gemma4_stream MODEL_DIR [PROMPT ...]".into())
    })?;
    let prompt = std::env::args().skip(2).collect::<Vec<_>>().join(" ");
    let tokenizer = Gemma4Tokenizer::from_pretrained(&directory)?;
    let input = tokenizer.encode_user_turn(if prompt.is_empty() {
        "Rustについて短く説明してください。"
    } else {
        &prompt
    })?;
    let model = Gemma4ForCausalLM::from_pretrained(&directory, Device::Cuda(0))?;
    let mut decoder = tokenizer.decode_stream(true);
    model.generate_stream(
        &input,
        &GenerationConfig {
            max_new_tokens: 256,
            ..GenerationConfig::default()
        },
        |token_id| {
            if let Some(chunk) = decoder.step(token_id)? {
                print!("{chunk}");
                std::io::stdout().flush().map_err(|error| {
                    oxide_torch::Error::Execution(format!("stdout flush failed: {error}"))
                })?;
            }
            Ok(())
        },
    )?;
    println!();
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("gemma4_stream requires --features cuda");
}
