use oxide_torch::Device;
use oxide_torch::models::gemma4::Gemma4ForCausalLM;

fn main() -> oxide_torch::Result<()> {
    let directory = std::env::args_os().nth(1).ok_or_else(|| {
        oxide_torch::Error::Execution(
            "usage: cargo run --release --example gemma4_load -- MODEL_DIR [TOKEN_ID ...]".into(),
        )
    })?;
    let token_ids = std::env::args()
        .skip(2)
        .map(|value| {
            value.parse::<u32>().map_err(|error| {
                oxide_torch::Error::Execution(format!("invalid token id `{value}`: {error}"))
            })
        })
        .collect::<oxide_torch::Result<Vec<_>>>()?;
    let token_ids = if token_ids.is_empty() {
        vec![2]
    } else {
        token_ids
    };
    let device = if std::env::var_os("OXIDE_TORCH_CUDA").is_some() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    let model = Gemma4ForCausalLM::from_pretrained(directory, device)?;
    let embeddings = model.embed(&token_ids)?;
    println!(
        "Gemma4 loaded: layers={} hidden={} vocab={} embeddings={:?} device={device:?}",
        model.config().num_hidden_layers,
        model.config().hidden_size,
        model.config().vocab_size,
        embeddings.shape(),
    );
    Ok(())
}
