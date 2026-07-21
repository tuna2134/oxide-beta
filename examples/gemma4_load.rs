use oxide_torch::Device;
use oxide_torch::models::gemma4::{Gemma4ForCausalLM, Gemma4Tokenizer};

fn main() -> oxide_torch::Result<()> {
    let directory = std::env::args_os().nth(1).ok_or_else(|| {
        oxide_torch::Error::Execution(
            "usage: cargo run --release --example gemma4_load -- MODEL_DIR [TOKEN_ID ...]".into(),
        )
    })?;
    let prompt = std::env::args().skip(2).collect::<Vec<_>>().join(" ");
    let device = if std::env::var_os("OXIDE_TORCH_CUDA").is_some() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    let tokenizer = Gemma4Tokenizer::from_pretrained(&directory)?;
    let token_ids =
        tokenizer.encode_user_turn(if prompt.is_empty() { "Hello" } else { &prompt })?;
    let model = Gemma4ForCausalLM::from_pretrained(&directory, device)?;
    #[cfg(feature = "cuda")]
    let cuda = if matches!(device, Device::Cuda(_)) {
        Some(model.prepare_cuda()?)
    } else {
        None
    };
    let embeddings = model.embed(&token_ids)?;
    println!(
        "Gemma4 loaded: layers={} hidden={} vocab={}/{} tokens={} embeddings={:?} device={device:?}",
        model.config().num_hidden_layers,
        model.config().hidden_size,
        model.config().vocab_size,
        tokenizer.vocabulary_size(),
        token_ids.len(),
        embeddings.shape(),
    );
    #[cfg(feature = "cuda")]
    if let Some(cuda) = cuda {
        println!(
            "CUDA persistent weights: tensors={} bytes={} MiB",
            cuda.weight_count(),
            cuda.weight_bytes() / (1024 * 1024),
        );
        let logits = cuda.embedding_logits(
            *token_ids
                .last()
                .ok_or_else(|| oxide_torch::Error::Execution("empty prompt".into()))?,
            model.config().hidden_size,
        )?;
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        println!(
            "CUDA BF16 cuBLAS smoke: logits={} max={maximum:.4}",
            logits.len()
        );
        let mlp = cuda.decoder_mlp_smoke(
            *token_ids
                .last()
                .ok_or_else(|| oxide_torch::Error::Execution("empty prompt".into()))?,
            0,
            model.config().hidden_size,
            model.config().rms_norm_eps,
        )?;
        let maximum = mlp.iter().map(|value| value.abs()).fold(0.0_f32, f32::max);
        println!(
            "CUDA decoder layer 0 MLP smoke: hidden={} abs_max={maximum:.4}",
            mlp.len()
        );
    }
    Ok(())
}
