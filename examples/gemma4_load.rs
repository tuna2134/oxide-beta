use oxide_torch::Device;
use oxide_torch::models::gemma4::{Gemma4ForCausalLM, Gemma4Tokenizer};
#[cfg(feature = "cuda")]
use oxide_torch::models::gemma4::{GenerationConfig, sample_token};

#[allow(clippy::too_many_lines)]
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
        let attention = cuda.decoder_attention_smoke(
            *token_ids
                .last()
                .ok_or_else(|| oxide_torch::Error::Execution("empty prompt".into()))?,
            0,
            model.config().hidden_size,
            model.config().num_attention_heads,
            model.config().num_key_value_heads,
            model.config().head_dim,
            model.config().rms_norm_eps,
            model.config().sliding_window,
        )?;
        let maximum = attention
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f32, f32::max);
        println!(
            "CUDA decoder layer 0 attention smoke: hidden={} abs_max={maximum:.4}",
            attention.len()
        );
        let mut cache = cuda.new_kv_cache(
            model.config().num_key_value_heads,
            model.config().head_dim,
            model.config().sliding_window,
        )?;
        for &token in token_ids.iter().rev().take(2).rev() {
            let output = cuda.cached_attention_smoke(
                token,
                0,
                model.config().hidden_size,
                model.config().num_attention_heads,
                model.config().rms_norm_eps,
                &mut cache,
            )?;
            if output.iter().any(|value| !value.is_finite()) {
                return Err(oxide_torch::Error::Execution(
                    "cached attention produced a non-finite value".into(),
                ));
            }
        }
        println!("CUDA persistent KV cache smoke: sequence={}", cache.len());
        let mut cache_table = cuda.new_cache_table(model.config(), token_ids.len() + 128)?;
        println!(
            "CUDA 35-layer cache table: layers={} physical={} shared={} last_source={:?}",
            cache_table.layer_count(),
            cache_table.physical_cache_count(),
            cache_table.shared_layer_count(),
            cache_table.source_layer(model.config().num_hidden_layers - 1),
        );
        if std::env::var_os("GEMMA4_SKIP_DECODE").is_none() {
            let mut logits = Vec::new();
            for (index, &token) in token_ids.iter().enumerate() {
                logits = cuda.decode_token(token, model.config(), &mut cache_table)?;
                eprintln!("Gemma4 prefill: {}/{}", index + 1, token_ids.len());
            }
            let generation = GenerationConfig::default();
            let mut random = generation.seed;
            let next = sample_token(&logits, &generation, &mut random)?;
            println!("CUDA 35-layer next_token={next}");
            println!(
                "CUDA 35-layer decoded={:?}",
                tokenizer.decode(&[next], true)?
            );
        }
    }
    Ok(())
}
