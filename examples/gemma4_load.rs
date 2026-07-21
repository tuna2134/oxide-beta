use oxide_torch::Device;
use oxide_torch::models::gemma4::{Gemma4ForCausalLM, Gemma4Tokenizer};
#[cfg(feature = "cuda")]
use oxide_torch::models::gemma4::{GenerationConfig, sample_token, sample_topk_candidates};

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
    let cuda_prepare_started = std::time::Instant::now();
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
        let profile = std::env::var_os("GEMMA4_PROFILE").is_some();
        let verbose = profile || std::env::var_os("GEMMA4_VERBOSE").is_some();
        if profile {
            eprintln!(
                "Gemma4 profile prepare_cuda={:.3}s",
                cuda_prepare_started.elapsed().as_secs_f64()
            );
        }
        println!(
            "CUDA persistent weights: tensors={} bytes={} MiB",
            cuda.weight_count(),
            cuda.weight_bytes() / (1024 * 1024),
        );
        if std::env::var_os("GEMMA4_DIAGNOSTICS").is_some() {
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
        }
        let max_new_tokens = std::env::var("GEMMA4_MAX_NEW_TOKENS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(128);
        let max_sequence = token_ids.len().checked_add(max_new_tokens).ok_or_else(|| {
            oxide_torch::Error::Execution("Gemma4 maximum sequence length overflow".into())
        })?;
        let mut cache_table = cuda.new_cache_table(model.config(), max_sequence)?;
        println!(
            "CUDA 35-layer cache table: layers={} physical={} shared={} last_source={:?}",
            cache_table.layer_count(),
            cache_table.physical_cache_count(),
            cache_table.shared_layer_count(),
            cache_table.source_layer(model.config().num_hidden_layers - 1),
        );
        if std::env::var_os("GEMMA4_SKIP_DECODE").is_none() {
            let mut logits = Vec::new();
            let prefill_started = std::time::Instant::now();
            let batched_prefill = std::env::var_os("GEMMA4_SEQUENTIAL_PREFILL").is_none();
            if batched_prefill {
                logits = cuda.prefill_prompt(&token_ids, model.config(), &mut cache_table)?;
                if verbose {
                    eprintln!(
                        "Gemma4 prefill: {}/{} (batched)",
                        token_ids.len(),
                        token_ids.len()
                    );
                }
            } else {
                for (index, &token) in token_ids.iter().enumerate() {
                    if index + 1 == token_ids.len() {
                        logits = cuda.decode_token(token, model.config(), &mut cache_table)?;
                    } else {
                        cuda.prefill_token(token, model.config(), &mut cache_table)?;
                    }
                    if verbose {
                        eprintln!("Gemma4 prefill: {}/{}", index + 1, token_ids.len());
                    }
                }
            }
            if profile {
                cuda.synchronize()?;
                eprintln!(
                    "Gemma4 profile prefill_tokens={} seconds={:.3} ms_per_token={:.3}",
                    token_ids.len(),
                    prefill_started.elapsed().as_secs_f64(),
                    prefill_started.elapsed().as_secs_f64() * 1000.0 / token_ids.len() as f64,
                );
            }
            let mut generation = GenerationConfig {
                max_new_tokens,
                ..GenerationConfig::default()
            };
            if let Ok(value) = std::env::var("GEMMA4_TEMPERATURE") {
                generation.temperature = value.parse().map_err(|_| {
                    oxide_torch::Error::Execution("invalid GEMMA4_TEMPERATURE".into())
                })?;
            }
            if let Ok(value) = std::env::var("GEMMA4_TOP_K") {
                generation.top_k = value
                    .parse()
                    .map_err(|_| oxide_torch::Error::Execution("invalid GEMMA4_TOP_K".into()))?;
            }
            if let Ok(value) = std::env::var("GEMMA4_TOP_P") {
                generation.top_p = value
                    .parse()
                    .map_err(|_| oxide_torch::Error::Execution("invalid GEMMA4_TOP_P".into()))?;
            }
            if let Ok(value) = std::env::var("GEMMA4_SEED") {
                generation.seed = value
                    .parse()
                    .map_err(|_| oxide_torch::Error::Execution("invalid GEMMA4_SEED".into()))?;
            } else {
                generation.seed = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(generation.seed, |duration| {
                        duration.as_secs() ^ u64::from(duration.subsec_nanos())
                    });
            }
            let repetition_penalty = std::env::var("GEMMA4_REPETITION_PENALTY")
                .ok()
                .map_or(Ok(1.1), |value| value.parse::<f32>())
                .map_err(|_| {
                    oxide_torch::Error::Execution("invalid GEMMA4_REPETITION_PENALTY".into())
                })?;
            if repetition_penalty < 1.0 || !repetition_penalty.is_finite() {
                return Err(oxide_torch::Error::Execution(
                    "GEMMA4_REPETITION_PENALTY must be finite and at least 1".into(),
                ));
            }
            let mut random = generation.seed;
            let mut generated = Vec::with_capacity(generation.max_new_tokens);
            let mut streamed = String::new();
            let mut topk_candidates: Option<Vec<(u32, f32)>> = None;
            let generation_started = std::time::Instant::now();
            let mut decode_seconds = 0.0_f64;
            let mut sampling_seconds = 0.0_f64;
            const TEXT_STOP_MARKERS: [&str; 4] =
                ["<turn|>", "<|turn>", "<end_of_turn>", "<start_of_turn>"];
            for index in 0..generation.max_new_tokens {
                if topk_candidates.is_none() && repetition_penalty > 1.0 {
                    let mut penalized = std::collections::HashSet::with_capacity(generated.len());
                    for &token in &generated {
                        if !penalized.insert(token) {
                            continue;
                        }
                        if let Some(logit) = logits.get_mut(token as usize) {
                            if *logit >= 0.0 {
                                *logit /= repetition_penalty;
                            } else {
                                *logit *= repetition_penalty;
                            }
                        }
                    }
                }
                let sampling_started = std::time::Instant::now();
                let next = if let Some(candidates) = &topk_candidates {
                    sample_topk_candidates(candidates, &generation, &mut random)?
                } else {
                    sample_token(&logits, &generation, &mut random)?
                };
                sampling_seconds += sampling_started.elapsed().as_secs_f64();
                if generation.eos_token_ids.contains(&next) {
                    break;
                }
                generated.push(next);
                let piece = tokenizer.decode(&[next], true)?;
                streamed.push_str(&piece);
                if verbose {
                    println!(
                        "Gemma4 generate: {}/{} token={} piece={:?}",
                        index + 1,
                        generation.max_new_tokens,
                        next,
                        piece,
                    );
                }
                if TEXT_STOP_MARKERS
                    .iter()
                    .any(|marker| streamed.ends_with(marker))
                {
                    break;
                }
                let decode_started = std::time::Instant::now();
                topk_candidates = Some(cuda.decode_topk(
                    next,
                    model.config(),
                    &mut cache_table,
                    generation.top_k,
                    repetition_penalty,
                    true,
                )?);
                decode_seconds += decode_started.elapsed().as_secs_f64();
            }
            if profile {
                let elapsed = generation_started.elapsed().as_secs_f64();
                eprintln!(
                    "Gemma4 profile generation_tokens={} seconds={elapsed:.3} decode={decode_seconds:.3} sampling={sampling_seconds:.3} tokens_per_second={:.3}",
                    generated.len(),
                    generated.len() as f64 / elapsed.max(f64::EPSILON),
                );
            }
            let mut response = tokenizer.decode(&generated, true)?;
            for marker in TEXT_STOP_MARKERS {
                if let Some(stripped) = response.strip_suffix(marker) {
                    response = stripped.to_owned();
                    break;
                }
            }
            println!("CUDA Gemma4 response: {response}");
        }
    }
    Ok(())
}
