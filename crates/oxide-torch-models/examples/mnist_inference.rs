use oxide_torch::data::Mnist;
use oxide_torch::nn::Module;
use oxide_torch::{Device, Result};
use oxide_torch_models::mobilenet_v4::MobileNetV4ConvSmall;
use std::path::PathBuf;

fn main() -> Result<()> {
    let use_cuda =
        std::env::args().any(|value| value == "--cuda") || environment_flag("OXIDE_TORCH_CUDA");
    let device = if use_cuda {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    let data_dir = std::env::var_os("MNIST_DATA_DIR")
        .map_or_else(|| PathBuf::from("data/mnist"), PathBuf::from);
    let checkpoint = std::env::var_os("MNIST_CHECKPOINT")
        .map_or_else(|| PathBuf::from("mobilenetv4-mnist.oxtr"), PathBuf::from);
    let batch_size = environment_usize("MNIST_INFERENCE_BATCH_SIZE", 128);
    let limit = environment_usize("MNIST_INFERENCE_LIMIT", 10_000);

    let mnist = Mnist::load(data_dir)?;
    let model = MobileNetV4ConvSmall::load_mnist(&checkpoint, device)?;
    println!(
        "MNIST inference: checkpoint={} device={device:?}",
        checkpoint.display()
    );

    let mut correct = 0;
    let mut seen = 0;
    for batch in mnist
        .test_batches(batch_size)?
        .take(limit.div_ceil(batch_size))
    {
        let (images, labels) = batch?;
        let label_values = labels.to_vec()?;
        let logits = model.forward(&images.to(device))?.to_vec()?;
        for (index, label) in label_values.iter().enumerate() {
            let row = &logits[index * 10..(index + 1) * 10];
            let prediction = row
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map_or(0, |(class, _)| class);
            let expected = label_to_usize(*label);
            correct += usize::from(prediction == expected);
            if seen + index < 10 {
                println!(
                    "sample={} prediction={prediction} label={expected}",
                    seen + index
                );
            }
        }
        seen += label_values.len();
        if seen >= limit {
            break;
        }
    }
    println!(
        "accuracy={:.2}% samples={seen}",
        100.0 * as_f32(correct) / as_f32(seen)
    );
    Ok(())
}

fn environment_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn environment_flag(name: &str) -> bool {
    std::env::var(name)
        .is_ok_and(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn label_to_usize(value: f32) -> usize {
    value as usize
}

#[allow(clippy::cast_precision_loss)]
fn as_f32(value: usize) -> f32 {
    value as f32
}
