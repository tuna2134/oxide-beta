use oxide_torch::data::Mnist;
use oxide_torch::loss::cross_entropy;
use oxide_torch::nn::Module;
use oxide_torch::optim::{AdamW, Optimizer};
use oxide_torch::{Device, Result, Tensor};
use oxide_torch_models::mobilenet_v4::MobileNetV4ConvSmall;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[allow(clippy::too_many_lines)]
fn main() -> Result<()> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let use_cuda = arguments.iter().any(|argument| argument == "--cuda")
        || environment_flag("OXIDE_TORCH_CUDA");
    let data_directory = arguments
        .iter()
        .find(|argument| !argument.starts_with("--"))
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("MNIST_DATA_DIR").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("data/mnist"));
    let device = if use_cuda {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    let epochs = environment_usize("MNIST_EPOCHS", 1);
    // The reference kernels favor clarity over speed. These defaults are deliberately
    // small so the complete example is practical on a CPU; raise them for a real run.
    let batch_size = environment_usize("MNIST_BATCH_SIZE", 2);
    let train_limit = environment_usize("MNIST_TRAIN_LIMIT", 4);
    let test_limit = environment_usize("MNIST_TEST_LIMIT", 8);
    let log_interval = environment_usize("MNIST_LOG_INTERVAL", if use_cuda { 50 } else { 1 });
    let profile = environment_flag("MNIST_PROFILE");
    let mut timings = Timings::default();

    let mnist = Mnist::load(&data_directory)?;
    let mut model = MobileNetV4ConvSmall::mnist(device)?;
    model.train();
    let mut optimizer = AdamW::new(1e-3, 1e-4)?;

    println!(
        "MNIST: train={} test={} device={device:?}",
        mnist.train_len(),
        mnist.test_len()
    );

    for epoch in 1..=epochs {
        let mut total_loss = 0.0;
        let mut correct = 0;
        let mut measured = 0;
        let mut measurements = 0;
        let mut seen = 0;
        let maximum_batches = train_limit.div_ceil(batch_size);

        let mut batches = mnist
            .train_batches(batch_size, true)?
            .take(maximum_batches)
            .enumerate();
        loop {
            let started = Instant::now();
            let Some((batch_index, batch)) = batches.next() else {
                break;
            };
            let (images, labels) = batch?;
            timings.data += started.elapsed();
            let started = Instant::now();
            let images = images.to(device);
            let labels = labels.to(device);
            if profile {
                images.synchronize()?;
                labels.synchronize()?;
            }
            timings.transfer += started.elapsed();

            optimizer.zero_grad(&model)?;
            let started = Instant::now();
            let logits = model.forward(&images)?;
            let loss = cross_entropy(&logits, &labels)?;
            timings.graph += started.elapsed();
            if profile {
                let started = Instant::now();
                loss.synchronize()?;
                timings.forward += started.elapsed();
            }
            let started = Instant::now();
            loss.backward()?;
            if profile {
                loss.synchronize()?;
            }
            timings.backward += started.elapsed();
            let started = Instant::now();
            optimizer.step(&mut model)?;
            if profile {
                loss.synchronize()?;
            }
            timings.optimizer += started.elapsed();

            seen += labels.shape()[0];
            let should_log = (batch_index + 1) % log_interval == 0 || seen >= train_limit;
            if should_log {
                let started = Instant::now();
                // Reading metrics synchronizes the CUDA stream. Keeping this
                // out of the hot path lets several training steps stay queued.
                let batch_loss = loss.item()?;
                let batch_correct = count_correct(&logits, &labels)?;
                total_loss += batch_loss;
                correct += batch_correct;
                measured += labels.shape()[0];
                measurements += 1;
                println!(
                    "epoch={epoch} samples={seen}/{train_limit} loss={batch_loss:.4} accuracy={:.2}%",
                    100.0 * as_f32(batch_correct) / as_f32(labels.shape()[0])
                );
                timings.metrics += started.elapsed();
            }
            if seen >= train_limit {
                break;
            }
        }
        println!(
            "epoch={epoch} sampled_loss={:.4} sampled_accuracy={:.2}% measured={measured}/{seen}",
            total_loss / as_f32(measurements),
            100.0 * as_f32(correct) / as_f32(measured)
        );
    }

    model.eval();
    let evaluation_started = Instant::now();
    let mut correct = 0;
    let mut seen = 0;
    let evaluation_batch_size = batch_size.max(16).min(test_limit);
    for batch in mnist
        .test_batches(evaluation_batch_size)?
        .take(test_limit.div_ceil(evaluation_batch_size))
    {
        let (images, labels) = batch?;
        let logits = model.forward(&images.to(device))?;
        correct += count_correct(&logits, &labels.to(device))?;
        seen += labels.shape()[0];
        if seen >= test_limit {
            break;
        }
    }
    println!(
        "test_accuracy={:.2}%",
        100.0 * as_f32(correct) / as_f32(seen)
    );
    timings.evaluation += evaluation_started.elapsed();
    let started = Instant::now();
    model.save("mobilenetv4-mnist.oxtr")?;
    timings.checkpoint += started.elapsed();
    println!("saved mobilenetv4-mnist.oxtr");
    if profile {
        timings.print();
    }
    Ok(())
}

#[derive(Default)]
struct Timings {
    data: Duration,
    transfer: Duration,
    graph: Duration,
    forward: Duration,
    backward: Duration,
    optimizer: Duration,
    metrics: Duration,
    evaluation: Duration,
    checkpoint: Duration,
}

impl Timings {
    fn print(&self) {
        let phases = [
            ("data", self.data),
            ("h2d", self.transfer),
            ("graph", self.graph),
            ("forward", self.forward),
            ("backward", self.backward),
            ("optimizer", self.optimizer),
            ("metrics/d2h/sync", self.metrics),
            ("evaluation", self.evaluation),
            ("checkpoint", self.checkpoint),
        ];
        let total: Duration = phases.iter().map(|(_, duration)| *duration).sum();
        println!("profile_total={:.3}s", total.as_secs_f64());
        for (name, duration) in phases {
            let percent = if total.is_zero() {
                0.0
            } else {
                100.0 * duration.as_secs_f64() / total.as_secs_f64()
            };
            println!(
                "profile_phase={name} seconds={:.3} percent={percent:.1}%",
                duration.as_secs_f64()
            );
        }
    }
}

fn count_correct(logits: &Tensor, labels: &Tensor) -> Result<usize> {
    let logits = logits.to_vec()?;
    let labels = labels.to_vec()?;
    let classes = 10;
    Ok(labels
        .iter()
        .enumerate()
        .filter(|(batch, label)| {
            let row = &logits[*batch * classes..(*batch + 1) * classes];
            let prediction = row
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map_or(0, |(index, _)| index);
            prediction == label_to_usize(**label)
        })
        .count())
}

#[allow(clippy::cast_precision_loss)]
fn as_f32(value: usize) -> f32 {
    value as f32
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn label_to_usize(value: f32) -> usize {
    value as usize
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
