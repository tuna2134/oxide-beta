use oxide_torch::nn::Module;
use oxide_torch::{Device, Result, Tensor, jit};

struct Mlp {
    weight: Tensor,
    bias: Tensor,
}

impl Module for Mlp {
    type Output = Tensor;

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.matmul(&self.weight)?.add(&self.bias)?.relu())
    }
}

fn main() -> Result<()> {
    let use_cuda = std::env::args().any(|argument| argument == "--cuda");
    let device = if use_cuda {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    let x = Tensor::from_vec(vec![1.0, -2.0, 3.0, 0.5], vec![2, 2])?.to(device);
    let weight = Tensor::from_vec(vec![0.5, 1.0, -1.0, 2.0], vec![2, 2])?.to(device);
    let bias = Tensor::from_vec(vec![0.25, 0.25, 0.25, 0.25], vec![2, 2])?.to(device);

    // User-defined models only implement `Module`; `jit::compile` traces their
    // lazy graph and creates a reusable fixed-memory execution plan.
    let model = Mlp { weight, bias };
    let compiled = jit::compile(&model, &x)?;

    let output = compiled.run(&[x])?;
    println!("device: {:?}", output.device());
    println!("shape:  {:?}", output.shape());
    println!("output: {:?}", output.to_vec()?);
    println!("JIT specializations: {}", compiled.cached_specializations());
    Ok(())
}
