use oxide_torch::{Device, Result, Tensor, jit};

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

    let model = jit::trace(&[x.clone(), weight.clone(), bias.clone()], |inputs| {
        Ok(inputs[0].matmul(&inputs[1])?.add(&inputs[2])?.relu())
    })?;

    let output = model.run(&[x, weight, bias])?;
    println!("device: {:?}", output.device());
    println!("shape:  {:?}", output.shape());
    println!("output: {:?}", output.to_vec()?);
    println!("JIT specializations: {}", model.cached_specializations());
    Ok(())
}
