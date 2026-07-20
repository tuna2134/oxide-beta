use oxide_torch::models::mobilenet_v4::MobileNetV4ConvSmall;
use oxide_torch::{Device, Result, Tensor};

fn main() -> Result<()> {
    let model = MobileNetV4ConvSmall::new(1000, Device::Cpu)?;
    let image = Tensor::zeros(vec![1, 3, 224, 224])?;
    let (logits, shapes) = model.forward_with_shapes(&image)?;

    println!("MobileNetV4-Conv-S (supplement table 11)");
    for (index, shape) in shapes.iter().enumerate() {
        println!("  row {:02}: {shape:?}", index + 1);
    }
    println!("logits: {:?}", logits.shape());
    Ok(())
}
