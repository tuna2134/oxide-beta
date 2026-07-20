# oxide-torch

NVLabs [`cuda-oxide`](https://github.com/NVlabs/cuda-oxide) を使う、小さな
Torch-like Rust MLライブラリのMVPです。

```rust
use oxide_torch::{Tensor, jit};

let x = Tensor::from_vec(vec![1., -2., 3., 4.], vec![2, 2])?;
let w = Tensor::from_vec(vec![1., 0., 0., 1.], vec![2, 2])?;
let model = jit::trace(&[x.clone(), w.clone()], |xs| {
    Ok(xs[0].matmul(&xs[1])?.relu())
})?;
let y = model.run(&[x, w])?;
println!("{:?}", y.to_vec()?);
# Ok::<(), oxide_torch::Error>(())
```

## 現在の機能

- 遅延 `Tensor` 計算グラフ
- CPUと `Device::Cuda(n)`
- `add`、`mul`、`relu`、rank-2 `matmul`
- `&a + &b` / `&a * &b` のTorch風演算子
- `jit::trace` による静的shapeトレースと特殊化キャッシュ
- cuda-oxideの純Rust `#[kernel]`、PTX埋め込み、CUDA Driver JITロード

## CPUで実行

このリポジトリはcuda-oxide公式指定のnightlyを固定しています。ツールチェーンが
まだない場合はインストールするか、CPUだけなら手元の安定版を明示できます。

```bash
cargo +stable test
cargo +stable run --example mlp
```

## CUDAで実行

cuda-oxideは通常の `cargo run` ではなく、専用codegen backendを使います。
Linux、NVIDIA GPU/driver、CUDA Toolkit、LLVM 22、固定nightlyが必要です。

```bash
cargo install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
cargo oxide doctor
cargo oxide run --features cuda -- --cuda
```

CUDA環境をNixで揃える場合は、cuda-oxide公式の `nix develop` も利用できます。
本プロジェクトは再現性のため依存をcuda-oxideの検証済みcommitへ固定しています。

## JITの意味

cuda-oxideは任意のRustソースを実行中にコンパイルするNVRTC風APIではありません。
ビルド時にRustカーネルをPTXへ変換し、実行時にCUDA Driverが対象GPU向けにJITします。
oxide-torch側ではさらに、最初の `run` でトレース済みグラフを入力shape/deviceに
特殊化してキャッシュします。

これはMVPであり、autograd、broadcast、mixed precision、最適化済みGEMM、
シリアライズは今後の拡張点です。

## MobileNetV4テストモデル

同梱された `mobilenet.pdf` はMobileNetV4論文の補足資料で、表11〜15に
各モデルの完全なアーキテクチャが記載されています。現在は最小の検証対象として
表11の `MNv4-Conv-S` を実装しています。

```bash
cargo +stable run --example mobilenet_v4
```

追加された演算はNCHW grouped `conv2d`、depthwise convolution、average/global
average pooling、reshapeです。MobileNetV4側にはFusedIBとUIBのIB、ConvNext、
ExtraDW構成、残差接続、分類headが含まれます。テストは19行すべてのshapeを
表11と照合します。

```rust
use oxide_torch::models::mobilenet_v4::MobileNetV4ConvSmall;
use oxide_torch::{Device, Tensor};

let model = MobileNetV4ConvSmall::new(1000, Device::Cpu)?;
let image = Tensor::zeros(vec![1, 3, 224, 224])?;
let (logits, shapes) = model.forward_with_shapes(&image)?;
assert_eq!(logits.shape(), &[1, 1000]);
# Ok::<(), oxide_torch::Error>(())
```

パラメータはテスト用にゼロ初期化されています。実際の学習済み推論では
`Conv2d::from_tensors` で重みを設定し、BatchNormを畳み込みのweight/biasへ
foldしてロードします。Hybrid-M/LのMobile-MQA、学習、checkpointローダーは
まだ含まれません。
