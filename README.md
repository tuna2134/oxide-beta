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
- NCHW `conv2d`、group/depthwise convolution、average pooling
- `&a + &b` / `&a * &b` のTorch風演算子
- `jit::trace` による静的shapeトレースと特殊化キャッシュ
- reverse-mode autograd、cross entropy、`Parameter`、AdamW
- MNIST IDX loader、MobileNetV4 Conv-SのMNIST学習example、checkpoint保存
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
./scripts/train_mnist.sh --cuda
```

CUDA環境をNixで揃える場合は、cuda-oxide公式の `nix develop` も利用できます。
本プロジェクトは再現性のため依存をcuda-oxideの検証済みcommitへ固定しています。

## JITの意味

cuda-oxideは任意のRustソースを実行中にコンパイルするNVRTC風APIではありません。
ビルド時にRustカーネルをPTXへ変換し、実行時にCUDA Driverが対象GPU向けにJITします。
oxide-torch側ではさらに、最初の `run` でトレース済みグラフを入力shape/deviceに
特殊化してキャッシュします。

これはMVPであり、broadcast、mixed precision、最適化済みGEMM、checkpointロードは
BatchNorm2dは学習可能なγ/β、running mean/variance、train/eval切替、autogradに
対応します。CUDAではforward、running statistics更新、input/γ/β backwardをGPU
kernelで実行します。backwardのchannel reductionを一度だけ行い、各要素から同じ
reductionを繰り返さない実装です。

CUDA executor、JIT module、streamはdeviceごとに一度だけ作られます。Tensor nodeの
device buffer、勾配、AdamWの1次・2次moment、更新後parameterもGPU上に保持され、
`item()`、`to_vec()`、checkpoint保存など明示的にhost値が必要な箇所だけ同期します。

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

パラメータは再現可能なvariance-scaled初期値です。推論用の学習済み重みでは
`Conv2d::from_tensors` で設定し、BatchNormを畳み込みのweight/biasへfoldできます。
Hybrid-M/LのMobile-MQAとcheckpointローダーはまだ含まれません。

## MNISTでMobileNetV4を学習

`MobileNetV4ConvSmall::mnist` は入力をgrayscale `1x28x28`、分類headを10クラスに
した学習用variantです。次のscriptがMNISTのダウンロード、SHA-256検証、展開、学習を
まとめて実行します。

```bash
./scripts/train_mnist.sh
```

データだけ準備する場合やCUDAで学習する場合も指定できます。

```bash
./scripts/train_mnist.sh --download-only
./scripts/train_mnist.sh --cuda
./scripts/train_mnist.sh --data-dir /path/to/mnist
```

展開後はIDXファイル4個が同じディレクトリに置かれます。

```text
data/mnist/
├── train-images-idx3-ubyte
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte
└── t10k-labels-idx1-ubyte
```

scriptを使わず、準備済みデータから直接exampleを起動することもできます。第1引数を
省略すると `data/mnist` を使います。

```bash
cargo +stable run --release --example mnist_training -- data/mnist
```

参照実装の畳み込みは最適化前なので、既定値は1 epoch、batch size 2、学習4枚、
評価8枚です。BatchNorm学習では各channelに2値以上必要なため、batch sizeを1に
しないでください。
環境変数で通常の学習量へ拡張できます。

```bash
MNIST_EPOCHS=10 \
MNIST_BATCH_SIZE=32 \
MNIST_TRAIN_LIMIT=60000 \
MNIST_TEST_LIMIT=10000 \
cargo +stable run --release --example mnist_training -- data/mnist
```

CUDA環境ではscriptへ `--cuda` を付けます。直接起動する場合は、cuda-oxideの`run`が
アプリ引数を転送しないため環境変数を使います。

```bash
OXIDE_TORCH_CUDA=1 \
MNIST_DATA_DIR=data/mnist \
cargo oxide run --features cuda --bin mnist-training
```

終了時に`mobilenetv4-mnist.oxtr`を保存します。checkpointには学習parameterに加えて
BatchNormのrunning mean/varianceも含まれます。
