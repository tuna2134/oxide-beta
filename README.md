# oxide-torch

NVLabs [`cuda-oxide`](https://github.com/NVlabs/cuda-oxide) を使う、小さな
Torch-like Rust MLライブラリのMVPです。

```rust
use oxide_torch::nn::Module;
use oxide_torch::{Result, Tensor, jit};

struct Model { weight: Tensor }

impl Module for Model {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.matmul(&self.weight)?.relu())
    }
}

let x = Tensor::from_vec(vec![1., -2., 3., 4.], vec![2, 2])?;
let w = Tensor::from_vec(vec![1., 0., 0., 1.], vec![2, 2])?;
let model = Model { weight: w };
let compiled = jit::compile(&model, &x)?;
let y = compiled.run(&[x])?;
println!("{:?}", y.to_vec()?);
# Ok::<(), oxide_torch::Error>(())
```

## 現在の機能

- 遅延 `Tensor` 計算グラフ
- CPUと `Device::Cuda(n)`
- `add`、`mul`、`relu`、rank-2 `matmul`
- NCHW `conv2d`、group/depthwise convolution、average pooling
- `&a + &b` / `&a * &b` のTorch風演算子
- `jit::compile` / `jit::trace` による固定実行プランと特殊化キャッシュ
- reverse-mode autograd、cross entropy、`Parameter`、AdamW
- MNIST IDX loader、MobileNetV4 Conv-SのMNIST学習example、checkpoint保存
- cuda-oxideの純Rust `#[kernel]`、PTX埋め込み、CUDA Driver JITロード

## Workspace構成

- `oxide-torch`: Tensor、autograd、`nn`、optimizer、データ、checkpoint、JIT、汎用CUDA backend
- `oxide-torch-models`: Gemma4、MobileNetV4、およびモデル専用CUDA実行コード

自作モデルや学習基盤だけが必要なら `oxide-torch` のみに依存できます。標準モデルを
利用する場合だけ `oxide-torch-models` を追加してください。依存方向は
`oxide-torch-models -> oxide-torch` の一方向です。

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

このリポジトリの既定CUDA targetはGoogle ColabのNVIDIA L4に合わせた
`sm_89`です。別GPUではビルド時にtargetを上書きしてください。

```bash
CUDA_OXIDE_TARGET=sm_80 cargo oxide build --arch sm_80 --features cuda -- \
  --example gemma4_load --release
```

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
特殊化し、演算順序と入出力・中間bufferを固定した実行プランとしてキャッシュします。
CUDAでは対応演算列全体を一度captureし、2回目以降は入力copyとCUDA Graph replayで
実行します。自作モデルは `nn::Module` を実装して `jit::compile(&model, &example)` に
渡せます。複数入力モデルには `jit::trace` を使用します。未対応演算は従来の遅延
executorへ自動的にfallbackします。

これはMVPであり、broadcast、mixed precision、最適化済みGEMM、checkpointロードは
BatchNorm2dは学習可能なγ/β、running mean/variance、train/eval切替、autogradに
対応します。CUDAではforward、running statistics更新、input/γ/β backwardをGPU
kernelで実行します。backwardのchannel reductionを一度だけ行い、各要素から同じ
reductionを繰り返さない実装です。

CUDA executor、JIT module、streamはdeviceごとに一度だけ作られます。Tensor nodeの
device buffer、勾配、AdamWの1次・2次moment、更新後parameterもGPU上に保持され、
`item()`、`to_vec()`、checkpoint保存など明示的にhost値が必要な箇所だけ同期します。

Conv2dのCUDA forward/input backwardは256-threadの空間tileでweightを共有メモリへ
cacheします。weight backwardは32-threadのweight tileで同一out-channelのoutput
gradientを共有し、通常畳み込みとgrouped/depthwise畳み込みの両方に適用します。

`cudnn` featureでは`libcudnn.so.9`、`.so.8`、または非versioned sonameを実行時に
検出します。利用可能ならConv2d forward、backward-data、backward-filter、
backward-biasを同じcuda-oxide streamとdevice bufferでゼロコピー実行します。
descriptorとworkspaceはshapeごとにcacheされます。ライブラリがない場合やoperationが
非対応の場合は上記のcuda-oxide tiled kernelへ自動的にfallbackします。

## MobileNetV4テストモデル

同梱された `mobilenet.pdf` はMobileNetV4論文の補足資料で、表11〜15に
各モデルの完全なアーキテクチャが記載されています。現在は最小の検証対象として
表11の `MNv4-Conv-S` を実装しています。

```bash
cargo +stable run -p oxide-torch-models --example mobilenet_v4
```

追加された演算はNCHW grouped `conv2d`、depthwise convolution、average/global
average pooling、reshapeです。MobileNetV4側にはFusedIBとUIBのIB、ConvNext、
ExtraDW構成、残差接続、分類headが含まれます。テストは19行すべてのshapeを
表11と照合します。

```rust
use oxide_torch::{Device, Tensor};
use oxide_torch_models::mobilenet_v4::MobileNetV4ConvSmall;

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
cargo +stable run -p oxide-torch-models --release --example mnist_training -- data/mnist
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
cargo +stable run -p oxide-torch-models --release --example mnist_training -- data/mnist
```

CUDAではGPU streamを毎batch停止しないよう、既定では50 batchごとにlossとaccuracyを
hostへ読み戻します。`MNIST_LOG_INTERVAL`で表示間隔を変更できます。小さい値ほど
表示は増えますが、GPU同期も増えて学習は遅くなります。

CUDA環境ではscriptへ `--cuda` を付けます。直接起動する場合は、cuda-oxideの`run`が
アプリ引数を転送しないため環境変数を使います。

```bash
OXIDE_TORCH_CUDA=1 \
MNIST_DATA_DIR=data/mnist \
cargo oxide run -p oxide-torch-models --features cuda,cudnn --bin mnist-training
```

終了時に`mobilenetv4-mnist.oxtr`を保存します。checkpointには学習parameterに加えて
BatchNormのrunning mean/varianceも含まれます。

## MNIST推論

保存済みcheckpointを読み込み、test setで推論します。

```bash
./scripts/infer_mnist.sh --cuda
```

対象数やbatch size、checkpointは環境変数で変更できます。

```bash
MNIST_CHECKPOINT=mobilenetv4-mnist.oxtr \
MNIST_INFERENCE_BATCH_SIZE=256 \
MNIST_INFERENCE_LIMIT=10000 \
./scripts/infer_mnist.sh --cuda
```

## CUDAプロファイル

ColabへNsight Systems (`nsys`) とNsight Compute (`ncu`)が入っている場合、学習phase、
CUDA API、memory allocation/copy、kernel、occupancy、帯域、registerを収集できます。

```bash
MNIST_BATCH_SIZE=256 \
MNIST_TRAIN_LIMIT=2500 \
MNIST_TEST_LIMIT=256 \
./scripts/profile_cuda.sh
```

結果は`profile-results/phase-timings.txt`、`nsys-summary.csv`、
`mobilenet-mnist.nsys-rep`、`mobilenet-conv.ncu-rep`へ保存されます。
