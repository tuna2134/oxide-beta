#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export MNIST_DATA_DIR="${MNIST_DATA_DIR:-${PROJECT_DIR}/data/mnist}"
export MNIST_CHECKPOINT="${MNIST_CHECKPOINT:-${PROJECT_DIR}/mobilenetv4-mnist.oxtr}"
cd -- "${PROJECT_DIR}"

if [[ "${1:-}" == "--cuda" ]]; then
    export OXIDE_TORCH_CUDA=1
    exec cargo oxide run --features cuda,cudnn --bin mnist-inference
fi
exec cargo +stable run --release --example mnist_inference
