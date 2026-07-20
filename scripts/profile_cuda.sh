#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPORT_DIR="${PROFILE_REPORT_DIR:-${PROJECT_DIR}/profile-results}"
mkdir -p -- "${REPORT_DIR}"
cd -- "${PROJECT_DIR}"

for command in cargo nsys ncu; do
    command -v "${command}" >/dev/null 2>&1 || {
        echo "error: ${command} is required" >&2
        exit 1
    }
done

export OXIDE_TORCH_CUDA=1
export MNIST_DATA_DIR="${MNIST_DATA_DIR:-${PROJECT_DIR}/data/mnist}"
export MNIST_EPOCHS=1
export MNIST_BATCH_SIZE="${MNIST_BATCH_SIZE:-256}"
export MNIST_TRAIN_LIMIT="${MNIST_TRAIN_LIMIT:-2500}"
export MNIST_TEST_LIMIT="${MNIST_TEST_LIMIT:-256}"
export MNIST_LOG_INTERVAL="${MNIST_LOG_INTERVAL:-10}"

echo "profile: build and warm up CUDA/cuDNN"
MNIST_TRAIN_LIMIT=256 MNIST_TEST_LIMIT=16 cargo oxide run --features cuda,cudnn --bin mnist-training

readonly BINARY="${PROJECT_DIR}/target/release/mnist-training"
[[ -x "${BINARY}" ]] || { echo "error: binary not found: ${BINARY}" >&2; exit 1; }

echo "profile: host phase timings"
MNIST_PROFILE=1 "${BINARY}" | tee "${REPORT_DIR}/phase-timings.txt"

echo "profile: Nsight Systems timeline"
nsys profile \
    --force-overwrite=true \
    --trace=cuda,osrt \
    --cuda-memory-usage=true \
    --sample=none \
    --output="${REPORT_DIR}/mobilenet-mnist" \
    "${BINARY}"
nsys stats \
    --report=cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
    --format=csv \
    "${REPORT_DIR}/mobilenet-mnist.nsys-rep" >"${REPORT_DIR}/nsys-summary.csv"

echo "profile: Nsight Compute kernel metrics (one batch)"
MNIST_TRAIN_LIMIT=256 MNIST_TEST_LIMIT=16 MNIST_LOG_INTERVAL=1 \
ncu \
    --force-overwrite \
    --set=full \
    --target-processes=all \
    --launch-count=80 \
    --export="${REPORT_DIR}/mobilenet-conv" \
    "${BINARY}"

echo "profile: reports written to ${REPORT_DIR}"
