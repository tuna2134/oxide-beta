#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPORT_DIR="${PROFILE_REPORT_DIR:-${PROJECT_DIR}/profile-results}"
mkdir -p -- "${REPORT_DIR}"
cd -- "${PROJECT_DIR}"

command -v cargo >/dev/null 2>&1 || {
    echo "error: cargo is required" >&2
    exit 1
}

# Colab images sometimes install Nsight under the CUDA Toolkit directory
# without adding it to PATH. Nsight remains optional: host phase timings are
# still useful when neither CLI is available.
if [[ -d /usr/local/cuda/bin ]]; then
    export PATH="/usr/local/cuda/bin:${PATH}"
fi
NSYS_COMMAND="$(command -v nsys || true)"
NCU_COMMAND="$(command -v ncu || true)"

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

if [[ -n "${NSYS_COMMAND}" ]]; then
    echo "profile: Nsight Systems timeline (${NSYS_COMMAND})"
    "${NSYS_COMMAND}" profile \
        --force-overwrite=true \
        --trace=cuda,osrt \
        --cuda-memory-usage=true \
        --sample=none \
        --output="${REPORT_DIR}/mobilenet-mnist" \
        "${BINARY}"
    "${NSYS_COMMAND}" stats \
        --report=cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
        --format=csv \
        "${REPORT_DIR}/mobilenet-mnist.nsys-rep" >"${REPORT_DIR}/nsys-summary.csv"
else
    echo "profile: nsys not found; skipping CUDA timeline" >&2
fi

if [[ -n "${NCU_COMMAND}" ]]; then
    echo "profile: Nsight Compute kernel metrics (${NCU_COMMAND}, one batch)"
    MNIST_TRAIN_LIMIT=256 MNIST_TEST_LIMIT=16 MNIST_LOG_INTERVAL=1 \
    "${NCU_COMMAND}" \
        --force-overwrite \
        --set=full \
        --target-processes=all \
        --launch-count=80 \
        --export="${REPORT_DIR}/mobilenet-conv" \
        "${BINARY}"
else
    echo "profile: ncu not found; skipping per-kernel hardware metrics" >&2
fi

echo "profile: reports written to ${REPORT_DIR}"
