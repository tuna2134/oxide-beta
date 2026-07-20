#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${MNIST_DATA_DIR:-${PROJECT_DIR}/data/mnist}"
USE_CUDA=0
DOWNLOAD_ONLY=0

usage() {
    cat <<'EOF'
Usage: scripts/train_mnist.sh [OPTIONS]

Download, verify, and extract MNIST, then train MobileNetV4.

Options:
  --cuda               Run the training example with Device::Cuda(0)
  --data-dir DIRECTORY Store/read MNIST in DIRECTORY
  --download-only      Prepare the dataset without starting training
  -h, --help           Show this help

Training settings are controlled with MNIST_EPOCHS, MNIST_BATCH_SIZE,
MNIST_TRAIN_LIMIT, and MNIST_TEST_LIMIT.
EOF
}

while (($# > 0)); do
    case "$1" in
        --cuda)
            USE_CUDA=1
            shift
            ;;
        --data-dir)
            if (($# < 2)); then
                echo "error: --data-dir requires a directory" >&2
                exit 2
            fi
            DATA_DIR="$2"
            shift 2
            ;;
        --download-only)
            DOWNLOAD_ONLY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

for command in gzip cargo; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "error: required command not found: ${command}" >&2
        exit 1
    fi
done

if command -v curl >/dev/null 2>&1; then
    DOWNLOADER=curl
elif command -v wget >/dev/null 2>&1; then
    DOWNLOADER=wget
else
    echo "error: curl or wget is required to download MNIST" >&2
    exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
    SHA256_COMMAND=sha256sum
elif command -v shasum >/dev/null 2>&1; then
    SHA256_COMMAND=shasum
else
    echo "error: sha256sum or shasum is required to verify MNIST" >&2
    exit 1
fi

readonly MIRRORS=(
    "https://ossci-datasets.s3.amazonaws.com/mnist"
    "https://storage.googleapis.com/cvdf-datasets/mnist"
)
readonly FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)
readonly SHA256=(
    "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609"
    "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c"
    "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6"
    "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6"
)

download() {
    local url="$1"
    local output="$2"
    if [[ "${DOWNLOADER}" == curl ]]; then
        curl --fail --location --retry 3 --connect-timeout 15 --output "${output}" "${url}"
    else
        wget --tries=3 --timeout=15 --output-document="${output}" "${url}"
    fi
}

file_sha256() {
    if [[ "${SHA256_COMMAND}" == sha256sum ]]; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

prepare_file() {
    local index="$1"
    local archive_name="${FILES[index]}"
    local expected_sha256="${SHA256[index]}"
    local output_name="${archive_name%.gz}"
    local archive_path="${DATA_DIR}/${archive_name}"
    local output_path="${DATA_DIR}/${output_name}"
    local part_path="${archive_path}.part"
    local extracted_part_path="${output_path}.part"

    if [[ -s "${output_path}" ]]; then
        echo "MNIST: reuse ${output_path}"
        return
    fi

    if [[ -f "${archive_path}" ]] && [[ "$(file_sha256 "${archive_path}")" != "${expected_sha256}" ]]; then
        echo "MNIST: cached checksum mismatch; downloading ${archive_name} again"
        mv -- "${archive_path}" "${part_path}"
    fi

    if [[ ! -f "${archive_path}" ]]; then
        local downloaded=0
        for mirror in "${MIRRORS[@]}"; do
            echo "MNIST: download ${mirror}/${archive_name}"
            if download "${mirror}/${archive_name}" "${part_path}"; then
                if [[ "$(file_sha256 "${part_path}")" == "${expected_sha256}" ]]; then
                    mv -- "${part_path}" "${archive_path}"
                    downloaded=1
                    break
                fi
                echo "MNIST: checksum mismatch from ${mirror}" >&2
            fi
        done
        if ((downloaded == 0)); then
            echo "error: failed to download a verified ${archive_name}" >&2
            exit 1
        fi
    fi

    echo "MNIST: extract ${archive_name}"
    gzip --test "${archive_path}"
    gzip --decompress --stdout "${archive_path}" >"${extracted_part_path}"
    mv -- "${extracted_part_path}" "${output_path}"
}

mkdir -p -- "${DATA_DIR}"
for index in "${!FILES[@]}"; do
    prepare_file "${index}"
done

echo "MNIST: dataset ready at ${DATA_DIR}"
if ((DOWNLOAD_ONLY == 1)); then
    exit 0
fi

cd -- "${PROJECT_DIR}"
if ((USE_CUDA == 1)); then
    export MNIST_DATA_DIR="${DATA_DIR}"
    export OXIDE_TORCH_CUDA=1
    exec cargo oxide run --features cuda --bin mnist-training
else
    exec cargo +stable run --release --example mnist_training -- "${DATA_DIR}"
fi
