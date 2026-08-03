#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TASK_TMP_DIR="${TMPDIR:-/tmp}"

MNIST_CACHE_DIR="${MNIST_CACHE_DIR:-${TASK_TMP_DIR}/ula-img-sampling-mnist}"
MNIST_IMAGE_DIR="${MNIST_IMAGE_DIR:-${MNIST_CACHE_DIR}/images}"
MNIST_LIMIT="${MNIST_LIMIT:-10000}"

"${PYTHON_BIN}" "${PROJECT_DIR}/prepare_mnist.py" \
  --download_dir "${MNIST_CACHE_DIR}/download" \
  --output_dir "${MNIST_IMAGE_DIR}" \
  --limit "${MNIST_LIMIT}"

export DATA_DIR="${MNIST_IMAGE_DIR}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/mnist_real}"
export TRAIN_STEPS="${TRAIN_STEPS:-5000}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
export IMAGE_CHANNELS=1
export NOISE_SCHEDULE="${NOISE_SCHEDULE:-cosine}"
export SCHEDULE_TUNE="${SCHEDULE_TUNE:-true}"
export SCHEDULE_LR="${SCHEDULE_LR:-0.01}"
export BETA_VALUES_FILE=
export OMEGA_VALUES_FILE=

exec "${PROJECT_DIR}/launch_training_example.sh" \
  --image_size 32 \
  --num_channels 64 \
  --num_res_blocks 2 \
  --diffusion_steps 100 \
  "$@"
