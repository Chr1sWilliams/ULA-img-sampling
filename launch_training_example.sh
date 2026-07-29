#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "${DATA_DIR:-}" ]]; then
  echo "Set DATA_DIR to a directory containing the training images." >&2
  exit 2
fi

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/diffusion_prior}"
TRAIN_STEPS="${TRAIN_STEPS:-300000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MICROBATCH="${MICROBATCH:--1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
USE_FP16="${USE_FP16:-false}"
IMAGE_CHANNELS="${IMAGE_CHANNELS:-1}"
NOISE_SCHEDULE="${NOISE_SCHEDULE:-cosine}"
SCHEDULE_TUNE="${SCHEDULE_TUNE:-true}"
SCHEDULE_LR="${SCHEDULE_LR:-0.01}"
BETA_VALUES_FILE="${BETA_VALUES_FILE:-}"
OMEGA_VALUES_FILE="${OMEGA_VALUES_FILE:-}"

cd "${PROJECT_DIR}"

COMMAND=(
  "${PYTHON_BIN}" image_train.py
  --data_dir "${DATA_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --image_size 128
  --num_channels 64
  --num_res_blocks 2
  --image_channels "${IMAGE_CHANNELS}"
  --noise_schedule "${NOISE_SCHEDULE}"
  --schedule_tune "${SCHEDULE_TUNE}"
  --schedule_lr "${SCHEDULE_LR}"
  --batch_size "${BATCH_SIZE}"
  --microbatch "${MICROBATCH}"
  --lr_anneal_steps "${TRAIN_STEPS}"
  --save_interval "${SAVE_INTERVAL}"
  --use_fp16 "${USE_FP16}"
)

if [[ -n "${BETA_VALUES_FILE}" || -n "${OMEGA_VALUES_FILE}" ]]; then
  COMMAND+=(
    --beta_values_file "${BETA_VALUES_FILE}"
    --omega_values_file "${OMEGA_VALUES_FILE}"
  )
fi

COMMAND+=("$@")
exec "${COMMAND[@]}"
