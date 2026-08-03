#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Activate the project environment first, or set PYTHON_BIN explicitly.
PYTHON_BIN="${PYTHON_BIN:-python}"

export MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/model/model300000.pt}"
export BETA_PATH="${BETA_PATH:-${PROJECT_DIR}/model/ddpm_betas300000.npy}"
export OMEGA_PATH="${OMEGA_PATH:-${PROJECT_DIR}/model/ddpm_omegas300000.npy}"

LIKELIHOOD_SPEC="${LIKELIHOOD_SPEC:-interferometric}"
WANDB_MODE="${WANDB_MODE:-offline}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs}"
RUN_NAME="${RUN_NAME:-eht_schedule_example}"
IMAGE_CHANNELS="${IMAGE_CHANNELS:-1}"
CONSISTENCY_RESULT="${CONSISTENCY_RESULT:-}"
GUIDANCE_THRESHOLD="${GUIDANCE_THRESHOLD:-}"

EXTRA_ARGS=()
if [[ -n "${CONSISTENCY_RESULT}" ]]; then
  EXTRA_ARGS+=(--consistency_result "${CONSISTENCY_RESULT}")
fi
if [[ -n "${GUIDANCE_THRESHOLD}" ]]; then
  EXTRA_ARGS+=(--guidance_threshold "${GUIDANCE_THRESHOLD}")
fi

cd "${PROJECT_DIR}"

"${PYTHON_BIN}" sample.py \
  --log_likelihood "${LIKELIHOOD_SPEC}" \
  --uvfile bh_util/sim_files/simdata_3598_grmhd5_seed3.uvfits \
  --schedule_path bh_util/sim_files/sbar_out_55_554000.npy \
  --num_schedule_points 4000 \
  --num_rounds 4 \
  --corrector_steps 3 \
  --num_samples 40 \
  --image_channels "${IMAGE_CHANNELS}" \
  --initial_sigma 0.1 \
  --step_tail_probability 0.05 \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --wandb_project bh_sampling \
  --wandb_mode "${WANDB_MODE}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
