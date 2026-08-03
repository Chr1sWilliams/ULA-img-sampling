#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/model/model300000.pt}"
export BETA_PATH="${BETA_PATH:-${PROJECT_DIR}/model/ddpm_betas300000.npy}"
export OMEGA_PATH="${OMEGA_PATH:-${PROJECT_DIR}/model/ddpm_omegas300000.npy}"

LIKELIHOOD_SPEC="${LIKELIHOOD_SPEC:-interferometric}"
UVFILE="${UVFILE:-${PROJECT_DIR}/bh_util/sim_files/simdata_3598_grmhd5_seed3.uvfits}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/model_consistency}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-200}"
NUM_GRID_POINTS="${NUM_GRID_POINTS:-50}"
SCHEDULE_UPDATE_RATE="${SCHEDULE_UPDATE_RATE:-1.0}"
ETA_TOLERANCE="${ETA_TOLERANCE:-0.1}"

cd "${PROJECT_DIR}"

"${PYTHON_BIN}" estimate_consistency_from_model.py \
  --num_samples "${NUM_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --num_grid_points "${NUM_GRID_POINTS}" \
  --schedule_update_rate "${SCHEDULE_UPDATE_RATE}" \
  --eta_tolerance "${ETA_TOLERANCE}" \
  --log_likelihood "${LIKELIHOOD_SPEC}" \
  --uvfile "${UVFILE}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
