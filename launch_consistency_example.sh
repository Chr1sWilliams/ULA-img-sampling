#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "${DATA_DIR:-}" ]]; then
  echo "Set DATA_DIR to representative clean q_0 images." >&2
  exit 2
fi

export MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/model/model300000.pt}"
export BETA_PATH="${BETA_PATH:-${PROJECT_DIR}/model/ddpm_betas300000.npy}"
export OMEGA_PATH="${OMEGA_PATH:-${PROJECT_DIR}/model/ddpm_omegas300000.npy}"

LIKELIHOOD_SPEC="${LIKELIHOOD_SPEC:-interferometric}"
UVFILE="${UVFILE:-${PROJECT_DIR}/bh_util/sim_files/simdata_3598_grmhd5_seed3.uvfits}"
SCHEDULE_PATH="${SCHEDULE_PATH:-${PROJECT_DIR}/bh_util/sim_files/sbar_out_55_554000.npy}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/consistency}"
NUM_GRID_POINTS="${NUM_GRID_POINTS:-50}"
NUM_SAMPLES="${NUM_SAMPLES:-40}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ETA_TOLERANCE="${ETA_TOLERANCE:-0.1}"

cd "${PROJECT_DIR}"

"${PYTHON_BIN}" estimate_consistency.py \
  --data_dir "${DATA_DIR}" \
  --schedule_path "${SCHEDULE_PATH}" \
  --num_grid_points "${NUM_GRID_POINTS}" \
  --num_samples "${NUM_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --eta_tolerance "${ETA_TOLERANCE}" \
  --log_likelihood "${LIKELIHOOD_SPEC}" \
  --uvfile "${UVFILE}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
