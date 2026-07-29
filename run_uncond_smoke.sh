#!/usr/bin/env bash
# Smoke test: UNCONDITIONAL (prior-only) sampling from the pretrained model.
# Small settings so it finishes fast on a Mac. Scale the numbers up once it runs.
#
# Requires:
#   - the pretrained model in ./model/ (model300000.pt + ddpm_*300000.npy)  [already present]
#   - the small "mps" device patch in sample.py (add "mps" to --device choices
#     and to select_device). Until that patch lands, set DEVICE=cpu below.
#   - core deps installed:  pip install -r requirements.txt
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Use the Apple GPU. Change to "cpu" if you haven't applied the mps patch yet.
DEVICE="${DEVICE:-mps}"
# Let unsupported MPS ops fall back to CPU instead of erroring out.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# wandb: "online" = live dashboard (needs `wandb login`, uploads to your account).
# Switch to "offline" to keep everything local, or "disabled" to turn it off.
WANDB_MODE="${WANDB_MODE:-online}"

# Timestamped run name, e.g. uncond_smoke_0728_1530 (monthday_hourminute).
RUN_NAME="${RUN_NAME:-uncond_smoke_$(date +%m%d_%H%M)}"

# Point the model loader at the bundled checkpoint + learned schedule.
export MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/model/model300000.pt}"
export BETA_PATH="${BETA_PATH:-${PROJECT_DIR}/model/ddpm_betas300000.npy}"
export OMEGA_PATH="${OMEGA_PATH:-${PROJECT_DIR}/model/ddpm_omegas300000.npy}"

cd "${PROJECT_DIR}"

"${PYTHON_BIN}" sample.py \
  --log_likelihood zero \
  --schedule_path bh_util/sim_files/sbar_out_55_554000.npy \
  --num_schedule_points 50 \
  --num_rounds 1 \
  --corrector_steps 3 \
  --num_samples 8 \
  --initial_sigma 0.1 \
  --step_tail_probability 0.05 \
  --image_size 128 \
  --device "${DEVICE}" \
  --output_dir "${PROJECT_DIR}/outputs" \
  --run_name "${RUN_NAME}" \
  --wandb_project bh_sampling \
  --wandb_mode "${WANDB_MODE}" \
  "$@"
