#!/usr/bin/env bash
# Submit the non-interactive Isambard environment setup and check job.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${PROJECTDIR:-}" ]]; then
  echo "PROJECTDIR is not set. Run this script on an Isambard login node." >&2
  exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable. Run this script on an Isambard login node." >&2
  exit 2
fi

CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniforge3}"
CONDA_SH="${CONDA_SH:-${CONDA_ROOT}/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-${PROJECTDIR}/${USER}/conda-envs/ula-prior}"

if [[ ! -f "${CONDA_SH}" ]]; then
  MACHINE_ARCHITECTURE="$(uname -m)"
  if [[ "${MACHINE_ARCHITECTURE}" != "aarch64" ]]; then
    echo "Expected Isambard aarch64, got ${MACHINE_ARCHITECTURE}." >&2
    exit 2
  fi
  INSTALLER="${HOME}/Miniforge3-Linux-aarch64.sh"
  INSTALLER_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
  echo "Installing Miniforge non-interactively in ${CONDA_ROOT}"
  curl --fail --location --output "${INSTALLER}" "${INSTALLER_URL}"
  bash "${INSTALLER}" -b -p "${CONDA_ROOT}"
fi

mkdir -p \
  "${REPO_DIR}/slurm/logs" \
  "$(dirname "${CONDA_ENV}")"

export CONDA_ENV
export CONDA_SH
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-bh_sampling}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export ENV_EXPORT_DIR="${ENV_EXPORT_DIR:-${PROJECTDIR}/${USER}/ula-img-sampling-env-exports}"

echo "Submitting Isambard setup and GPU verification job"
echo "Conda environment: ${CONDA_ENV}"
echo "Resolved environment exports: ${ENV_EXPORT_DIR}"
echo "W&B: ${WANDB_MODE} project=${WANDB_PROJECT}"

cd "${REPO_DIR}"
sbatch "$@" slurm/setup_isambard_env.sh
