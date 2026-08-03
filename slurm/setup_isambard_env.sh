#!/bin/bash -l
#SBATCH --job-name=ula-env-setup
#SBATCH --output=slurm/logs/ula_env_setup_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${ULA_REPO_DIR:-}" ]]; then
  REPO_DIR="$(cd "${ULA_REPO_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
if [[ ! -f "${REPO_DIR}/environment-isambard-aarch64.yml" ]]; then
  echo "Repository root not found at: ${REPO_DIR}" >&2
  echo "Submit from the repository root or set ULA_REPO_DIR." >&2
  exit 2
fi

module reset
if [[ -n "${ISAMBARD_PROGRAMMING_ENVIRONMENT:-}" ]]; then
  module load "${ISAMBARD_PROGRAMMING_ENVIRONMENT}"
fi

if [[ -z "${PROJECTDIR:-}" ]]; then
  echo "PROJECTDIR is not set." >&2
  exit 2
fi

CONDA_SH="${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-${PROJECTDIR}/${USER}/conda-envs/ula-prior}"
ENVIRONMENT_FILE="${REPO_DIR}/environment-isambard-aarch64.yml"
ENV_EXPORT_DIR="${ENV_EXPORT_DIR:-${PROJECTDIR}/${USER}/ula-img-sampling-env-exports}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Conda initialization script not found: ${CONDA_SH}" >&2
  exit 2
fi
if [[ ! -f "${ENVIRONMENT_FILE}" ]]; then
  echo "Environment definition not found: ${ENVIRONMENT_FILE}" >&2
  exit 2
fi

source "${CONDA_SH}"
mkdir -p "$(dirname "${CONDA_ENV}")" "${ENV_EXPORT_DIR}"

if [[ -d "${CONDA_ENV}/conda-meta" ]]; then
  echo "Updating existing environment: ${CONDA_ENV}"
  conda env update \
    --prefix "${CONDA_ENV}" \
    --file "${ENVIRONMENT_FILE}"
else
  echo "Creating environment: ${CONDA_ENV}"
  conda env create \
    --prefix "${CONDA_ENV}" \
    --file "${ENVIRONMENT_FILE}"
fi

conda activate "${CONDA_ENV}"
export PYTHONDONTWRITEBYTECODE=1

echo "Checking installed package consistency"
python -m pip check

echo "Exporting the resolved Isambard environment"
conda env export --prefix "${CONDA_ENV}" \
  | sed '/^prefix:/d' \
  > "${ENV_EXPORT_DIR}/environment-isambard-aarch64-resolved.yml"
conda list --explicit --prefix "${CONDA_ENV}" \
  > "${ENV_EXPORT_DIR}/conda-isambard-aarch64-explicit.txt"
python -m pip freeze --all \
  > "${ENV_EXPORT_DIR}/pip-freeze-isambard.txt"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-bh_sampling}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_DIR="${WANDB_DIR:-${ENV_EXPORT_DIR}/wandb}"
export WANDB_DIR
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_DIR}/cache}"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}"

echo "Running CUDA, model, EHT likelihood, and W&B checks"
cd "${REPO_DIR}"
srun --ntasks=1 python check_isambard_environment.py \
  --wandb_mode "${WANDB_MODE}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}"

echo "Environment setup and verification completed successfully."
echo "Environment: ${CONDA_ENV}"
echo "Resolved exports: ${ENV_EXPORT_DIR}"
