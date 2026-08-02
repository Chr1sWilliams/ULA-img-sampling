#!/bin/bash -l
#SBATCH --job-name=model-consistency
#SBATCH --output=slurm/logs/model_consistency_%A_%a.out
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${ULA_REPO_DIR:-}" ]]; then
  REPO_DIR="$(cd "${ULA_REPO_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
if [[ ! -f "${REPO_DIR}/estimate_consistency_from_model.py" ]]; then
  echo "Repository root not found at: ${REPO_DIR}" >&2
  echo "Submit from the repository root or set ULA_REPO_DIR." >&2
  exit 2
fi

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    DATA_LABEL="hi"
    UVFILE="${REPO_DIR}/bh_util/sim_files/SR1_M87_2017_095_hi_hops_netcal_StokesI.uvfits"
    ;;
  1)
    DATA_LABEL="lo"
    UVFILE="${REPO_DIR}/bh_util/sim_files/SR1_M87_2017_095_lo_hops_netcal_StokesI.uvfits"
    ;;
  *)
    echo "SLURM_ARRAY_TASK_ID must be 0 (hi) or 1 (lo)." >&2
    exit 2
    ;;
esac

module reset
if [[ -n "${ISAMBARD_PROGRAMMING_ENVIRONMENT:-}" ]]; then
  module load "${ISAMBARD_PROGRAMMING_ENVIRONMENT}"
fi

if [[ -z "${CONDA_ENV:-}" ]]; then
  if [[ -z "${PROJECTDIR:-}" ]]; then
    echo "Set CONDA_ENV, or set PROJECTDIR so the environment can be found." >&2
    exit 2
  fi
  CONDA_ENV="${PROJECTDIR}/${USER}/conda-envs/ula-prior"
fi
CONDA_SH="${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Conda initialization script not found: ${CONDA_SH}" >&2
  exit 2
fi
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

PYTHON_BIN="${PYTHON_BIN:-python}"
export MODEL_PATH="${MODEL_PATH:-${REPO_DIR}/model/model300000.pt}"
export BETA_PATH="${BETA_PATH:-${REPO_DIR}/model/ddpm_betas300000.npy}"
export OMEGA_PATH="${OMEGA_PATH:-${REPO_DIR}/model/ddpm_omegas300000.npy}"
export PYTHONDONTWRITEBYTECODE=1

for required_file in \
  "${UVFILE}" \
  "${MODEL_PATH}" \
  "${BETA_PATH}" \
  "${OMEGA_PATH}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file not found: ${required_file}" >&2
    exit 2
  fi
done

if [[ -n "${PROJECTDIR:-}" ]]; then
  DEFAULT_OUTPUT_ROOT="${PROJECTDIR}/${USER}/ula-img-sampling-runs/model-consistency"
else
  DEFAULT_OUTPUT_ROOT="${REPO_DIR}/outputs/model-consistency"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RUN_NAME="${RUN_NAME:-day-095-${DATA_LABEL}-${JOB_ID}-${TASK_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"

NUM_SAMPLES="${NUM_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-200}"
NUM_GRID_POINTS="${NUM_GRID_POINTS:-50}"
SCHEDULE_UPDATE_RATE="${SCHEDULE_UPDATE_RATE:-1.0}"
ETA_TOLERANCE="${ETA_TOLERANCE:-0.1}"
SEED="${SEED:-0}"

mkdir -p "${OUTPUT_DIR}"
"${PYTHON_BIN}" -c \
  "import ehtim, torch, torchkbnufft; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('CUDA:', torch.cuda.get_device_name(0)); print('torchkbnufft:', torchkbnufft.__version__)"

echo "Dataset: ${DATA_LABEL} (${UVFILE})"
echo "Samples per stage: ${NUM_SAMPLES}"
echo "Output: ${OUTPUT_DIR}"

cd "${REPO_DIR}"
srun --ntasks=1 "${PYTHON_BIN}" estimate_consistency_from_model.py \
  --num_samples "${NUM_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --num_grid_points "${NUM_GRID_POINTS}" \
  --schedule_update_rate "${SCHEDULE_UPDATE_RATE}" \
  --eta_tolerance "${ETA_TOLERANCE}" \
  --seed "${SEED}" \
  --device cuda \
  --log_likelihood interferometric \
  --uvfile "${UVFILE}" \
  --output_dir "${OUTPUT_DIR}"
