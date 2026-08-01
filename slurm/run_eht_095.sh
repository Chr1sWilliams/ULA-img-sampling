#!/bin/bash -l
#SBATCH --job-name=eht-095
#SBATCH --output=slurm/logs/eht_095_%A_%a.out
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

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

CONDA_ENV="${CONDA_ENV:-}"
if [[ -z "${CONDA_ENV}" ]]; then
  if [[ -z "${PROJECTDIR:-}" ]]; then
    echo "Set CONDA_ENV, or set PROJECTDIR so the default environment can be found." >&2
    exit 2
  fi
  CONDA_ENV="${PROJECTDIR}/${USER}/conda-envs/ula-prior"
fi

CONDA_SH="${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Conda initialization script not found: ${CONDA_SH}" >&2
  echo "Set CONDA_SH to your conda.sh location." >&2
  exit 2
fi
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SCHEDULE_PATH="${SCHEDULE_PATH:-${REPO_DIR}/bh_util/sim_files/sbar_out_55_554000.npy}"
export MODEL_PATH="${MODEL_PATH:-${REPO_DIR}/model/model300000.pt}"
export BETA_PATH="${BETA_PATH:-${REPO_DIR}/model/ddpm_betas300000.npy}"
export OMEGA_PATH="${OMEGA_PATH:-${REPO_DIR}/model/ddpm_omegas300000.npy}"
export PYTHONDONTWRITEBYTECODE=1

for required_file in \
  "${UVFILE}" \
  "${SCHEDULE_PATH}" \
  "${MODEL_PATH}" \
  "${BETA_PATH}" \
  "${OMEGA_PATH}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file not found: ${required_file}" >&2
    exit 2
  fi
done

if [[ -n "${PROJECTDIR:-}" ]]; then
  DEFAULT_OUTPUT_ROOT="${PROJECTDIR}/${USER}/ula-img-sampling-runs/eht-095"
else
  DEFAULT_OUTPUT_ROOT="${REPO_DIR}/outputs/eht-095"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-bh_sampling}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-eht-2017-day-095}"
WANDB_TAGS="${WANDB_TAGS:-isambard,eht,day-095,${DATA_LABEL}}"
WANDB_DIR="${WANDB_DIR:-${OUTPUT_ROOT}/wandb}"
export WANDB_DIR
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_DIR}/cache}"
mkdir -p "${OUTPUT_ROOT}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}"

JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RUN_NAME="${RUN_NAME:-eht-095-${DATA_LABEL}-${JOB_ID}-${TASK_ID}}"

NUM_SCHEDULE_POINTS="${NUM_SCHEDULE_POINTS:-4000}"
NUM_ROUNDS="${NUM_ROUNDS:-4}"
CORRECTOR_STEPS="${CORRECTOR_STEPS:-3}"
NUM_SAMPLES="${NUM_SAMPLES:-40}"
INITIAL_SIGMA="${INITIAL_SIGMA:-0.1}"
STEP_TAIL_PROBABILITY="${STEP_TAIL_PROBABILITY:-0.05}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
SEED="${SEED:-0}"
CONSISTENCY_RESULT="${CONSISTENCY_RESULT:-}"
GUIDANCE_THRESHOLD="${GUIDANCE_THRESHOLD:-}"

EXTRA_ARGS=()
if [[ -n "${CONSISTENCY_RESULT}" ]]; then
  EXTRA_ARGS+=(--consistency_result "${CONSISTENCY_RESULT}")
fi
if [[ -n "${GUIDANCE_THRESHOLD}" ]]; then
  EXTRA_ARGS+=(--guidance_threshold "${GUIDANCE_THRESHOLD}")
fi
if [[ -n "${WANDB_ENTITY}" ]]; then
  EXTRA_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
fi

"${PYTHON_BIN}" -c \
  "import ehtim, torch, torchkbnufft, wandb; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('CUDA:', torch.cuda.get_device_name(0)); print('torchkbnufft:', torchkbnufft.__version__); print('W&B:', wandb.__version__)"

echo "Dataset: ${DATA_LABEL} (${UVFILE})"
echo "Output: ${OUTPUT_ROOT}/${RUN_NAME}"
echo "W&B: ${WANDB_MODE} project=${WANDB_PROJECT} group=${WANDB_GROUP}"

cd "${REPO_DIR}"
srun --ntasks=1 "${PYTHON_BIN}" sample.py \
  --log_likelihood interferometric \
  --uvfile "${UVFILE}" \
  --schedule_path "${SCHEDULE_PATH}" \
  --num_schedule_points "${NUM_SCHEDULE_POINTS}" \
  --num_rounds "${NUM_ROUNDS}" \
  --corrector_steps "${CORRECTOR_STEPS}" \
  --num_samples "${NUM_SAMPLES}" \
  --initial_sigma "${INITIAL_SIGMA}" \
  --step_tail_probability "${STEP_TAIL_PROBABILITY}" \
  --seed "${SEED}" \
  --device cuda \
  --output_dir "${OUTPUT_ROOT}" \
  --run_name "${RUN_NAME}" \
  --log_interval "${LOG_INTERVAL}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_group "${WANDB_GROUP}" \
  --wandb_tags "${WANDB_TAGS}" \
  --wandb_mode "${WANDB_MODE}" \
  --wandb_log_artifacts \
  "${EXTRA_ARGS[@]}"
