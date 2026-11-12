#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${IAD_LOW_N_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
: "${BERZELIUS_ACCOUNT:?Set BERZELIUS_ACCOUNT to the Slurm project account}"
: "${IAD_DATA_DIR:?Set IAD_DATA_DIR to the project-storage IAD directory}"

export IAD_LOW_N_PROJECT_ROOT="${PROJECT_ROOT}"
export IAD_LOW_N_ENV_PREFIX=${IAD_LOW_N_ENV_PREFIX:-"${PROJECT_ROOT}/.envs/iad-cflow"}
export IAD_LOW_N_ASSET_ROOT=${IAD_LOW_N_ASSET_ROOT:-"${PROJECT_ROOT}/.cache/iad_assets"}
export IAD_LOW_N_TORCH_HOME=${IAD_LOW_N_TORCH_HOME:-"${PROJECT_ROOT}/.cache/torch"}
export IAD_LOW_N_OUTPUT_ROOT_BASE=${IAD_LOW_N_OUTPUT_ROOT_BASE:-"${PROJECT_ROOT}/outputs/iad_training_samples_low_budget_berzelius"}
export IAD_LOW_N_WORKERS=${IAD_LOW_N_WORKERS:-8}
export IAD_LOW_N_SMOKE=${IAD_LOW_N_SMOKE:-0}
export IAD_DATA_DIR
export CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV-}

module load Miniforge3/24.7.1-2-hpc1-bdist
mkdir -p "${PROJECT_ROOT}/logs"
for BUDGET in 16 32 128; do
  mkdir -p "${IAD_LOW_N_OUTPUT_ROOT_BASE}/n${BUDGET}"
done

if [[ ! -x "${IAD_LOW_N_ENV_PREFIX}/bin/python" ]]; then
  echo "Missing environment; run berzelius/prepare_cflow_env.sh first" >&2
  exit 2
fi
if [[ ! -d "${IAD_DATA_DIR}" ]]; then
  echo "IAD_DATA_DIR is not a directory: ${IAD_DATA_DIR}" >&2
  exit 2
fi
mamba run -p "${IAD_LOW_N_ENV_PREFIX}" python -c \
  "import FrEIA, sklearn, torch, torchvision; print(torch.__version__, torchvision.__version__)"

for required in \
  "${IAD_LOW_N_ASSET_ROOT}/ASSET_MANIFEST.json" \
  "${IAD_LOW_N_ASSET_ROOT}/efficientad/weights/efficientad_pretrained_weights/pretrained_teacher_medium.pth" \
  "${IAD_LOW_N_ASSET_ROOT}/efficientad/imagenette2.tgz"
do
  if [[ ! -f "${required}" ]]; then
    echo "Missing EfficientAD prerequisite: ${required}" >&2
    exit 2
  fi
done

MAX_CONCURRENT=${IAD_LOW_N_MAX_CONCURRENT:-5}
BERZELIUS_SYSTEM=${BERZELIUS_SYSTEM:-ampere}
TOTAL_TASKS=0
GLOBAL_OFFSET=0
for BUDGET in 16 32 128; do
  CONFIG="${PROJECT_ROOT}/configs/iad_training_samples_n${BUDGET}_berzelius.json"
  if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing config: ${CONFIG}" >&2
    exit 2
  fi
  TASK_COUNT=$(mamba run -p "${IAD_LOW_N_ENV_PREFIX}" python \
    "${PROJECT_ROOT}/scripts/run_iad_full_training_berzelius.py" \
    --config "${CONFIG}" --task-kind gpu --task-count)
  if [[ "${TASK_COUNT}" != "20" ]]; then
    echo "Expected 20 cells for budget ${BUDGET}, got ${TASK_COUNT}" >&2
    exit 2
  fi
  echo "Budget ${BUDGET} local task map (global offset ${GLOBAL_OFFSET}):"
  mamba run -p "${IAD_LOW_N_ENV_PREFIX}" python \
    "${PROJECT_ROOT}/scripts/run_iad_full_training_berzelius.py" \
    --config "${CONFIG}" --task-kind gpu --list-tasks
  TOTAL_TASKS=$((TOTAL_TASKS + TASK_COUNT))
  GLOBAL_OFFSET=$((GLOBAL_OFFSET + TASK_COUNT))
done
if [[ "${TOTAL_TASKS}" != "60" ]]; then
  echo "Expected 60 total cells, got ${TOTAL_TASKS}" >&2
  exit 2
fi

if [[ -n "${IAD_LOW_N_ARRAY_SPEC:-}" ]]; then
  ARRAY_SPEC="${IAD_LOW_N_ARRAY_SPEC}"
elif [[ "${IAD_LOW_N_SMOKE}" == "1" ]]; then
  # All four methods at N=16, plus CFLOW at N=32 and N=128.
  ARRAY_SPEC="0,5,10,15,30,50%4"
else
  ARRAY_SPEC="0-$((TOTAL_TASKS - 1))%${MAX_CONCURRENT}"
fi

SBATCH_ARGS=(
  --account "${BERZELIUS_ACCOUNT}"
  --array "${ARRAY_SPEC}"
  --export="ALL,IAD_LOW_N_PROJECT_ROOT=${PROJECT_ROOT},IAD_DATA_DIR=${IAD_DATA_DIR},IAD_LOW_N_ENV_PREFIX=${IAD_LOW_N_ENV_PREFIX},IAD_LOW_N_OUTPUT_ROOT_BASE=${IAD_LOW_N_OUTPUT_ROOT_BASE},IAD_LOW_N_ASSET_ROOT=${IAD_LOW_N_ASSET_ROOT},IAD_LOW_N_TORCH_HOME=${IAD_LOW_N_TORCH_HOME},IAD_LOW_N_WORKERS=${IAD_LOW_N_WORKERS},IAD_LOW_N_SMOKE=${IAD_LOW_N_SMOKE}"
)
if [[ "${BERZELIUS_SYSTEM}" == "ampere" ]]; then
  SBATCH_ARGS+=(--constraint "${IAD_LOW_N_CONSTRAINT:-thin}")
elif [[ "${BERZELIUS_SYSTEM}" != "hopper" ]]; then
  echo "BERZELIUS_SYSTEM must be ampere or hopper" >&2
  exit 2
fi

echo "Submitting ${TOTAL_TASKS}-cell matrix with array ${ARRAY_SPEC}, smoke=${IAD_LOW_N_SMOKE}"
cd "${PROJECT_ROOT}"
sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/iad_training_samples_low_budget_gpu.sbatch"
