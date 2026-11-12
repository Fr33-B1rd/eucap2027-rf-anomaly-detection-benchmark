#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${IAD_SAMPLE_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
: "${BERZELIUS_ACCOUNT:?Set BERZELIUS_ACCOUNT to the Slurm project account}"
: "${IAD_DATA_DIR:?Set IAD_DATA_DIR to the project-storage IAD directory}"

export IAD_SAMPLE_PROJECT_ROOT="${PROJECT_ROOT}"
export IAD_SAMPLE_ENV_PREFIX=${IAD_SAMPLE_ENV_PREFIX:-"${PROJECT_ROOT}/.envs/iad-cflow"}
export IAD_SAMPLE_ASSET_ROOT=${IAD_SAMPLE_ASSET_ROOT:-"${PROJECT_ROOT}/.cache/iad_assets"}
export IAD_SAMPLE_TORCH_HOME=${IAD_SAMPLE_TORCH_HOME:-"${PROJECT_ROOT}/.cache/torch"}
export IAD_SAMPLE_OUTPUT_ROOT_BASE=${IAD_SAMPLE_OUTPUT_ROOT_BASE:-"${PROJECT_ROOT}/outputs/iad_training_samples_berzelius"}
export IAD_SAMPLE_WORKERS=${IAD_SAMPLE_WORKERS:-8}
export IAD_SAMPLE_SMOKE=${IAD_SAMPLE_SMOKE:-0}
export IAD_DATA_DIR
export CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV-}

module load Miniforge3/24.7.1-2-hpc1-bdist
mkdir -p "${PROJECT_ROOT}/logs" "${IAD_SAMPLE_OUTPUT_ROOT_BASE}/n64" "${IAD_SAMPLE_OUTPUT_ROOT_BASE}/n256"

if [[ ! -x "${IAD_SAMPLE_ENV_PREFIX}/bin/python" ]]; then
  echo "Missing environment; run berzelius/prepare_cflow_env.sh first" >&2
  exit 2
fi
if [[ ! -d "${IAD_DATA_DIR}" ]]; then
  echo "IAD_DATA_DIR is not a directory: ${IAD_DATA_DIR}" >&2
  exit 2
fi
mamba run -p "${IAD_SAMPLE_ENV_PREFIX}" python -c \
  "import FrEIA, sklearn, torch, torchvision; print(torch.__version__, torchvision.__version__)"

for required in \
  "${IAD_SAMPLE_ASSET_ROOT}/ASSET_MANIFEST.json" \
  "${IAD_SAMPLE_ASSET_ROOT}/efficientad/weights/efficientad_pretrained_weights/pretrained_teacher_medium.pth" \
  "${IAD_SAMPLE_ASSET_ROOT}/efficientad/imagenette2.tgz"
do
  if [[ ! -f "${required}" ]]; then
    echo "Missing EfficientAD prerequisite: ${required}" >&2
    exit 2
  fi
done

MAX_CONCURRENT=${IAD_SAMPLE_MAX_CONCURRENT:-5}
BERZELIUS_SYSTEM=${BERZELIUS_SYSTEM:-ampere}
TOTAL_TASKS=0
for BUDGET in 64 256; do
  CONFIG="${PROJECT_ROOT}/configs/iad_training_samples_n${BUDGET}_berzelius.json"
  if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing config: ${CONFIG}" >&2
    exit 2
  fi
  TASK_COUNT=$(mamba run -p "${IAD_SAMPLE_ENV_PREFIX}" python \
    "${PROJECT_ROOT}/scripts/run_iad_full_training_berzelius.py" \
    --config "${CONFIG}" --task-kind gpu --task-count)
  if [[ "${TASK_COUNT}" != "20" ]]; then
    echo "Expected 20 cells for budget ${BUDGET}, got ${TASK_COUNT}" >&2
    exit 2
  fi
  TOTAL_TASKS=$((TOTAL_TASKS + TASK_COUNT))
  echo "Budget ${BUDGET} local task map (global offset $((BUDGET == 64 ? 0 : 20))):"
  mamba run -p "${IAD_SAMPLE_ENV_PREFIX}" python \
    "${PROJECT_ROOT}/scripts/run_iad_full_training_berzelius.py" \
    --config "${CONFIG}" --task-kind gpu --list-tasks

done
if [[ "${TOTAL_TASKS}" != "40" ]]; then
  echo "Expected 40 total cells, got ${TOTAL_TASKS}" >&2
  exit 2
fi

if [[ -n "${IAD_SAMPLE_ARRAY_SPEC:-}" ]]; then
  ARRAY_SPEC="${IAD_SAMPLE_ARRAY_SPEC}"
elif [[ "${IAD_SAMPLE_SMOKE}" == "1" ]]; then
  ARRAY_SPEC="0,5,10,15%4"
else
  ARRAY_SPEC="0-$((TOTAL_TASKS - 1))%${MAX_CONCURRENT}"
fi

SBATCH_ARGS=(
  --account "${BERZELIUS_ACCOUNT}"
  --array "${ARRAY_SPEC}"
  --export="ALL,IAD_SAMPLE_PROJECT_ROOT=${PROJECT_ROOT},IAD_DATA_DIR=${IAD_DATA_DIR},IAD_SAMPLE_ENV_PREFIX=${IAD_SAMPLE_ENV_PREFIX},IAD_SAMPLE_OUTPUT_ROOT_BASE=${IAD_SAMPLE_OUTPUT_ROOT_BASE},IAD_SAMPLE_ASSET_ROOT=${IAD_SAMPLE_ASSET_ROOT},IAD_SAMPLE_TORCH_HOME=${IAD_SAMPLE_TORCH_HOME},IAD_SAMPLE_WORKERS=${IAD_SAMPLE_WORKERS},IAD_SAMPLE_SMOKE=${IAD_SAMPLE_SMOKE}"
)
if [[ "${BERZELIUS_SYSTEM}" == "ampere" ]]; then
  SBATCH_ARGS+=(--constraint "${IAD_SAMPLE_CONSTRAINT:-thin}")
elif [[ "${BERZELIUS_SYSTEM}" != "hopper" ]]; then
  echo "BERZELIUS_SYSTEM must be ampere or hopper" >&2
  exit 2
fi

echo "Submitting ${TOTAL_TASKS}-cell matrix with array ${ARRAY_SPEC}, smoke=${IAD_SAMPLE_SMOKE}"
cd "${PROJECT_ROOT}"
sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/iad_training_samples_gpu.sbatch"
