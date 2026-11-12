#!/bin/bash
#SBATCH --job-name=iad-heatmap
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${IAD_HEATMAP_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
: "${IAD_DATA_DIR:?Set IAD_DATA_DIR to the project-storage IAD directory}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"

ENV_PREFIX=${IAD_HEATMAP_ENV_PREFIX:-"${PROJECT_ROOT}/.envs/iad-cflow"}
CONFIG=${IAD_HEATMAP_CONFIG:-"${PROJECT_ROOT}/configs/iad_common_normal_berzelius.json"}
SELECTION=${IAD_HEATMAP_SELECTION:-"${PROJECT_ROOT}/configs/iad_formal_heatmap_selection.json"}
OUTPUT_ROOT=${IAD_HEATMAP_OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs/iad_formal_heatmaps"}
SEED=${IAD_HEATMAP_SEED:-2026}
WORKERS=${IAD_HEATMAP_WORKERS:-8}
EFFICIENTAD_ROOT=${IAD_HEATMAP_EFFICIENTAD_ROOT:-"${PROJECT_ROOT}/.cache/iad_assets"}
TORCH_HOME_ROOT=${IAD_HEATMAP_TORCH_HOME:-"${PROJECT_ROOT}/.cache/torch"}

export CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV-}
module load Miniforge3/24.7.1-2-hpc1-bdist
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "Missing environment: ${ENV_PREFIX}" >&2
  exit 2
fi
if [[ ! -d "${IAD_DATA_DIR}" ]]; then
  echo "IAD_DATA_DIR is not a directory: ${IAD_DATA_DIR}" >&2
  exit 2
fi

TASK_METHOD=$(mamba run -p "${ENV_PREFIX}" python \
  "${PROJECT_ROOT}/scripts/run_iad_formal_heatmap_export.py" \
  --seed "${SEED}" --task-id "${SLURM_ARRAY_TASK_ID}" --print-task-method)

SCRATCH_ROOT="/scratch/local/${SLURM_JOB_ID}/iad-formal-heatmap"
SCRATCH_DATA="${SCRATCH_ROOT}/IAD"
mkdir -p "${SCRATCH_DATA}" "${OUTPUT_ROOT}"
cp -a "${IAD_DATA_DIR}/." "${SCRATCH_DATA}/"

RUN_ARGS=(
  scripts/run_iad_formal_heatmap_export.py
  --config "${CONFIG}"
  --selection "${SELECTION}"
  --task-id "${SLURM_ARRAY_TASK_ID}"
  --seed "${SEED}"
  --dataset-dir "${SCRATCH_DATA}"
  --output-root "${OUTPUT_ROOT}"
  --device cuda
  --workers "${WORKERS}"
)

if [[ "${TASK_METHOD}" == "efficientad" ]]; then
  SOURCE_TEACHER="${EFFICIENTAD_ROOT}/efficientad_pretrained_weights/pretrained_teacher_medium.pth"
  SOURCE_IMAGENETTE="${EFFICIENTAD_ROOT}/imagenette2"
  if [[ ! -f "${SOURCE_TEACHER}" || ! -d "${SOURCE_IMAGENETTE}/train" ]]; then
    echo "Missing EfficientAD assets below ${EFFICIENTAD_ROOT}" >&2
    exit 2
  fi
  SCRATCH_EFFICIENTAD="${SCRATCH_ROOT}/EfficientAD_assets"
  mkdir -p "${SCRATCH_EFFICIENTAD}/efficientad_pretrained_weights"
  cp "${SOURCE_TEACHER}" \
    "${SCRATCH_EFFICIENTAD}/efficientad_pretrained_weights/"
  cp -a "${SOURCE_IMAGENETTE}" "${SCRATCH_EFFICIENTAD}/"
  RUN_ARGS+=(
    --teacher-weights "${SCRATCH_EFFICIENTAD}/efficientad_pretrained_weights/pretrained_teacher_medium.pth"
    --imagenette-dir "${SCRATCH_EFFICIENTAD}/imagenette2"
  )
fi

if [[ "${IAD_HEATMAP_SMOKE:-0}" == "1" ]]; then
  RUN_ARGS+=(--smoke --force)
fi
if [[ "${IAD_HEATMAP_FORCE:-0}" == "1" ]]; then
  RUN_ARGS+=(--force)
fi

export PYTHONNOUSERSITE=1
export TORCH_HOME="${TORCH_HOME_ROOT}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} method=${TASK_METHOD} seed=${SEED}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
cd "${PROJECT_ROOT}"
mamba run -p "${ENV_PREFIX}" python "${RUN_ARGS[@]}"
