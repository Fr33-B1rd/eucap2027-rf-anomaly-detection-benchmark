#!/bin/bash
#SBATCH --job-name=iad-full-gpu
#SBATCH --gpus=1
#SBATCH --time=02:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${IAD_FULL_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
: "${IAD_DATA_DIR:?Set IAD_DATA_DIR to the project-storage IAD directory}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"

ENV_PREFIX=${IAD_FULL_ENV_PREFIX:-"${PROJECT_ROOT}/.envs/iad-final"}
CONFIG=${IAD_FULL_CONFIG:-"${PROJECT_ROOT}/configs/iad_benchmark.json"}
OUTPUT_ROOT=${IAD_FULL_OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs/iad_benchmark"}
ASSET_ROOT=${IAD_FULL_ASSET_ROOT:-"${PROJECT_ROOT}/.cache/iad_assets"}
TORCH_HOME_ROOT=${IAD_FULL_TORCH_HOME:-"${PROJECT_ROOT}/.cache/torch"}
WORKERS=${IAD_FULL_WORKERS:-8}

# NSC's Miniforge wrapper reads this variable while `set -u` is active.
export CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV-}
module load Miniforge3/24.7.1-2-hpc1-bdist
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "Missing environment; run berzelius/prepare_iad_final_env.sh first" >&2
  exit 2
fi
if [[ ! -d "${IAD_DATA_DIR}" ]]; then
  echo "IAD_DATA_DIR is not a directory: ${IAD_DATA_DIR}" >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing full-training config: ${CONFIG}" >&2
  exit 2
fi

TASK_METHOD=$(mamba run -p "${ENV_PREFIX}" python \
  "${PROJECT_ROOT}/scripts/run_iad_full_training_berzelius.py" \
  --config "${CONFIG}" --task-kind gpu \
  --task-id "${SLURM_ARRAY_TASK_ID}" --print-task-method)

SCRATCH_ROOT="/scratch/local/${SLURM_JOB_ID}/iad-full-training"
SCRATCH_DATA="${SCRATCH_ROOT}/IAD"
SCRATCH_ASSETS="${SCRATCH_ROOT}/assets"
mkdir -p "${SCRATCH_DATA}" "${OUTPUT_ROOT}"
cp -a "${IAD_DATA_DIR}/." "${SCRATCH_DATA}/"

RUN_ASSET_ROOT="${ASSET_ROOT}"
if [[ "${TASK_METHOD}" == "efficientad" ]]; then
  if [[ ! -f "${ASSET_ROOT}/ASSET_MANIFEST.json" ]]; then
    echo "Missing EfficientAD asset manifest: ${ASSET_ROOT}/ASSET_MANIFEST.json" >&2
    exit 2
  fi
  mkdir -p \
    "${SCRATCH_ASSETS}/efficientad/weights/efficientad_pretrained_weights"
  cp "${ASSET_ROOT}/ASSET_MANIFEST.json" "${SCRATCH_ASSETS}/"
  cp \
    "${ASSET_ROOT}/efficientad/weights/efficientad_pretrained_weights/pretrained_teacher_medium.pth" \
    "${SCRATCH_ASSETS}/efficientad/weights/efficientad_pretrained_weights/"
  tar -xzf "${ASSET_ROOT}/efficientad/imagenette2.tgz" \
    -C "${SCRATCH_ASSETS}/efficientad"
  RUN_ASSET_ROOT="${SCRATCH_ASSETS}"
fi

export PYTHONNOUSERSITE=1
export TORCH_HOME="${TORCH_HOME_ROOT}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} method=${TASK_METHOD}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

cd "${PROJECT_ROOT}"
RUN_ARGS=(
  scripts/run_iad_full_training_berzelius.py
  --config "${CONFIG}"
  --task-kind gpu
  --task-id "${SLURM_ARRAY_TASK_ID}"
  --dataset-dir "${SCRATCH_DATA}"
  --asset-root "${RUN_ASSET_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --device cuda
  --workers "${WORKERS}"
  --resume
)
if [[ "${IAD_FULL_SMOKE:-0}" == "1" ]]; then
  RUN_ARGS+=(--smoke)
fi
mamba run -p "${ENV_PREFIX}" python "${RUN_ARGS[@]}"
