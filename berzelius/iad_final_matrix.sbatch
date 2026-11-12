#!/bin/bash
#SBATCH --job-name=iad-final
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${IAD_FINAL_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
: "${IAD_DATA_DIR:?Set IAD_DATA_DIR to the project-storage IAD directory}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"

IAD_FINAL_ENV_PREFIX=${IAD_FINAL_ENV_PREFIX:-"${PROJECT_ROOT}/.envs/iad-final"}
IAD_FINAL_TORCH_HOME=${IAD_FINAL_TORCH_HOME:-"${PROJECT_ROOT}/.cache/torch"}
IAD_FINAL_ASSET_ROOT=${IAD_FINAL_ASSET_ROOT:-"${PROJECT_ROOT}/.cache/iad_assets"}
IAD_FINAL_OUTPUT_ROOT=${IAD_FINAL_OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs/iad_final_berzelius"}
IAD_FINAL_CONFIG=${IAD_FINAL_CONFIG:-"${PROJECT_ROOT}/configs/iad_final_berzelius_matrix.json"}
IAD_FINAL_WORKERS=${IAD_FINAL_WORKERS:-8}

module load Miniforge3/24.7.1-2-hpc1-bdist
if [[ ! -x "${IAD_FINAL_ENV_PREFIX}/bin/python" ]]; then
  echo "Missing environment; run prepare_iad_final_env.sh first" >&2
  exit 2
fi
if [[ ! -d "${IAD_DATA_DIR}" ]]; then
  echo "IAD_DATA_DIR is not a directory: ${IAD_DATA_DIR}" >&2
  exit 2
fi
if [[ ! -f "${IAD_FINAL_CONFIG}" ]]; then
  echo "Missing final matrix config: ${IAD_FINAL_CONFIG}" >&2
  exit 2
fi

SCRATCH_ROOT="/scratch/local/${SLURM_JOB_ID}/iad-final"
SCRATCH_DATA="${SCRATCH_ROOT}/IAD"
SCRATCH_ASSETS="${SCRATCH_ROOT}/assets"
mkdir -p "${SCRATCH_DATA}" "${IAD_FINAL_OUTPUT_ROOT}"
cp -a "${IAD_DATA_DIR}/." "${SCRATCH_DATA}/"

# Tasks 20--39 are EfficientAD. Stage ImageNette on node-local NVMe to avoid
# 70,000 steps of small-file reads from shared project storage.
RUN_ASSET_ROOT="${IAD_FINAL_ASSET_ROOT}"
if (( SLURM_ARRAY_TASK_ID >= 20 && SLURM_ARRAY_TASK_ID <= 39 )); then
  if [[ ! -f "${IAD_FINAL_ASSET_ROOT}/ASSET_MANIFEST.json" ]]; then
    echo "Missing EfficientAD assets; run prepare_iad_final_env.sh first" >&2
    exit 2
  fi
  mkdir -p \
    "${SCRATCH_ASSETS}/efficientad/weights/efficientad_pretrained_weights"
  cp "${IAD_FINAL_ASSET_ROOT}/ASSET_MANIFEST.json" "${SCRATCH_ASSETS}/"
  cp \
    "${IAD_FINAL_ASSET_ROOT}/efficientad/weights/efficientad_pretrained_weights/pretrained_teacher_medium.pth" \
    "${SCRATCH_ASSETS}/efficientad/weights/efficientad_pretrained_weights/"
  tar -xzf "${IAD_FINAL_ASSET_ROOT}/efficientad/imagenette2.tgz" \
    -C "${SCRATCH_ASSETS}/efficientad"
  RUN_ASSET_ROOT="${SCRATCH_ASSETS}"
fi

export PYTHONNOUSERSITE=1
export TORCH_HOME="${IAD_FINAL_TORCH_HOME}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

cd "${PROJECT_ROOT}"
RUN_ARGS=(
  scripts/run_iad_final_berzelius.py
  --config "${IAD_FINAL_CONFIG}"
  --task-id "${SLURM_ARRAY_TASK_ID}"
  --dataset-dir "${SCRATCH_DATA}"
  --asset-root "${RUN_ASSET_ROOT}"
  --output-root "${IAD_FINAL_OUTPUT_ROOT}"
  --device cuda
  --workers "${IAD_FINAL_WORKERS}"
  --resume
)
if [[ "${IAD_FINAL_SMOKE:-0}" == "1" ]]; then
  RUN_ARGS+=(--smoke)
fi
mamba run -p "${IAD_FINAL_ENV_PREFIX}" python "${RUN_ARGS[@]}"
