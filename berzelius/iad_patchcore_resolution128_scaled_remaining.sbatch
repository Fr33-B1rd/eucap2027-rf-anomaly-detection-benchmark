#!/bin/bash
#SBATCH --job-name=iad-pc128-rem
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${IAD_FINAL_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
: "${IAD_DATA_DIR:?Set IAD_DATA_DIR to the project-storage IAD directory}"

ENV_PREFIX=${IAD_FINAL_ENV_PREFIX:-"${PROJECT_ROOT}/.envs/iad-final"}
TORCH_CACHE=${IAD_FINAL_TORCH_HOME:-"${PROJECT_ROOT}/.cache/torch"}
CONFIG=${IAD_FINAL_CONFIG:-"${PROJECT_ROOT}/configs/iad_patchcore_resolution128_scaled_seeds2027_2030_berzelius.json"}
OUTPUT_ROOT=${IAD_FINAL_OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs/iad_patchcore_resolution128_scaled_seeds2027_2030"}
WORKERS=${IAD_FINAL_WORKERS:-8}

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
  echo "Missing PatchCore config: ${CONFIG}" >&2
  exit 2
fi

TASK_COUNT=$(mamba run -p "${ENV_PREFIX}" python \
  "${PROJECT_ROOT}/scripts/run_iad_final_berzelius.py" \
  --config "${CONFIG}" --task-count)
if [[ "${TASK_COUNT}" != "16" ]]; then
  echo "Expected 16 PatchCore cells, got ${TASK_COUNT}" >&2
  exit 2
fi

SCRATCH_ROOT="/scratch/local/${SLURM_JOB_ID}/iad-pc128-remaining"
SCRATCH_DATA="${SCRATCH_ROOT}/IAD"
mkdir -p "${SCRATCH_DATA}" "${OUTPUT_ROOT}"
cp -a "${IAD_DATA_DIR}/." "${SCRATCH_DATA}/"

export PYTHONNOUSERSITE=1
export TORCH_HOME="${TORCH_CACHE}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

echo "host=$(hostname) job=${SLURM_JOB_ID} tasks=${TASK_COUNT}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

cd "${PROJECT_ROOT}"
for (( TASK_ID=0; TASK_ID<TASK_COUNT; TASK_ID++ )); do
  echo "Running PatchCore scaled-budget cell $((TASK_ID + 1))/${TASK_COUNT}"
  mamba run -p "${ENV_PREFIX}" python \
    scripts/run_iad_final_berzelius.py \
    --config "${CONFIG}" \
    --task-id "${TASK_ID}" \
    --dataset-dir "${SCRATCH_DATA}" \
    --asset-root "${PROJECT_ROOT}/.cache/iad_assets" \
    --output-root "${OUTPUT_ROOT}" \
    --device cuda \
    --workers "${WORKERS}" \
    --resume
done

echo "Completed all ${TASK_COUNT} PatchCore scaled-budget cells"
