#!/bin/bash
#SBATCH --job-name=iad-vae1000
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${IAD_VAE1000_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
: "${IAD_DATA_DIR:?Set IAD_DATA_DIR to the project-storage IAD directory}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"

ENV_PREFIX=${IAD_VAE1000_ENV_PREFIX:-"${PROJECT_ROOT}/.envs/iad-final"}
CONFIG=${IAD_VAE1000_CONFIG:-"${PROJECT_ROOT}/configs/iad_vae_per1000_berzelius.json"}
OUTPUT_ROOT=${IAD_VAE1000_OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs/iad_vae_per1000_berzelius"}
WORKERS=${IAD_VAE1000_WORKERS:-8}

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

SCRATCH_ROOT="/scratch/local/${SLURM_JOB_ID}/iad-vae1000"
SCRATCH_DATA="${SCRATCH_ROOT}/IAD"
mkdir -p "${SCRATCH_DATA}" "${OUTPUT_ROOT}"
cp -a "${IAD_DATA_DIR}/." "${SCRATCH_DATA}/"

export PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

cd "${PROJECT_ROOT}"
RUN_ARGS=(
  scripts/run_iad_vae_epoch_ablation.py
  --config "${CONFIG}"
  --task-id "${SLURM_ARRAY_TASK_ID}"
  --dataset-dir "${SCRATCH_DATA}"
  --output-root "${OUTPUT_ROOT}"
  --device cuda
  --workers "${WORKERS}"
  --resume
)
if [[ "${IAD_VAE1000_SMOKE:-0}" == "1" ]]; then
  RUN_ARGS+=(--smoke)
fi
"${ENV_PREFIX}/bin/python" "${RUN_ARGS[@]}"
