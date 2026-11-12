#!/bin/bash
#SBATCH --job-name=iad-pca64
#SBATCH --partition=berzelius-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${IAD_PCA_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
: "${IAD_DATA_DIR:?Set IAD_DATA_DIR to the project-storage IAD directory}"

ENV_PREFIX=${IAD_PCA_ENV_PREFIX:-"${PROJECT_ROOT}/.envs/iad-final"}
CONFIG=${IAD_PCA_CONFIG:-"${PROJECT_ROOT}/configs/iad_pca64_berzelius.json"}
OUTPUT_ROOT=${IAD_PCA_OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs/iad_pca64_berzelius"}

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
  echo "Missing PCA config: ${CONFIG}" >&2
  exit 2
fi

SCRATCH_ROOT="/scratch/local/${SLURM_JOB_ID}/iad-pca64"
SCRATCH_DATA="${SCRATCH_ROOT}/IAD"
mkdir -p "${SCRATCH_DATA}" "${OUTPUT_ROOT}"
cp -a "${IAD_DATA_DIR}/." "${SCRATCH_DATA}/"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

cd "${PROJECT_ROOT}"
RUN_ARGS=(
  scripts/run_iad_pca_baseline.py
  --config "${CONFIG}"
  --dataset-dir "${SCRATCH_DATA}"
  --output-root "${OUTPUT_ROOT}"
  --resume
)
if [[ -n "${IAD_PCA_TASK_ID:-}" ]]; then
  RUN_ARGS+=(--task-id "${IAD_PCA_TASK_ID}")
else
  RUN_ARGS+=(--all)
fi

echo "host=$(hostname) job=${SLURM_JOB_ID} cpus=${SLURM_CPUS_PER_TASK:-4}"
"${ENV_PREFIX}/bin/python" "${RUN_ARGS[@]}"
