#!/bin/bash
#SBATCH --job-name=iad-common-pca
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
PROJECT_ROOT=${IAD_COMMON_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
: "${IAD_DATA_DIR:?Set IAD_DATA_DIR to the project-storage IAD directory}"

ENV_PREFIX=${IAD_COMMON_ENV_PREFIX:-"${PROJECT_ROOT}/.envs/iad-final"}
CONFIG=${IAD_COMMON_CONFIG:-"${PROJECT_ROOT}/configs/iad_common_normal_berzelius.json"}
OUTPUT_ROOT=${IAD_COMMON_OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs/iad_common_normal_berzelius"}

export CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV-}
module load Miniforge3/24.7.1-2-hpc1-bdist
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "Missing environment; run berzelius/prepare_iad_final_env.sh first" >&2
  exit 2
fi

SCRATCH_ROOT="/scratch/local/${SLURM_JOB_ID}/iad-common-pca"
SCRATCH_DATA="${SCRATCH_ROOT}/IAD"
mkdir -p "${SCRATCH_DATA}" "${OUTPUT_ROOT}"
cp -a "${IAD_DATA_DIR}/." "${SCRATCH_DATA}/"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

TASK_COUNT=$(mamba run -p "${ENV_PREFIX}" python \
  "${PROJECT_ROOT}/scripts/run_iad_common_normal_berzelius.py" \
  --config "${CONFIG}" --task-kind cpu --task-count)
for (( TASK_ID=0; TASK_ID<TASK_COUNT; TASK_ID++ )); do
  mamba run -p "${ENV_PREFIX}" python \
    "${PROJECT_ROOT}/scripts/run_iad_common_normal_berzelius.py" \
    --config "${CONFIG}" \
    --task-kind cpu \
    --task-id "${TASK_ID}" \
    --dataset-dir "${SCRATCH_DATA}" \
    --output-root "${OUTPUT_ROOT}" \
    --device cpu \
    --workers 0 \
    --resume
done
