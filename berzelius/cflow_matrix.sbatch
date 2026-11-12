#!/bin/bash
#SBATCH --job-name=iad-cflow
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${CFLOW_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
: "${IAD_DATA_DIR:?Set IAD_DATA_DIR to the project-storage IAD directory}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"

CFLOW_ENV_PREFIX=${CFLOW_ENV_PREFIX:-"${PROJECT_ROOT}/.envs/iad-cflow"}
CFLOW_TORCH_HOME=${CFLOW_TORCH_HOME:-"${PROJECT_ROOT}/.cache/torch"}
CFLOW_OUTPUT_ROOT=${CFLOW_OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs/iad_cflow_official_matrix"}
CFLOW_CONFIG=${CFLOW_CONFIG:-"${PROJECT_ROOT}/configs/iad_cflow_official_matrix.json"}
CFLOW_WORKERS=${CFLOW_WORKERS:-8}

module load Miniforge3/24.7.1-2-hpc1-bdist
if [[ ! -x "${CFLOW_ENV_PREFIX}/bin/python" ]]; then
  echo "Missing environment ${CFLOW_ENV_PREFIX}; run prepare_cflow_env.sh first" >&2
  exit 2
fi
if [[ ! -d "${IAD_DATA_DIR}" ]]; then
  echo "Missing IAD data directory: ${IAD_DATA_DIR}" >&2
  exit 2
fi

SCRATCH_ROOT="/scratch/local/${SLURM_JOB_ID}/iad-cflow"
SCRATCH_DATA="${SCRATCH_ROOT}/IAD"
mkdir -p "${SCRATCH_DATA}" "${CFLOW_OUTPUT_ROOT}"
cp -a "${IAD_DATA_DIR}/." "${SCRATCH_DATA}/"

export PYTHONNOUSERSITE=1
export TORCH_HOME="${CFLOW_TORCH_HOME}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
echo "data=${SCRATCH_DATA} output=${CFLOW_OUTPUT_ROOT}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

cd "${PROJECT_ROOT}"
RUN_ARGS=(
  scripts/run_iad_cflow_official.py \
  --config "${CFLOW_CONFIG}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --dataset-dir "${SCRATCH_DATA}" \
  --output-root "${CFLOW_OUTPUT_ROOT}" \
  --device cuda \
  --workers "${CFLOW_WORKERS}" \
  --resume
)
if [[ "${CFLOW_SMOKE:-0}" == "1" ]]; then
  RUN_ARGS+=(--smoke)
fi
mamba run -p "${CFLOW_ENV_PREFIX}" python "${RUN_ARGS[@]}"
