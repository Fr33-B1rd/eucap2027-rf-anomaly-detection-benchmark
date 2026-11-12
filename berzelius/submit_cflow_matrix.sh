#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${CFLOW_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
: "${BERZELIUS_ACCOUNT:?Set BERZELIUS_ACCOUNT to the Slurm project account}"
: "${IAD_DATA_DIR:?Set IAD_DATA_DIR to the project-storage IAD directory}"

export CFLOW_PROJECT_ROOT="${PROJECT_ROOT}"
export CFLOW_ENV_PREFIX=${CFLOW_ENV_PREFIX:-"${PROJECT_ROOT}/.envs/iad-cflow"}
export CFLOW_TORCH_HOME=${CFLOW_TORCH_HOME:-"${PROJECT_ROOT}/.cache/torch"}
export CFLOW_OUTPUT_ROOT=${CFLOW_OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs/iad_cflow_official_matrix"}
export CFLOW_CONFIG=${CFLOW_CONFIG:-"${PROJECT_ROOT}/configs/iad_cflow_official_matrix.json"}
export CFLOW_WORKERS=${CFLOW_WORKERS:-8}
export IAD_DATA_DIR

module load Miniforge3/24.7.1-2-hpc1-bdist
mkdir -p "${PROJECT_ROOT}/logs" "${CFLOW_OUTPUT_ROOT}"

TASK_COUNT=$(mamba run -p "${CFLOW_ENV_PREFIX}" python \
  "${PROJECT_ROOT}/scripts/run_iad_cflow_official.py" \
  --config "${CFLOW_CONFIG}" --task-count)
MAX_CONCURRENT=${CFLOW_MAX_CONCURRENT:-4}
ARRAY_SPEC=${CFLOW_ARRAY_SPEC:-"0-$((TASK_COUNT - 1))%${MAX_CONCURRENT}"}
BERZELIUS_SYSTEM=${BERZELIUS_SYSTEM:-ampere}

SBATCH_ARGS=(
  --account "${BERZELIUS_ACCOUNT}"
  --array "${ARRAY_SPEC}"
  --export ALL
)
if [[ "${BERZELIUS_SYSTEM}" == "ampere" ]]; then
  SBATCH_ARGS+=(--constraint "${CFLOW_CONSTRAINT:-thin}")
elif [[ "${BERZELIUS_SYSTEM}" != "hopper" ]]; then
  echo "BERZELIUS_SYSTEM must be ampere or hopper" >&2
  exit 2
fi

echo "Submitting ${TASK_COUNT}-cell matrix with array ${ARRAY_SPEC}"
cd "${PROJECT_ROOT}"
sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/cflow_matrix.sbatch"
