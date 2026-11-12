#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${CFLOW_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
CFLOW_ENV_PREFIX=${CFLOW_ENV_PREFIX:-"${PROJECT_ROOT}/.envs/iad-cflow"}
CFLOW_OUTPUT_ROOT=${CFLOW_OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs/iad_cflow_official_matrix"}
CFLOW_CONFIG=${CFLOW_CONFIG:-"${PROJECT_ROOT}/configs/iad_cflow_official_matrix.json"}

module load Miniforge3/24.7.1-2-hpc1-bdist
cd "${PROJECT_ROOT}"
mamba run -p "${CFLOW_ENV_PREFIX}" python \
  scripts/collect_iad_cflow_official.py \
  --config "${CFLOW_CONFIG}" \
  --output-root "${CFLOW_OUTPUT_ROOT}" \
  "$@"
