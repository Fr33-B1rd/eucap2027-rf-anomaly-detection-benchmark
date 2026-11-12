#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${IAD_FINAL_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
: "${BERZELIUS_ACCOUNT:?Set BERZELIUS_ACCOUNT to the Slurm project account}"
: "${IAD_DATA_DIR:?Set IAD_DATA_DIR to the project-storage IAD directory}"

export CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV-}
export IAD_FINAL_PROJECT_ROOT="${PROJECT_ROOT}"
export IAD_FINAL_CONFIG="${PROJECT_ROOT}/configs/iad_patchcore_resolution128_scaled_seed2026_berzelius.json"
export IAD_FINAL_OUTPUT_ROOT=${IAD_PATCHCORE_SCALED_OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs/iad_patchcore_resolution128_scaled_seed2026"}
export IAD_FINAL_ARRAY_SPEC=${IAD_PATCHCORE_SCALED_ARRAY_SPEC:-"0-3%1"}
export IAD_FINAL_MAX_CONCURRENT=1
export IAD_FINAL_WORKERS=${IAD_FINAL_WORKERS:-8}
export IAD_DATA_DIR

echo "Submitting four PatchCore 128 scaled-budget pilot cells sequentially"
bash "${SCRIPT_DIR}/submit_iad_final_matrix.sh"
