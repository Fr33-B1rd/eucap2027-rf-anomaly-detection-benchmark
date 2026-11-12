#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${IAD_PCA_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
: "${BERZELIUS_ACCOUNT:?Set BERZELIUS_ACCOUNT to the Slurm project account}"
: "${IAD_DATA_DIR:?Set IAD_DATA_DIR to the project-storage IAD directory}"

export CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV-}
export IAD_PCA_PROJECT_ROOT="${PROJECT_ROOT}"
export IAD_PCA_ENV_PREFIX=${IAD_PCA_ENV_PREFIX:-"${PROJECT_ROOT}/.envs/iad-final"}
export IAD_PCA_CONFIG=${IAD_PCA_CONFIG:-"${PROJECT_ROOT}/configs/iad_pca64_berzelius.json"}
export IAD_PCA_OUTPUT_ROOT=${IAD_PCA_OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs/iad_pca64_berzelius"}
export IAD_DATA_DIR

if [[ ! -x "${IAD_PCA_ENV_PREFIX}/bin/python" ]]; then
  echo "Missing environment; run berzelius/prepare_iad_final_env.sh first" >&2
  exit 2
fi
if [[ ! -f "${IAD_PCA_CONFIG}" ]]; then
  echo "Missing PCA config: ${IAD_PCA_CONFIG}" >&2
  exit 2
fi
mkdir -p "${PROJECT_ROOT}/logs" "${IAD_PCA_OUTPUT_ROOT}"

TASK_COUNT=$("${IAD_PCA_ENV_PREFIX}/bin/python" \
  "${PROJECT_ROOT}/scripts/run_iad_pca_baseline.py" \
  --config "${IAD_PCA_CONFIG}" --task-count)
if [[ "${TASK_COUNT}" != "20" ]]; then
  echo "Expected 20 PCA cells, got ${TASK_COUNT}" >&2
  exit 2
fi
if [[ -n "${IAD_PCA_TASK_ID:-}" ]] && \
   { (( IAD_PCA_TASK_ID < 0 )) || (( IAD_PCA_TASK_ID >= TASK_COUNT )); }; then
  echo "IAD_PCA_TASK_ID must be in [0, $((TASK_COUNT - 1))]" >&2
  exit 2
fi

PARTITION=${IAD_PCA_PARTITION:-berzelius-cpu}
SBATCH_ARGS=(
  --account "${BERZELIUS_ACCOUNT}"
  --partition "${PARTITION}"
  --export ALL
)

if [[ -n "${IAD_PCA_TASK_ID:-}" ]]; then
  echo "Submitting formal PCA validation cell ${IAD_PCA_TASK_ID} on ${PARTITION}"
else
  echo "Submitting all ${TASK_COUNT} PCA cells sequentially on ${PARTITION}"
fi
cd "${PROJECT_ROOT}"
sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/iad_pca64.sbatch"
