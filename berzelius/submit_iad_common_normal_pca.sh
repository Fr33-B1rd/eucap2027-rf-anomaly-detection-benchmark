#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${IAD_COMMON_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
: "${BERZELIUS_ACCOUNT:?Set BERZELIUS_ACCOUNT to the Slurm project account}"
: "${IAD_DATA_DIR:?Set IAD_DATA_DIR to the project-storage IAD directory}"

export IAD_COMMON_PROJECT_ROOT="${PROJECT_ROOT}"
export IAD_COMMON_ENV_PREFIX=${IAD_COMMON_ENV_PREFIX:-"${PROJECT_ROOT}/.envs/iad-final"}
export IAD_COMMON_CONFIG=${IAD_COMMON_CONFIG:-"${PROJECT_ROOT}/configs/iad_common_normal_berzelius.json"}
export IAD_COMMON_OUTPUT_ROOT=${IAD_COMMON_OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs/iad_common_normal_berzelius"}
export IAD_DATA_DIR
export CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV-}

if [[ ! -x "${IAD_COMMON_ENV_PREFIX}/bin/python" ]]; then
  echo "Missing environment; run berzelius/prepare_iad_final_env.sh first" >&2
  exit 2
fi
mkdir -p "${PROJECT_ROOT}/logs" "${IAD_COMMON_OUTPUT_ROOT}"

PARTITION=${IAD_COMMON_CPU_PARTITION:-berzelius-cpu}
echo "Submitting five sequential PCA seeds on ${PARTITION}"
cd "${PROJECT_ROOT}"
sbatch \
  --account "${BERZELIUS_ACCOUNT}" \
  --partition "${PARTITION}" \
  --export ALL \
  "${SCRIPT_DIR}/iad_common_normal_pca.sbatch"
