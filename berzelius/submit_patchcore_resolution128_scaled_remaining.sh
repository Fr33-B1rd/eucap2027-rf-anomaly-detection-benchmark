#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${IAD_FINAL_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
: "${BERZELIUS_ACCOUNT:?Set BERZELIUS_ACCOUNT to the Slurm project account}"
: "${IAD_DATA_DIR:?Set IAD_DATA_DIR to the project-storage IAD directory}"

export CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV-}
export IAD_FINAL_PROJECT_ROOT="${PROJECT_ROOT}"
export IAD_FINAL_ENV_PREFIX=${IAD_FINAL_ENV_PREFIX:-"${PROJECT_ROOT}/.envs/iad-final"}
export IAD_FINAL_TORCH_HOME=${IAD_FINAL_TORCH_HOME:-"${PROJECT_ROOT}/.cache/torch"}
export IAD_FINAL_CONFIG="${PROJECT_ROOT}/configs/iad_patchcore_resolution128_scaled_seeds2027_2030_berzelius.json"
export IAD_FINAL_OUTPUT_ROOT=${IAD_PATCHCORE_SCALED_OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs/iad_patchcore_resolution128_scaled_seeds2027_2030"}
export IAD_FINAL_WORKERS=${IAD_FINAL_WORKERS:-8}
export IAD_DATA_DIR

module load Miniforge3/24.7.1-2-hpc1-bdist
mkdir -p "${PROJECT_ROOT}/logs" "${IAD_FINAL_OUTPUT_ROOT}"
if [[ ! -x "${IAD_FINAL_ENV_PREFIX}/bin/python" ]]; then
  echo "Missing environment; run berzelius/prepare_iad_final_env.sh first" >&2
  exit 2
fi

TASK_COUNT=$(mamba run -p "${IAD_FINAL_ENV_PREFIX}" python \
  "${PROJECT_ROOT}/scripts/run_iad_final_berzelius.py" \
  --config "${IAD_FINAL_CONFIG}" --task-count)
if [[ "${TASK_COUNT}" != "16" ]]; then
  echo "Expected 16 PatchCore cells, got ${TASK_COUNT}" >&2
  exit 2
fi

BERZELIUS_SYSTEM=${BERZELIUS_SYSTEM:-ampere}
SBATCH_ARGS=(--account "${BERZELIUS_ACCOUNT}" --export ALL)
if [[ "${BERZELIUS_SYSTEM}" == "ampere" ]]; then
  SBATCH_ARGS+=(--constraint "${IAD_FINAL_CONSTRAINT:-thin}")
elif [[ "${BERZELIUS_SYSTEM}" != "hopper" ]]; then
  echo "BERZELIUS_SYSTEM must be ampere or hopper" >&2
  exit 2
fi

echo "Submitting one sequential GPU job for ${TASK_COUNT} PatchCore cells"
cd "${PROJECT_ROOT}"
sbatch "${SBATCH_ARGS[@]}" \
  "${SCRIPT_DIR}/iad_patchcore_resolution128_scaled_remaining.sbatch"
