#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${IAD_VAE1000_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
: "${BERZELIUS_ACCOUNT:?Set BERZELIUS_ACCOUNT to the Slurm project account}"
: "${IAD_DATA_DIR:?Set IAD_DATA_DIR to the project-storage IAD directory}"

export CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV-}
export IAD_VAE1000_PROJECT_ROOT="${PROJECT_ROOT}"
export IAD_VAE1000_ENV_PREFIX=${IAD_VAE1000_ENV_PREFIX:-"${PROJECT_ROOT}/.envs/iad-final"}
export IAD_VAE1000_CONFIG=${IAD_VAE1000_CONFIG:-"${PROJECT_ROOT}/configs/iad_vae_per1000_berzelius.json"}
export IAD_VAE1000_OUTPUT_ROOT=${IAD_VAE1000_OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs/iad_vae_per1000_berzelius"}
export IAD_VAE1000_WORKERS=${IAD_VAE1000_WORKERS:-8}
export IAD_DATA_DIR

if [[ ! -x "${IAD_VAE1000_ENV_PREFIX}/bin/python" ]]; then
  echo "Missing environment; run berzelius/prepare_iad_final_env.sh first" >&2
  exit 2
fi
mkdir -p "${PROJECT_ROOT}/logs" "${IAD_VAE1000_OUTPUT_ROOT}"
TASK_COUNT=$("${IAD_VAE1000_ENV_PREFIX}/bin/python" \
  "${PROJECT_ROOT}/scripts/run_iad_vae_epoch_ablation.py" \
  --config "${IAD_VAE1000_CONFIG}" --task-count)
MAX_CONCURRENT=${IAD_VAE1000_MAX_CONCURRENT:-4}
ARRAY_SPEC=${IAD_VAE1000_ARRAY_SPEC:-"0-$((TASK_COUNT - 1))%${MAX_CONCURRENT}"}
BERZELIUS_SYSTEM=${BERZELIUS_SYSTEM:-ampere}

SBATCH_ARGS=(
  --account "${BERZELIUS_ACCOUNT}"
  --array "${ARRAY_SPEC}"
  --export ALL
)
if [[ "${BERZELIUS_SYSTEM}" == "ampere" ]]; then
  SBATCH_ARGS+=(--constraint "${IAD_VAE1000_CONSTRAINT:-thin}")
elif [[ "${BERZELIUS_SYSTEM}" != "hopper" ]]; then
  echo "BERZELIUS_SYSTEM must be ampere or hopper" >&2
  exit 2
fi

echo "Submitting ${TASK_COUNT}-cell VAE-PER epoch ablation: ${ARRAY_SPEC}"
cd "${PROJECT_ROOT}"
sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/iad_vae_per1000.sbatch"
