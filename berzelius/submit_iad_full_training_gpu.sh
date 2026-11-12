#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${IAD_FULL_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
: "${BERZELIUS_ACCOUNT:?Set BERZELIUS_ACCOUNT to the Slurm project account}"
: "${IAD_DATA_DIR:?Set IAD_DATA_DIR to the project-storage IAD directory}"

export IAD_FULL_PROJECT_ROOT="${PROJECT_ROOT}"
export IAD_FULL_ENV_PREFIX=${IAD_FULL_ENV_PREFIX:-"${PROJECT_ROOT}/.envs/iad-final"}
export IAD_FULL_CONFIG=${IAD_FULL_CONFIG:-"${PROJECT_ROOT}/configs/iad_benchmark.json"}
export IAD_FULL_OUTPUT_ROOT=${IAD_FULL_OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs/iad_benchmark"}
export IAD_FULL_ASSET_ROOT=${IAD_FULL_ASSET_ROOT:-"${PROJECT_ROOT}/.cache/iad_assets"}
export IAD_FULL_TORCH_HOME=${IAD_FULL_TORCH_HOME:-"${PROJECT_ROOT}/.cache/torch"}
export IAD_FULL_WORKERS=${IAD_FULL_WORKERS:-8}
export IAD_DATA_DIR
export CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV-}

module load Miniforge3/24.7.1-2-hpc1-bdist
mkdir -p "${PROJECT_ROOT}/logs" "${IAD_FULL_OUTPUT_ROOT}"
if [[ ! -x "${IAD_FULL_ENV_PREFIX}/bin/python" ]]; then
  echo "Missing environment; run berzelius/prepare_iad_final_env.sh first" >&2
  exit 2
fi
mamba run -p "${IAD_FULL_ENV_PREFIX}" python -c \
  "import FrEIA, sklearn, torch, torchvision; print(torch.__version__, torchvision.__version__)"
for required in \
  "${IAD_FULL_ASSET_ROOT}/ASSET_MANIFEST.json" \
  "${IAD_FULL_ASSET_ROOT}/efficientad/weights/efficientad_pretrained_weights/pretrained_teacher_medium.pth" \
  "${IAD_FULL_ASSET_ROOT}/efficientad/imagenette2.tgz"
do
  if [[ ! -f "${required}" ]]; then
    echo "Missing EfficientAD prerequisite: ${required}" >&2
    exit 2
  fi
done

TASK_COUNT=$(mamba run -p "${IAD_FULL_ENV_PREFIX}" python \
  "${PROJECT_ROOT}/scripts/run_iad_full_training_berzelius.py" \
  --config "${IAD_FULL_CONFIG}" --task-kind gpu --task-count)
if [[ "${TASK_COUNT}" != "20" ]]; then
  echo "Expected 20 GPU method-by-seed cells, got ${TASK_COUNT}" >&2
  exit 2
fi

MAX_CONCURRENT=${IAD_FULL_MAX_CONCURRENT:-5}
ARRAY_SPEC=${IAD_FULL_ARRAY_SPEC:-"0-$((TASK_COUNT - 1))%${MAX_CONCURRENT}"}
BERZELIUS_SYSTEM=${BERZELIUS_SYSTEM:-ampere}
SBATCH_ARGS=(--account "${BERZELIUS_ACCOUNT}" --array "${ARRAY_SPEC}" --export ALL)
if [[ "${BERZELIUS_SYSTEM}" == "ampere" ]]; then
  SBATCH_ARGS+=(--constraint "${IAD_FULL_CONSTRAINT:-thin}")
elif [[ "${BERZELIUS_SYSTEM}" != "hopper" ]]; then
  echo "BERZELIUS_SYSTEM must be ampere or hopper" >&2
  exit 2
fi

echo "Submitting ${TASK_COUNT} full-training GPU cells with array ${ARRAY_SPEC}"
cd "${PROJECT_ROOT}"
sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/iad_full_training_gpu.sbatch"
