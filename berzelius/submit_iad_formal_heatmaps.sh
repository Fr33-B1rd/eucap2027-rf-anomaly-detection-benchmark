#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${IAD_HEATMAP_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
: "${BERZELIUS_ACCOUNT:?Set BERZELIUS_ACCOUNT to the Slurm project account}"
: "${IAD_DATA_DIR:?Set IAD_DATA_DIR to the project-storage IAD directory}"

export IAD_HEATMAP_PROJECT_ROOT="${PROJECT_ROOT}"
export IAD_HEATMAP_ENV_PREFIX=${IAD_HEATMAP_ENV_PREFIX:-"${PROJECT_ROOT}/.envs/iad-cflow"}
export IAD_HEATMAP_CONFIG=${IAD_HEATMAP_CONFIG:-"${PROJECT_ROOT}/configs/iad_common_normal_berzelius.json"}
export IAD_HEATMAP_SELECTION=${IAD_HEATMAP_SELECTION:-"${PROJECT_ROOT}/configs/iad_formal_heatmap_selection.json"}
export IAD_HEATMAP_OUTPUT_ROOT=${IAD_HEATMAP_OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs/iad_formal_heatmaps"}
export IAD_HEATMAP_EFFICIENTAD_ROOT=${IAD_HEATMAP_EFFICIENTAD_ROOT:-"${PROJECT_ROOT}/.cache/iad_assets"}
export IAD_HEATMAP_TORCH_HOME=${IAD_HEATMAP_TORCH_HOME:-"${PROJECT_ROOT}/.cache/torch"}
export IAD_HEATMAP_WORKERS=${IAD_HEATMAP_WORKERS:-8}
export IAD_HEATMAP_SEED=${IAD_HEATMAP_SEED:-2026}
export IAD_DATA_DIR
export CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV-}

module load Miniforge3/24.7.1-2-hpc1-bdist
mkdir -p "${PROJECT_ROOT}/logs" "${IAD_HEATMAP_OUTPUT_ROOT}"
for required in \
  "${IAD_HEATMAP_ENV_PREFIX}/bin/python" \
  "${IAD_HEATMAP_CONFIG}" \
  "${IAD_HEATMAP_SELECTION}" \
  "${IAD_HEATMAP_EFFICIENTAD_ROOT}/efficientad_pretrained_weights/pretrained_teacher_medium.pth"
do
  if [[ ! -e "${required}" ]]; then
    echo "Missing prerequisite: ${required}" >&2
    exit 2
  fi
done
if [[ ! -d "${IAD_HEATMAP_EFFICIENTAD_ROOT}/imagenette2/train" ]]; then
  echo "Missing ImageNette train directory" >&2
  exit 2
fi

mamba run -p "${IAD_HEATMAP_ENV_PREFIX}" python -c \
  "import FrEIA, sklearn, torch, torchvision; print(torch.__version__, torchvision.__version__)"
TASK_COUNT=$(mamba run -p "${IAD_HEATMAP_ENV_PREFIX}" python \
  "${PROJECT_ROOT}/scripts/run_iad_formal_heatmap_export.py" \
  --seed "${IAD_HEATMAP_SEED}" --task-count)
if [[ "${TASK_COUNT}" != "5" ]]; then
  echo "Expected 5 method cells, got ${TASK_COUNT}" >&2
  exit 2
fi

MAX_CONCURRENT=${IAD_HEATMAP_MAX_CONCURRENT:-5}
ARRAY_SPEC=${IAD_HEATMAP_ARRAY_SPEC:-"0-4%${MAX_CONCURRENT}"}
BERZELIUS_SYSTEM=${BERZELIUS_SYSTEM:-ampere}
SBATCH_ARGS=(--account "${BERZELIUS_ACCOUNT}" --array "${ARRAY_SPEC}" --export ALL)
if [[ "${BERZELIUS_SYSTEM}" == "ampere" ]]; then
  SBATCH_ARGS+=(--constraint "${IAD_HEATMAP_CONSTRAINT:-thin}")
elif [[ "${BERZELIUS_SYSTEM}" != "hopper" ]]; then
  echo "BERZELIUS_SYSTEM must be ampere or hopper" >&2
  exit 2
fi

echo "Submitting formal IAD heatmap cells ${ARRAY_SPEC}, seed ${IAD_HEATMAP_SEED}"
cd "${PROJECT_ROOT}"
sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/iad_formal_heatmaps.sbatch"
