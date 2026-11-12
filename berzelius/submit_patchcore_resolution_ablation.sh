#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${IAD_FINAL_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
: "${BERZELIUS_ACCOUNT:?Set BERZELIUS_ACCOUNT to the Slurm project account}"
: "${IAD_DATA_DIR:?Set IAD_DATA_DIR to the project-storage IAD directory}"

# NSC's Miniforge module wrapper reads this variable while the calling scripts
# use `set -u`. Defining it, even when no environment is active, avoids the
# wrapper's "CONDA_DEFAULT_ENV: unbound variable" failure.
export CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV-}

RESOLUTIONS=${IAD_PATCHCORE_RESOLUTIONS:-"128 256"}
OUTPUT_BASE=${IAD_PATCHCORE_RESOLUTION_OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs/iad_patchcore_resolution_ablation"}
MAX_CONCURRENT=${IAD_PATCHCORE_RESOLUTION_MAX_CONCURRENT:-4}
ARRAY_SPEC=${IAD_PATCHCORE_RESOLUTION_ARRAY_SPEC:-"0-19%${MAX_CONCURRENT}"}
SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_iad_final_matrix.sh"

for INPUT_SIZE in ${RESOLUTIONS}; do
  case "${INPUT_SIZE}" in
    128|256) ;;
    *)
      echo "IAD_PATCHCORE_RESOLUTIONS may contain only 128 and/or 256" >&2
      exit 2
      ;;
  esac

  CONFIG="${PROJECT_ROOT}/configs/iad_patchcore_resolution${INPUT_SIZE}_berzelius.json"
  OUTPUT_ROOT="${OUTPUT_BASE}/input${INPUT_SIZE}"
  if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing locked resolution config: ${CONFIG}" >&2
    exit 2
  fi

  echo "Submitting PatchCore input-${INPUT_SIZE}: array ${ARRAY_SPEC} -> ${OUTPUT_ROOT}"
  IAD_FINAL_CONFIG="${CONFIG}" \
  IAD_FINAL_OUTPUT_ROOT="${OUTPUT_ROOT}" \
  IAD_FINAL_ARRAY_SPEC="${ARRAY_SPEC}" \
  IAD_FINAL_MAX_CONCURRENT="${MAX_CONCURRENT}" \
  IAD_FINAL_SMOKE="${IAD_PATCHCORE_RESOLUTION_SMOKE:-0}" \
    bash "${SUBMIT_SCRIPT}"
done
