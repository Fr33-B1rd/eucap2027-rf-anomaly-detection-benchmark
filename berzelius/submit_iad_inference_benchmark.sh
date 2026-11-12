#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${IAD_BENCHMARK_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
: "${BERZELIUS_ACCOUNT:?Set BERZELIUS_ACCOUNT to the Slurm project account}"
: "${IAD_DATA_DIR:?Set IAD_DATA_DIR to the project-storage IAD directory}"

export IAD_BENCHMARK_PROJECT_ROOT="${PROJECT_ROOT}"
export IAD_BENCHMARK_ENV_PREFIX=${IAD_BENCHMARK_ENV_PREFIX:-"${PROJECT_ROOT}/.envs/iad-final"}
export IAD_BENCHMARK_ASSET_ROOT=${IAD_BENCHMARK_ASSET_ROOT:-"${PROJECT_ROOT}/.cache/iad_assets"}
export IAD_BENCHMARK_CONFIG=${IAD_BENCHMARK_CONFIG:-"${PROJECT_ROOT}/configs/iad_inference_benchmark.json"}
export IAD_BENCHMARK_FINAL_CONFIG=${IAD_BENCHMARK_FINAL_CONFIG:-"${PROJECT_ROOT}/configs/iad_final_berzelius_matrix.json"}
export IAD_BENCHMARK_CFLOW_CONFIG=${IAD_BENCHMARK_CFLOW_CONFIG:-"${PROJECT_ROOT}/configs/iad_cflow_official_matrix.json"}
export IAD_BENCHMARK_METHODS=${IAD_BENCHMARK_METHODS:-}
if [[ ! -x "${IAD_BENCHMARK_ENV_PREFIX}/bin/python" ]]; then
  echo "Missing benchmark environment; run berzelius/prepare_iad_final_env.sh first" >&2
  exit 2
fi
for required in \
  "${IAD_BENCHMARK_CONFIG}" \
  "${IAD_BENCHMARK_FINAL_CONFIG}" \
  "${IAD_BENCHMARK_CFLOW_CONFIG}"
do
  if [[ ! -f "${required}" ]]; then
    echo "Missing benchmark prerequisite: ${required}" >&2
    exit 2
  fi
done
if [[ -z "${IAD_BENCHMARK_METHODS}" || " ${IAD_BENCHMARK_METHODS} " == *" efficientad "* ]]; then
  for required in \
    "${IAD_BENCHMARK_ASSET_ROOT}/ASSET_MANIFEST.json" \
    "${IAD_BENCHMARK_ASSET_ROOT}/efficientad/weights/efficientad_pretrained_weights/pretrained_teacher_medium.pth" \
    "${IAD_BENCHMARK_ASSET_ROOT}/efficientad/imagenette2.tgz"
  do
    if [[ ! -f "${required}" ]]; then
      echo "Missing EfficientAD benchmark prerequisite: ${required}" >&2
      exit 2
    fi
  done
fi

module load Miniforge3/24.7.1-2-hpc1-bdist
"${IAD_BENCHMARK_ENV_PREFIX}/bin/python" -c \
  "import FrEIA, torch, torchvision; print(torch.__version__, torchvision.__version__)"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/outputs/iad_inference_benchmark"
SBATCH_ARGS=(--account "${BERZELIUS_ACCOUNT}" --export ALL)
BERZELIUS_SYSTEM=${BERZELIUS_SYSTEM:-ampere}
if [[ "${BERZELIUS_SYSTEM}" == "ampere" ]]; then
  SBATCH_ARGS+=(--constraint "${IAD_BENCHMARK_CONSTRAINT:-thin}")
elif [[ "${BERZELIUS_SYSTEM}" != "hopper" ]]; then
  echo "BERZELIUS_SYSTEM must be ampere or hopper" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/iad_inference_benchmark.sbatch"
