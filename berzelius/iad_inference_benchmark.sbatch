#!/bin/bash
#SBATCH --job-name=iad-benchmark
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${IAD_BENCHMARK_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
: "${IAD_DATA_DIR:?Set IAD_DATA_DIR to the project-storage IAD directory}"

# The shared environment contains the locked PyTorch stack and FrEIA, so it can
# execute all four benchmark methods in one software environment.
IAD_BENCHMARK_ENV_PREFIX=${IAD_BENCHMARK_ENV_PREFIX:-"${PROJECT_ROOT}/.envs/iad-final"}
IAD_BENCHMARK_TORCH_HOME=${IAD_BENCHMARK_TORCH_HOME:-"${PROJECT_ROOT}/.cache/torch"}
IAD_BENCHMARK_ASSET_ROOT=${IAD_BENCHMARK_ASSET_ROOT:-"${PROJECT_ROOT}/.cache/iad_assets"}
IAD_BENCHMARK_CONFIG=${IAD_BENCHMARK_CONFIG:-"${PROJECT_ROOT}/configs/iad_inference_benchmark.json"}
IAD_BENCHMARK_FINAL_CONFIG=${IAD_BENCHMARK_FINAL_CONFIG:-"${PROJECT_ROOT}/configs/iad_final_berzelius_matrix.json"}
IAD_BENCHMARK_CFLOW_CONFIG=${IAD_BENCHMARK_CFLOW_CONFIG:-"${PROJECT_ROOT}/configs/iad_cflow_official_matrix.json"}
IAD_BENCHMARK_METHODS=${IAD_BENCHMARK_METHODS:-}
IAD_BENCHMARK_OUTPUT=${IAD_BENCHMARK_OUTPUT:-"${PROJECT_ROOT}/outputs/iad_inference_benchmark/benchmark.json"}

module load Miniforge3/24.7.1-2-hpc1-bdist
if [[ ! -x "${IAD_BENCHMARK_ENV_PREFIX}/bin/python" ]]; then
  echo "Missing benchmark environment; run berzelius/prepare_iad_final_env.sh first" >&2
  exit 2
fi
if [[ ! -d "${IAD_DATA_DIR}" ]]; then
  echo "IAD_DATA_DIR is not a directory: ${IAD_DATA_DIR}" >&2
  exit 2
fi
if [[ ! -f "${IAD_BENCHMARK_ASSET_ROOT}/ASSET_MANIFEST.json" ]]; then
  echo "Missing EfficientAD asset manifest under ${IAD_BENCHMARK_ASSET_ROOT}" >&2
  exit 2
fi

SCRATCH_ROOT="/scratch/local/${SLURM_JOB_ID}/iad-benchmark"
SCRATCH_DATA="${SCRATCH_ROOT}/IAD"
SCRATCH_ASSETS="${SCRATCH_ROOT}/assets"
mkdir -p \
  "${SCRATCH_DATA}" \
  "${SCRATCH_ASSETS}" \
  "$(dirname -- "${IAD_BENCHMARK_OUTPUT}")"
cp -a "${IAD_DATA_DIR}/." "${SCRATCH_DATA}/"
if [[ -z "${IAD_BENCHMARK_METHODS}" || " ${IAD_BENCHMARK_METHODS} " == *" efficientad "* ]]; then
  mkdir -p "${SCRATCH_ASSETS}/efficientad/weights/efficientad_pretrained_weights"
  cp "${IAD_BENCHMARK_ASSET_ROOT}/ASSET_MANIFEST.json" "${SCRATCH_ASSETS}/"
  cp \
    "${IAD_BENCHMARK_ASSET_ROOT}/efficientad/weights/efficientad_pretrained_weights/pretrained_teacher_medium.pth" \
    "${SCRATCH_ASSETS}/efficientad/weights/efficientad_pretrained_weights/"
  tar -xzf "${IAD_BENCHMARK_ASSET_ROOT}/efficientad/imagenette2.tgz" \
    -C "${SCRATCH_ASSETS}/efficientad"
fi

export PYTHONNOUSERSITE=1
export TORCH_HOME="${IAD_BENCHMARK_TORCH_HOME}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

echo "host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

RUN_ARGS=(
  scripts/benchmark_iad_inference.py
  --benchmark-config "${IAD_BENCHMARK_CONFIG}"
  --final-config "${IAD_BENCHMARK_FINAL_CONFIG}"
  --cflow-config "${IAD_BENCHMARK_CFLOW_CONFIG}"
  --dataset-dir "${SCRATCH_DATA}"
  --asset-root "${SCRATCH_ASSETS}"
  --output "${IAD_BENCHMARK_OUTPUT}"
  --device cuda
)
if [[ -n "${IAD_BENCHMARK_METHODS}" ]]; then
  read -r -a BENCHMARK_METHOD_ARRAY <<< "${IAD_BENCHMARK_METHODS}"
  RUN_ARGS+=(--methods "${BENCHMARK_METHOD_ARRAY[@]}")
fi
if [[ "${IAD_BENCHMARK_SMOKE:-0}" == "1" ]]; then
  RUN_ARGS+=(--smoke)
fi

cd "${PROJECT_ROOT}"
"${IAD_BENCHMARK_ENV_PREFIX}/bin/python" "${RUN_ARGS[@]}"
