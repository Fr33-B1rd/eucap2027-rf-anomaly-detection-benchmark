#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${IAD_BENCHMARK_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}

export IAD_BENCHMARK_PROJECT_ROOT="${PROJECT_ROOT}"
export IAD_BENCHMARK_FINAL_CONFIG="${PROJECT_ROOT}/configs/iad_vae_per1000_berzelius.json"
export IAD_BENCHMARK_METHODS="vae_per"
export IAD_BENCHMARK_OUTPUT=${IAD_BENCHMARK_OUTPUT:-"${PROJECT_ROOT}/outputs/iad_inference_benchmark_vae1000/benchmark_vae1000.json"}

exec bash "${SCRIPT_DIR}/submit_iad_inference_benchmark.sh"
