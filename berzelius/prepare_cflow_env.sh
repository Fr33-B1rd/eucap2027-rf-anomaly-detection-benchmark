#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${CFLOW_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
CFLOW_ENV_PREFIX=${CFLOW_ENV_PREFIX:-"${PROJECT_ROOT}/.envs/iad-cflow"}
CFLOW_TORCH_HOME=${CFLOW_TORCH_HOME:-"${PROJECT_ROOT}/.cache/torch"}

module load Miniforge3/24.7.1-2-hpc1-bdist
mkdir -p "$(dirname -- "${CFLOW_ENV_PREFIX}")" "${CFLOW_TORCH_HOME}"

if [[ ! -x "${CFLOW_ENV_PREFIX}/bin/python" ]]; then
  CONDA_OVERRIDE_CUDA=12.4 mamba create -y -p "${CFLOW_ENV_PREFIX}" \
    -c conda-forge \
    python=3.11 \
    pip \
    "pytorch=2.6.*=cuda*" \
    "torchvision=0.21.*=cuda*" \
    "numpy=1.26.*" \
    "scipy=1.13.*" \
    "scikit-learn=1.7.*"
fi

mamba run -p "${CFLOW_ENV_PREFIX}" python -m pip install \
  --requirement "${SCRIPT_DIR}/requirements-cflow.txt"

TORCH_HOME="${CFLOW_TORCH_HOME}" mamba run -p "${CFLOW_ENV_PREFIX}" python - <<'PY'
import FrEIA
import torch
import torchvision
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("FrEIA import OK")
print("WRN50-2 weights cached")
PY

echo "Environment: ${CFLOW_ENV_PREFIX}"
echo "Torch cache: ${CFLOW_TORCH_HOME}"
