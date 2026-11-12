#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${IAD_FINAL_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
IAD_FINAL_ENV_PREFIX=${IAD_FINAL_ENV_PREFIX:-"${PROJECT_ROOT}/.envs/iad-final"}
IAD_FINAL_TORCH_HOME=${IAD_FINAL_TORCH_HOME:-"${PROJECT_ROOT}/.cache/torch"}
IAD_FINAL_ASSET_ROOT=${IAD_FINAL_ASSET_ROOT:-"${PROJECT_ROOT}/.cache/iad_assets"}

module load Miniforge3/24.7.1-2-hpc1-bdist
mkdir -p \
  "$(dirname -- "${IAD_FINAL_ENV_PREFIX}")" \
  "${IAD_FINAL_TORCH_HOME}" \
  "${IAD_FINAL_ASSET_ROOT}"

if [[ ! -x "${IAD_FINAL_ENV_PREFIX}/bin/python" ]]; then
  CONDA_OVERRIDE_CUDA=12.4 mamba create -y -p "${IAD_FINAL_ENV_PREFIX}" \
    -c conda-forge \
    python=3.11 \
    pip \
    "pytorch=2.6.*=cuda*" \
    "torchvision=0.21.*=cuda*" \
    "numpy=1.26.*" \
    "scipy=1.13.*" \
    "scikit-learn=1.7.*" \
    "matplotlib=3.10.*"
fi

mamba run -p "${IAD_FINAL_ENV_PREFIX}" python -m pip install \
  --requirement "${SCRIPT_DIR}/requirements-cflow.txt"

mamba run -p "${IAD_FINAL_ENV_PREFIX}" python - \
  "${IAD_FINAL_ASSET_ROOT}" <<'PY'
import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

root = Path(sys.argv[1]).resolve()
efficientad = root / "efficientad"
efficientad.mkdir(parents=True, exist_ok=True)

teacher_url = (
    "https://github.com/open-edge-platform/anomalib/releases/download/"
    "efficientad_pretrained_weights/efficientad_pretrained_weights.zip"
)
teacher_bundle_hash = "c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a"
teacher_hash = "f7356663c8e00ada12ae01fb8c8aad0a1de2f800f8eadf252a46d29bbdfdf718"
imagenette_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
imagenette_hash = "6cbfac238434d89fe99e651496f0812ebc7a10fa62bd42d6874042bf01de4efd"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path, expected: str) -> None:
    if path.is_file() and sha256(path) == expected:
        return
    temporary = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
    actual = sha256(temporary)
    if actual != expected:
        raise RuntimeError(f"checksum mismatch for {url}: {actual} != {expected}")
    os.replace(temporary, path)


teacher_bundle = efficientad / "efficientad_pretrained_weights.zip"
imagenette_bundle = efficientad / "imagenette2.tgz"
download(teacher_url, teacher_bundle, teacher_bundle_hash)
download(imagenette_url, imagenette_bundle, imagenette_hash)

weights_root = efficientad / "weights"
teacher_path = (
    weights_root
    / "efficientad_pretrained_weights"
    / "pretrained_teacher_medium.pth"
)
if not teacher_path.is_file() or sha256(teacher_path) != teacher_hash:
    weights_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(teacher_bundle) as archive:
        for member in archive.infolist():
            destination = (weights_root / member.filename).resolve()
            if weights_root.resolve() not in destination.parents:
                raise RuntimeError(f"unsafe zip member: {member.filename}")
        archive.extractall(weights_root)
if sha256(teacher_path) != teacher_hash:
    raise RuntimeError("extracted EfficientAD medium teacher checksum mismatch")

imagenette_root = efficientad / "imagenette2"
if not (imagenette_root / "train").is_dir():
    with tarfile.open(imagenette_bundle, "r:gz") as archive:
        for member in archive.getmembers():
            destination = (efficientad / member.name).resolve()
            if efficientad.resolve() not in destination.parents:
                raise RuntimeError(f"unsafe tar member: {member.name}")
        archive.extractall(efficientad)

manifest = {
    "teacher_bundle_url": teacher_url,
    "teacher_bundle_sha256": sha256(teacher_bundle),
    "teacher_medium_sha256": sha256(teacher_path),
    "imagenette_url": imagenette_url,
    "imagenette_archive_sha256": sha256(imagenette_bundle),
}
(root / "ASSET_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
PY

TORCH_HOME="${IAD_FINAL_TORCH_HOME}" \
mamba run -p "${IAD_FINAL_ENV_PREFIX}" python - <<'PY'
import torch
import torchvision
import FrEIA
from torchvision.models import (
    ResNet18_Weights,
    Wide_ResNet50_2_Weights,
    resnet18,
    wide_resnet50_2,
)

resnet18(weights=ResNet18_Weights.DEFAULT)
wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("FrEIA import OK")
print("RN18 and WRN50-2 weights cached")
PY

mamba list -p "${IAD_FINAL_ENV_PREFIX}" --explicit \
  > "${IAD_FINAL_ENV_PREFIX}/environment-explicit.txt"

echo "Environment: ${IAD_FINAL_ENV_PREFIX}"
echo "Torch cache: ${IAD_FINAL_TORCH_HOME}"
echo "Assets: ${IAD_FINAL_ASSET_ROOT}"
