#!/usr/bin/env python
"""Download and verify external assets required by the IAD benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "iad_benchmark.json"
DEFAULT_ASSET_ROOT = REPOSITORY_ROOT / ".cache" / "iad_assets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_download(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_sha256 = expected_sha256.lower()
    if destination.is_file() and file_sha256(destination) == expected_sha256:
        print(f"[cached] {destination}", flush=True)
        return

    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    print(f"[download] {url}", flush=True)
    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    actual = file_sha256(temporary)
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"checksum mismatch for {url}: expected {expected_sha256}, got {actual}"
        )
    os.replace(temporary, destination)


def safe_member_path(root: Path, member_name: str) -> Path:
    candidate = (root / member_name).resolve()
    root = root.resolve()
    if not candidate.is_relative_to(root):
        raise RuntimeError(f"archive member escapes destination: {member_name}")
    return candidate


def extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            safe_member_path(destination, member.filename)
        source.extractall(destination)


def extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as source:
        for member in source.getmembers():
            target = safe_member_path(destination, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise RuntimeError(f"unsupported archive member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = source.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"could not read archive member: {member.name}")
            with extracted, target.open("wb") as output:
                shutil.copyfileobj(extracted, output)


def prepare_assets(
    config: dict[str, Any], asset_root: Path, *, force: bool = False
) -> dict[str, Any]:
    assets = config["assets"]
    method = config["methods"]["efficientad"]
    downloads = asset_root / "downloads"
    teacher_archive = downloads / "efficientad_pretrained_weights.zip"
    imagenette_archive = downloads / "imagenette2.tgz"

    verified_download(
        str(assets["teacher_bundle_url"]),
        teacher_archive,
        str(assets["teacher_bundle_sha256"]),
    )
    verified_download(
        str(assets["imagenette_url"]),
        imagenette_archive,
        str(assets["imagenette_sha256"]),
    )

    teacher_root = asset_root / "efficientad" / "weights"
    imagenette_parent = asset_root / "efficientad"
    teacher_path = asset_root / str(method["teacher_weights_relative"])
    imagenette_path = asset_root / str(method["imagenette_dir_relative"])

    if force or not teacher_path.is_file():
        extract_zip(teacher_archive, teacher_root)
    if force or not (imagenette_path / "train").is_dir():
        extract_tar(imagenette_archive, imagenette_parent)

    if not teacher_path.is_file():
        raise FileNotFoundError(f"teacher weights were not extracted to {teacher_path}")
    teacher_sha256 = file_sha256(teacher_path)
    expected_teacher_sha256 = str(method["teacher_weights_sha256"]).lower()
    if teacher_sha256 != expected_teacher_sha256:
        raise RuntimeError(
            "teacher weight checksum mismatch: "
            f"expected {expected_teacher_sha256}, got {teacher_sha256}"
        )
    if not (imagenette_path / "train").is_dir():
        raise FileNotFoundError(f"ImageNette train split is missing: {imagenette_path}")

    manifest = {
        "format_version": 1,
        "teacher_bundle": {
            "url": assets["teacher_bundle_url"],
            "sha256": assets["teacher_bundle_sha256"],
            "archive": str(teacher_archive.resolve()),
        },
        "teacher_weights": {
            "path": str(teacher_path.resolve()),
            "sha256": teacher_sha256,
        },
        "imagenette": {
            "url": assets["imagenette_url"],
            "sha256": assets["imagenette_sha256"],
            "path": str(imagenette_path.resolve()),
        },
    }
    manifest_path = asset_root / "ASSET_MANIFEST.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, manifest_path)
    return manifest


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = prepare_assets(config, args.asset_root.resolve(), force=args.force)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
