#!/usr/bin/env python
"""Run one independently resumable cell of the legacy IAD method matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataio.iad import discover_iad_sources, safe_load_iad_pickle  # noqa: E402
from iad_detection.data import (  # noqa: E402
    IADTensorDataset,
    IndexedLabelDataset,
    build_normal_split,
)
from iad_detection.final_matrix import (  # noqa: E402
    FinalTask,
    canonical_config_hash,
    expand_final_tasks,
    load_final_config,
    resolve_method_config,
)
from iad_detection.metrics import evaluate_detection  # noqa: E402
from iad_detection.methods import build_detector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/iad_method_configs.json",
    )
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--task-count", action="store_true")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "outputs/iad_final_matrix",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def git_state() -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=REPOSITORY_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": "unknown", "dirty": None}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_snapshot_sha256() -> str:
    """Hash every source file that can affect a final-matrix run."""

    roots = (
        REPOSITORY_ROOT / "core",
        REPOSITORY_ROOT / "dataio" / "iad",
        REPOSITORY_ROOT / "iad_detection",
    )
    paths = [REPOSITORY_ROOT / "scripts" / "run_iad_final_matrix.py"]
    for root in roots:
        paths.extend(root.rglob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.as_posix()):
        relative = path.relative_to(REPOSITORY_ROOT).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_loader(
    dataset: IADTensorDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        IndexedLabelDataset(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
    )


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_scores(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def print_tasks(tasks: list[FinalTask]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(tasks[0].to_dict()),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(task.to_dict() for task in tasks)
    print(buffer.getvalue(), end="")


def smoke_test_indices(labels: np.ndarray) -> np.ndarray:
    normal = np.flatnonzero(labels == 0)[:16]
    anomaly = np.flatnonzero(labels == 1)[:16]
    return np.sort(np.concatenate((normal, anomaly))).astype(np.int64)


def adapt_smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    if "epochs" in result:
        result["epochs"] = 1
    if "max_steps" in result:
        result["max_steps"] = 2
    for key, value in (
        ("candidate_pool_size", 256),
        ("max_memory_bank_size", 64),
        ("sample_patches_per_image", 4),
    ):
        if key in result:
            result[key] = value
    return result


def main() -> None:
    args = parse_args()
    config = load_final_config(args.config)
    tasks = expand_final_tasks(config)
    if args.task_count:
        print(len(tasks))
        return
    if args.list_tasks:
        print_tasks(tasks)
        return
    if args.task_id is None or not 0 <= args.task_id < len(tasks):
        raise SystemExit(f"--task-id must be in [0, {len(tasks) - 1}]")
    if args.workers < 0:
        raise SystemExit("--workers cannot be negative")

    task = tasks[args.task_id]
    dataset_dir = (args.dataset_dir or Path(config["dataset_dir"])).expanduser()
    asset_root = (args.asset_root or REPOSITORY_ROOT / ".cache/iad_assets").expanduser()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    set_seed(task.seed)

    output_dir = args.output_root.expanduser() / task.relative_output_dir
    if args.smoke:
        output_dir = args.output_root.expanduser() / "smoke" / task.relative_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "run.json"
    scores_path = output_dir / "scores.npz"

    sources = {item.modulation: item for item in discover_iad_sources(dataset_dir)}
    if task.modulation not in sources:
        raise FileNotFoundError(
            f"missing {task.modulation}; found {sorted(sources)} in {dataset_dir}"
        )
    source = sources[task.modulation]
    source_digest = file_sha256(source.path)
    identity = {
        "protocol_version": config["protocol_version"],
        "config_sha256": canonical_config_hash(config),
        "code_snapshot_sha256": code_snapshot_sha256(),
        "task": task.to_dict(),
        "source_sha256": source_digest,
        "smoke": bool(args.smoke),
    }
    if run_path.exists():
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        if previous.get("identity") != identity:
            raise RuntimeError(f"existing result identity mismatch: {run_path}")
        if args.resume:
            print(f"[resume] completed task {task.task_id}: {run_path}", flush=True)
            return
        raise FileExistsError(f"result exists; pass --resume: {run_path}")

    started_utc = datetime.now(timezone.utc).isoformat()
    end_to_end_started = time.perf_counter()
    print(
        json.dumps({"task": task.to_dict(), "output_dir": str(output_dir)}),
        flush=True,
    )
    bundle = safe_load_iad_pickle(source.path)
    split = build_normal_split(
        bundle.train_data,
        calibration_fraction=float(config["evaluation"]["calibration_fraction"]),
        seed=task.seed,
    )
    fit_indices = split.fit_indices
    calibration_indices = split.calibration_indices
    test_indices = np.arange(bundle.test_data.shape[0], dtype=np.int64)
    method_config = resolve_method_config(config, task.method, asset_root)
    method_config["seed"] = task.seed
    if args.smoke:
        method_config = adapt_smoke_config(method_config)
        batch_size = int(method_config.get("batch_size", 32))
        fit_indices = fit_indices[: max(64, 2 * batch_size)]
        calibration_indices = calibration_indices[:32]
        test_indices = smoke_test_indices(bundle.test_labels)

    if task.method == "efficientad":
        asset_manifest = asset_root / "ASSET_MANIFEST.json"
        if not asset_manifest.is_file():
            raise FileNotFoundError(
                f"missing EfficientAD asset manifest: {asset_manifest}"
            )
        asset_manifest_value = json.loads(asset_manifest.read_text(encoding="utf-8"))
    else:
        asset_manifest_value = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    initialization_started = time.perf_counter()
    detector = build_detector(task.method, method_config, device)
    synchronize(device)
    initialization_seconds = time.perf_counter() - initialization_started

    fit_dataset = IADTensorDataset(
        bundle.train_data,
        fit_indices,
        profile=detector.preprocessing_profile,
        input_size=detector.input_size,
    )
    calibration_dataset = IADTensorDataset(
        bundle.train_data,
        calibration_indices,
        profile=detector.preprocessing_profile,
        input_size=detector.input_size,
    )
    test_dataset = IADTensorDataset(
        bundle.test_data,
        test_indices,
        profile=detector.preprocessing_profile,
        input_size=detector.input_size,
    )
    batch_size = int(method_config.get("batch_size", 32))
    fit_loader = make_loader(
        fit_dataset,
        batch_size=batch_size,
        workers=args.workers,
        shuffle=True,
        seed=task.seed,
    )
    calibration_loader = make_loader(
        calibration_dataset,
        batch_size=batch_size,
        workers=args.workers,
        shuffle=False,
        seed=task.seed,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=batch_size,
        workers=args.workers,
        shuffle=False,
        seed=task.seed,
    )

    synchronize(device)
    fit_started = time.perf_counter()
    fit_info = detector.fit(fit_loader, calibration_loader)
    synchronize(device)
    fit_seconds = time.perf_counter() - fit_started
    training_seconds = float(fit_info.get("training_seconds", fit_seconds))

    synchronize(device)
    calibration_started = time.perf_counter()
    calibration_scores = detector.score(calibration_loader)
    synchronize(device)
    calibration_scoring_seconds = time.perf_counter() - calibration_started

    synchronize(device)
    test_started = time.perf_counter()
    test_scores = detector.score(test_loader)
    synchronize(device)
    test_scoring_seconds = time.perf_counter() - test_started

    # Test labels are first consumed after model fitting and scores are fixed.
    test_labels = bundle.test_labels[test_indices]
    metrics = evaluate_detection(
        calibration_scores,
        test_labels,
        test_scores,
        threshold_quantile=float(config["evaluation"]["threshold_quantile"]),
        bootstrap_samples=(
            20 if args.smoke else int(config["evaluation"]["bootstrap_samples"])
        ),
        bootstrap_seed=task.seed,
    )
    if bool(config.get("save_scores", True)):
        atomic_scores(
            scores_path,
            calibration_indices=calibration_indices,
            calibration_scores=np.asarray(calibration_scores, dtype=np.float64),
            test_indices=test_indices,
            test_scores=np.asarray(test_scores, dtype=np.float64),
            test_labels=np.asarray(test_labels, dtype=np.int64),
            threshold=np.asarray([metrics["threshold"]], dtype=np.float64),
        )

    environment_git = git_state()
    end_to_end_seconds = time.perf_counter() - end_to_end_started
    result = {
        "identity": identity,
        "publication_eligible": not args.smoke,
        "method": task.method,
        "implementation_variant": detector.implementation_variant,
        "modulation": bundle.modulation,
        "seed": task.seed,
        "preprocessing_profile": detector.preprocessing_profile,
        "input_size": detector.input_size,
        "fit_samples": int(len(fit_dataset)),
        "calibration_samples": int(len(calibration_dataset)),
        "test_samples": int(len(test_dataset)),
        "normal_hash_overlap": len(
            set(split.fit_hashes).intersection(split.calibration_hashes)
        ),
        "config": method_config,
        "fit": fit_info,
        "metrics": metrics,
        "timing": {
            "initialization_seconds": initialization_seconds,
            "training_wall_seconds": training_seconds,
            "fit_pipeline_wall_seconds": fit_seconds,
            "calibration_scoring_wall_seconds": calibration_scoring_seconds,
            "test_scoring_wall_seconds": test_scoring_seconds,
            "milliseconds_per_test_image": 1000.0
            * test_scoring_seconds
            / len(test_dataset),
            "end_to_end_wall_seconds": end_to_end_seconds,
            "definition": (
                "CUDA-synchronized wall clock; training excludes post-training "
                "normal calibration, while fit_pipeline includes it when a method "
                "performs calibration inside detector.fit"
            ),
        },
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else None
        ),
        "scores_file": (
            scores_path.name if bool(config.get("save_scores", True)) else None
        ),
        "asset_manifest": asset_manifest_value,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "git_revision": environment_git["revision"],
            "git_dirty": environment_git["dirty"],
        },
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(run_path, result)
    print(
        f"AUROC={metrics['auroc']:.6f} AUPRC={metrics['auprc']:.6f} "
        f"training={training_seconds:.3f}s fit_pipeline={fit_seconds:.3f}s "
        f"wrote={run_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
