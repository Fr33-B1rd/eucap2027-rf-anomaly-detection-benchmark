#!/usr/bin/env python
"""Run one resumable cell of the official-architecture IAD CFLOW matrix."""

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
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataio.iad import discover_iad_sources, safe_load_iad_pickle  # noqa: E402
from iad_detection.cflow_matrix import (  # noqa: E402
    CFLOWTask,
    canonical_config_hash,
    expand_tasks,
    load_matrix_config,
)
from iad_detection.data import (  # noqa: E402
    IADTensorDataset,
    IndexedLabelDataset,
    build_normal_split,
)
from iad_detection.metrics import evaluate_detection  # noqa: E402
from iad_detection.methods.cflow_official import (  # noqa: E402
    OFFICIAL_COMMIT,
    OFFICIAL_REPOSITORY,
    OfficialCFLOW,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/iad_cflow_official_matrix.json",
    )
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--task-count", action="store_true")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "outputs/iad_cflow_official_matrix",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one meta/sub epoch on small subsets; always marked smoke-only",
    )
    return parser.parse_args()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def set_epoch_seed(seed: int, epoch: int) -> None:
    epoch_seed = seed + epoch * 100_003
    random.seed(epoch_seed)
    np.random.seed(epoch_seed % (2**32 - 1))
    torch.manual_seed(epoch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(epoch_seed)


def make_loader(
    dataset: IADTensorDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    drop_last: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        IndexedLabelDataset(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
    )


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_scores(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def print_tasks(tasks: list[CFLOWTask]) -> None:
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


def resolve_source(dataset_dir: Path, modulation: str):
    sources = {item.modulation: item for item in discover_iad_sources(dataset_dir)}
    if modulation not in sources:
        raise FileNotFoundError(
            f"matrix requests {modulation}, found only {sorted(sources)} in "
            f"{dataset_dir}"
        )
    return sources[modulation]


def main() -> None:
    args = parse_args()
    config = load_matrix_config(args.config)
    tasks = expand_tasks(config)
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
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    set_seed(task.seed)

    output_dir = args.output_root.expanduser() / task.relative_output_dir
    if args.smoke:
        output_dir = args.output_root.expanduser() / "smoke" / task.relative_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "run.json"
    checkpoint_path = output_dir / "checkpoint.pt"
    scores_path = output_dir / "scores.npz"

    source = resolve_source(dataset_dir, task.modulation)
    source_digest = file_sha256(source.path)
    identity = {
        "protocol_version": config["protocol_version"],
        "config_sha256": canonical_config_hash(config),
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
        raise FileExistsError(f"result already exists; pass --resume: {run_path}")

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
    training = dict(config["training"])
    if args.smoke:
        training["meta_epochs"] = 1
        training["sub_epochs"] = 1
        fit_indices = fit_indices[: max(task.train_batch_size, 64)]
        calibration_indices = calibration_indices[:32]
        test_indices = smoke_test_indices(bundle.test_labels)

    fit_dataset = IADTensorDataset(
        bundle.train_data,
        fit_indices,
        profile="imagenet_x10",
        input_size=task.input_size,
    )
    calibration_dataset = IADTensorDataset(
        bundle.train_data,
        calibration_indices,
        profile="imagenet_x10",
        input_size=task.input_size,
    )
    test_dataset = IADTensorDataset(
        bundle.test_data,
        test_indices,
        profile="imagenet_x10",
        input_size=task.input_size,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    initialization_started = time.perf_counter()
    model = OfficialCFLOW(training, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    initialization_seconds = time.perf_counter() - initialization_started
    start_epoch = 0
    train_history: list[dict[str, Any]] = []
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        start_epoch, train_history = model.load_checkpoint(
            checkpoint,
            expected_identity=identity,
        )
        print(f"[resume] continuing at meta epoch {start_epoch + 1}", flush=True)

    checkpoint_every = int(config["checkpoint"]["every_meta_epochs"])
    total_fit_seconds = sum(float(item["seconds"]) for item in train_history)
    for epoch in range(start_epoch, int(training["meta_epochs"])):
        set_epoch_seed(task.seed, epoch)
        fit_loader = make_loader(
            fit_dataset,
            batch_size=task.train_batch_size,
            workers=args.workers,
            shuffle=True,
            drop_last=True,
            seed=task.seed + epoch * 100_003,
        )
        stats = model.train_meta_epoch(epoch, fit_loader)
        row = {
            "meta_epoch": epoch + 1,
            "mean_loss": stats.mean_loss,
            "optimizer_steps": stats.optimizer_steps,
            "samples_seen": stats.samples_seen,
            "seconds": stats.seconds,
            "learning_rate": stats.learning_rate,
        }
        train_history.append(row)
        total_fit_seconds += stats.seconds
        print(json.dumps(row), flush=True)
        next_epoch = epoch + 1
        if next_epoch % checkpoint_every == 0 or next_epoch == int(
            training["meta_epochs"]
        ):
            atomic_checkpoint(
                checkpoint_path,
                model.checkpoint_state(
                    next_meta_epoch=next_epoch,
                    identity=identity,
                    train_history=train_history,
                ),
            )
            print(f"checkpoint: {checkpoint_path}", flush=True)

    calibration_loader = make_loader(
        calibration_dataset,
        batch_size=task.score_batch_size,
        workers=args.workers,
        shuffle=False,
        drop_last=False,
        seed=task.seed,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=task.score_batch_size,
        workers=args.workers,
        shuffle=False,
        drop_last=False,
        seed=task.seed,
    )
    calibration_scoring_started = time.perf_counter()
    calibration_maps, feature_grids = model.collect_log_probability_maps(
        calibration_loader
    )
    references = model.calibration_references(calibration_maps)
    calibration_scores = model.image_scores(
        calibration_maps, references, input_size=task.input_size
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    calibration_scoring_seconds = time.perf_counter() - calibration_scoring_started
    test_scoring_started = time.perf_counter()
    test_maps, test_feature_grids = model.collect_log_probability_maps(test_loader)
    if test_feature_grids != feature_grids:
        raise RuntimeError(
            f"calibration/test feature grids differ: {feature_grids}, "
            f"{test_feature_grids}"
        )
    test_scores = model.image_scores(
        test_maps, references, input_size=task.input_size
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_scoring_seconds = time.perf_counter() - test_scoring_started
    scoring_seconds = calibration_scoring_seconds + test_scoring_seconds

    # Test labels are exposed only after training and anomaly scores are fixed.
    test_labels = bundle.test_labels[test_indices]
    metrics = evaluate_detection(
        calibration_scores,
        test_labels,
        test_scores,
        threshold_quantile=float(config["evaluation"]["threshold_quantile"]),
        bootstrap_samples=(
            20
            if args.smoke
            else int(config["evaluation"]["bootstrap_samples"])
        ),
        bootstrap_seed=task.seed,
    )
    if bool(config.get("save_scores", True)):
        atomic_scores(
            scores_path,
            calibration_indices=calibration_indices,
            calibration_scores=calibration_scores,
            test_indices=test_indices,
            test_scores=test_scores,
            test_labels=test_labels,
        )

    result = {
        "identity": identity,
        "publication_eligible": not args.smoke,
        "method": "cflow",
        "implementation_variant": (
            "official_cflow_wrn50_2_l2_l3_l4_"
            "iad_calibration_reference_scoring"
        ),
        "official_reference": {
            "repository": OFFICIAL_REPOSITORY,
            "commit": OFFICIAL_COMMIT,
        },
        "iad_adaptations": [
            "IAD normal-only fit/calibration split",
            "final-only test evaluation; no test-based checkpoint selection",
            "normal calibration maxima replace test-set likelihood maxima",
            "64x64 uses batch 64 so layer4 supplies one 256-patch fiber",
        ],
        "task": task.to_dict(),
        "preprocessing_profile": "imagenet_x10",
        "fit_samples": int(len(fit_dataset)),
        "calibration_samples": int(len(calibration_dataset)),
        "test_samples": int(len(test_dataset)),
        "normal_hash_overlap": len(
            set(split.fit_hashes).intersection(split.calibration_hashes)
        ),
        "training_config": training,
        "evaluation_config": config["evaluation"],
        "feature_dims": list(model.feature_dims),
        "feature_grids": feature_grids,
        "calibration_log_probability_maxima": references,
        "train_history": train_history,
        "metrics": metrics,
        "initialization_seconds": initialization_seconds,
        "fit_seconds": total_fit_seconds,
        "scoring_seconds": scoring_seconds,
        "calibration_scoring_seconds": calibration_scoring_seconds,
        "test_scoring_seconds": test_scoring_seconds,
        "milliseconds_per_test_image": 1000.0
        * test_scoring_seconds
        / len(test_dataset),
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else None
        ),
        "scores_file": (
            scores_path.name if bool(config.get("save_scores", True)) else None
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "freia": __import__("FrEIA").__version__
            if hasattr(__import__("FrEIA"), "__version__")
            else "unknown",
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "git_revision": git_revision(),
        },
    }
    atomic_json(run_path, result)
    print(
        f"AUROC={metrics['auroc']:.6f} AUPRC={metrics['auprc']:.6f} "
        f"wrote={run_path}",
        flush=True,
    )
    if checkpoint_path.exists() and not bool(
        config["checkpoint"].get("keep_after_success", False)
    ):
        checkpoint_path.unlink()
        print(
            "removed completed-run checkpoint to conserve project storage",
            flush=True,
        )


if __name__ == "__main__":
    main()
