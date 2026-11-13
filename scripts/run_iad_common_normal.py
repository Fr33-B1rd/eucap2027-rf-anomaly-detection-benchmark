#!/usr/bin/env python
"""Run one resumable method-by-seed cell of the common-normal IAD protocol."""

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
from iad_detection.common_normal_protocol import (  # noqa: E402
    CommonNormalTask,
    canonical_config_sha256,
    expand_common_normal_tasks,
    load_common_normal_config,
    seeded_index_split,
)
from iad_detection.data import IADTensorDataset, IndexedLabelDataset  # noqa: E402
from iad_detection.metrics import evaluate_detection  # noqa: E402
from iad_detection.methods import build_detector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/iad_common_normal.json",
    )
    parser.add_argument("--task-kind", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--task-count", action="store_true")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--print-task-method", action="store_true")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "outputs/iad_common_normal",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def set_epoch_seed(seed: int, epoch: int) -> None:
    value = seed + epoch * 100_003
    random.seed(value)
    np.random.seed(value % (2**32 - 1))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


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
    drop_last: bool = False,
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


def make_dataset(
    data: torch.Tensor,
    indices: np.ndarray,
    *,
    profile: str,
    input_size: int,
) -> IADTensorDataset:
    return IADTensorDataset(
        data,
        indices,
        profile=profile,
        input_size=input_size,
    )


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_scores(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def atomic_checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_snapshot_sha256() -> str:
    roots = (
        REPOSITORY_ROOT / "core",
        REPOSITORY_ROOT / "dataio" / "iad",
        REPOSITORY_ROOT / "iad_detection",
    )
    paths = [REPOSITORY_ROOT / "scripts/run_iad_common_normal.py"]
    for root in roots:
        paths.extend(root.rglob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.as_posix()):
        relative = path.relative_to(REPOSITORY_ROOT).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


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


def print_tasks(tasks: list[CommonNormalTask]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(tasks[0].to_dict()), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(task.to_dict() for task in tasks)
    print(buffer.getvalue(), end="")


def adapt_smoke_config(method: str, config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    if "epochs" in result:
        result["epochs"] = 1
    if "max_steps" in result:
        result["max_steps"] = 2
    if method == "cflow":
        result["meta_epochs"] = 1
        result["sub_epochs"] = 1
        result["checkpoint_every_meta_epochs"] = 1
    for key, value in (
        ("candidate_pool_size", 256),
        ("max_memory_bank_size", 64),
        ("sample_patches_per_image", 4),
    ):
        if key in result:
            result[key] = value
    return result


def resolve_method_config(
    config: dict[str, Any], method: str, asset_root: Path
) -> dict[str, Any]:
    result = dict(config["methods"][method])
    if method == "efficientad":
        result["teacher_weights"] = str(
            asset_root / result.pop("teacher_weights_relative")
        )
        result["imagenette_dir"] = str(
            asset_root / result.pop("imagenette_dir_relative")
        )
    return result


def validate_and_load_sources(
    dataset_dir: Path, config: dict[str, Any], *, smoke: bool
) -> tuple[Any, dict[str, Any], dict[str, str]]:
    discovered = {item.modulation: item for item in discover_iad_sources(dataset_dir)}
    protocol = config["protocol"]
    required = {
        str(protocol["normal_reference_source"]).upper(),
        str(protocol["normal_test_source"]).upper(),
        *(str(value).upper() for value in protocol["anomaly_conditions"]),
    }
    missing = required.difference(discovered)
    if missing:
        raise FileNotFoundError(
            f"missing IAD sources {sorted(missing)}; found {sorted(discovered)}"
        )
    bundles = {
        name: safe_load_iad_pickle(discovered[name].path) for name in sorted(required)
    }
    reference_name = str(protocol["normal_reference_source"]).upper()
    reference = bundles[reference_name]
    expected = protocol["expected_counts"]
    if not smoke and int(reference.train_data.shape[0]) != int(
        expected["normal_reference"]
    ):
        raise RuntimeError(
            f"expected {expected['normal_reference']} CHIRP reference normals, "
            f"found {reference.train_data.shape[0]}"
        )
    normal_name = str(protocol["normal_test_source"]).upper()
    normal_indices = np.flatnonzero(bundles[normal_name].test_labels == 0)
    if not smoke and normal_indices.size != int(expected["normal_test"]):
        raise RuntimeError(
            f"expected {expected['normal_test']} CHIRP test normals, "
            f"found {normal_indices.size}"
        )
    for condition in protocol["anomaly_conditions"]:
        name = str(condition).upper()
        anomaly_count = int(np.sum(bundles[name].test_labels == 1))
        if not smoke and anomaly_count != int(expected["anomaly_per_condition"]):
            raise RuntimeError(
                f"expected {expected['anomaly_per_condition']} {name} anomalies, "
                f"found {anomaly_count}"
            )
    digests = {
        name: file_sha256(discovered[name].path) for name in sorted(required)
    }
    return reference, bundles, digests


def score_generic(
    detector: Any,
    data: torch.Tensor,
    indices: np.ndarray,
    *,
    batch_size: int,
    workers: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    dataset = make_dataset(
        data,
        indices,
        profile=detector.preprocessing_profile,
        input_size=detector.input_size,
    )
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        workers=workers,
        shuffle=False,
        seed=seed,
    )
    synchronize(device)
    started = time.perf_counter()
    values = np.asarray(detector.score(loader), dtype=np.float64)
    synchronize(device)
    return values, time.perf_counter() - started


def run_generic_detector(
    task: CommonNormalTask,
    *,
    method_config: dict[str, Any],
    reference: Any,
    bundles: dict[str, Any],
    config: dict[str, Any],
    fit_indices: np.ndarray,
    calibration_indices: np.ndarray,
    common_normal_indices: np.ndarray,
    anomaly_indices: dict[str, np.ndarray],
    device: torch.device,
    workers: int,
) -> dict[str, Any]:
    synchronize(device)
    initialized = time.perf_counter()
    detector = build_detector(task.method, method_config, device)
    synchronize(device)
    initialization_seconds = time.perf_counter() - initialized
    batch_size = int(method_config.get("batch_size", 32))
    fit_dataset = make_dataset(
        reference.train_data,
        fit_indices,
        profile=detector.preprocessing_profile,
        input_size=detector.input_size,
    )
    calibration_dataset = make_dataset(
        reference.train_data,
        calibration_indices,
        profile=detector.preprocessing_profile,
        input_size=detector.input_size,
    )
    fit_loader = make_loader(
        fit_dataset,
        batch_size=batch_size,
        workers=workers,
        shuffle=True,
        seed=task.seed,
    )
    calibration_loader = make_loader(
        calibration_dataset,
        batch_size=batch_size,
        workers=workers,
        shuffle=False,
        seed=task.seed,
    )
    synchronize(device)
    fitted = time.perf_counter()
    fit_info = detector.fit(fit_loader, calibration_loader)
    synchronize(device)
    fit_seconds = time.perf_counter() - fitted

    calibration_scores, calibration_seconds = score_generic(
        detector,
        reference.train_data,
        calibration_indices,
        batch_size=batch_size,
        workers=workers,
        seed=task.seed,
        device=device,
    )
    normal_name = str(config["protocol"]["normal_test_source"]).upper()
    common_normal_scores, common_normal_seconds = score_generic(
        detector,
        bundles[normal_name].test_data,
        common_normal_indices,
        batch_size=batch_size,
        workers=workers,
        seed=task.seed,
        device=device,
    )
    anomaly_scores: dict[str, np.ndarray] = {}
    anomaly_seconds: dict[str, float] = {}
    for condition in config["protocol"]["anomaly_conditions"]:
        name = str(condition).upper()
        anomaly_scores[name], anomaly_seconds[name] = score_generic(
            detector,
            bundles[name].test_data,
            anomaly_indices[name],
            batch_size=batch_size,
            workers=workers,
            seed=task.seed,
            device=device,
        )
    return {
        "implementation_variant": detector.implementation_variant,
        "preprocessing_profile": detector.preprocessing_profile,
        "input_size": detector.input_size,
        "fit_info": fit_info,
        "calibration_scores": calibration_scores,
        "common_normal_scores": common_normal_scores,
        "anomaly_scores": anomaly_scores,
        "timing": {
            "initialization_seconds": initialization_seconds,
            "training_wall_seconds": float(
                fit_info.get("training_seconds", fit_seconds)
            ),
            "fit_pipeline_wall_seconds": fit_seconds,
            "calibration_scoring_wall_seconds": calibration_seconds,
            "common_normal_scoring_wall_seconds": common_normal_seconds,
            "anomaly_scoring_wall_seconds": anomaly_seconds,
        },
    }


def run_cflow(
    task: CommonNormalTask,
    *,
    method_config: dict[str, Any],
    reference: Any,
    bundles: dict[str, Any],
    config: dict[str, Any],
    fit_indices: np.ndarray,
    calibration_indices: np.ndarray,
    common_normal_indices: np.ndarray,
    anomaly_indices: dict[str, np.ndarray],
    device: torch.device,
    workers: int,
    checkpoint_path: Path,
    identity: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    from iad_detection.methods.cflow_official import (
        OFFICIAL_COMMIT,
        OFFICIAL_REPOSITORY,
        OfficialCFLOW,
    )

    input_size = int(method_config["input_size"])
    fit_dataset = make_dataset(
        reference.train_data,
        fit_indices,
        profile="imagenet_x10",
        input_size=input_size,
    )
    calibration_dataset = make_dataset(
        reference.train_data,
        calibration_indices,
        profile="imagenet_x10",
        input_size=input_size,
    )
    synchronize(device)
    initialized = time.perf_counter()
    model = OfficialCFLOW(method_config, device)
    synchronize(device)
    initialization_seconds = time.perf_counter() - initialized
    start_epoch = 0
    train_history: list[dict[str, Any]] = []
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        start_epoch, train_history = model.load_checkpoint(
            checkpoint, expected_identity=identity
        )
        print(f"[resume] CFLOW continues at meta epoch {start_epoch + 1}", flush=True)

    checkpoint_every = int(method_config["checkpoint_every_meta_epochs"])
    for epoch in range(start_epoch, int(method_config["meta_epochs"])):
        set_epoch_seed(task.seed, epoch)
        fit_loader = make_loader(
            fit_dataset,
            batch_size=int(method_config["train_batch_size"]),
            workers=workers,
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
        print(json.dumps(row), flush=True)
        next_epoch = epoch + 1
        if next_epoch % checkpoint_every == 0 or next_epoch == int(
            method_config["meta_epochs"]
        ):
            atomic_checkpoint(
                checkpoint_path,
                model.checkpoint_state(
                    next_meta_epoch=next_epoch,
                    identity=identity,
                    train_history=train_history,
                ),
            )

    score_batch_size = int(method_config["score_batch_size"])

    def collect(data: torch.Tensor, indices: np.ndarray) -> tuple[list[np.ndarray], list[Any], float]:
        dataset = make_dataset(
            data, indices, profile="imagenet_x10", input_size=input_size
        )
        loader = make_loader(
            dataset,
            batch_size=score_batch_size,
            workers=workers,
            shuffle=False,
            seed=task.seed,
        )
        synchronize(device)
        started = time.perf_counter()
        maps, grids = model.collect_log_probability_maps(loader)
        synchronize(device)
        return maps, grids, time.perf_counter() - started

    calibration_maps, feature_grids, calibration_seconds = collect(
        reference.train_data, calibration_indices
    )
    references = model.calibration_references(calibration_maps)
    calibration_scores = model.image_scores(
        calibration_maps, references, input_size=input_size
    )
    normal_name = str(config["protocol"]["normal_test_source"]).upper()
    normal_maps, normal_grids, common_normal_seconds = collect(
        bundles[normal_name].test_data, common_normal_indices
    )
    if normal_grids != feature_grids:
        raise RuntimeError("CFLOW calibration and common-normal feature grids differ")
    common_normal_scores = model.image_scores(
        normal_maps, references, input_size=input_size
    )
    anomaly_scores: dict[str, np.ndarray] = {}
    anomaly_seconds: dict[str, float] = {}
    for condition in config["protocol"]["anomaly_conditions"]:
        name = str(condition).upper()
        maps, grids, seconds = collect(
            bundles[name].test_data, anomaly_indices[name]
        )
        if grids != feature_grids:
            raise RuntimeError(f"CFLOW feature grids differ for {name}")
        anomaly_scores[name] = model.image_scores(
            maps, references, input_size=input_size
        )
        anomaly_seconds[name] = seconds

    fit_seconds = float(sum(row["seconds"] for row in train_history))
    return {
        "implementation_variant": (
            "official_cflow_wrn50_2_l2_l3_l4_"
            "iad_calibration_reference_scoring"
        ),
        "preprocessing_profile": "imagenet_x10",
        "input_size": input_size,
        "fit_info": {
            "official_reference": {
                "repository": OFFICIAL_REPOSITORY,
                "commit": OFFICIAL_COMMIT,
            },
            "feature_dims": list(model.feature_dims),
            "feature_grids": feature_grids,
            "calibration_log_probability_maxima": references,
            "train_history": train_history,
        },
        "calibration_scores": np.asarray(calibration_scores, dtype=np.float64),
        "common_normal_scores": np.asarray(
            common_normal_scores, dtype=np.float64
        ),
        "anomaly_scores": {
            name: np.asarray(values, dtype=np.float64)
            for name, values in anomaly_scores.items()
        },
        "timing": {
            "initialization_seconds": initialization_seconds,
            "training_wall_seconds": fit_seconds,
            "fit_pipeline_wall_seconds": fit_seconds,
            "calibration_scoring_wall_seconds": calibration_seconds,
            "common_normal_scoring_wall_seconds": common_normal_seconds,
            "anomaly_scoring_wall_seconds": anomaly_seconds,
        },
    }


def flatten_native_x10(data: torch.Tensor, indices: np.ndarray) -> np.ndarray:
    values = data[indices].detach().cpu().numpy().astype(np.float32, copy=False)
    return np.ascontiguousarray(values.reshape(values.shape[0], -1) * 10.0)


def run_pca(
    task: CommonNormalTask,
    *,
    method_config: dict[str, Any],
    reference: Any,
    bundles: dict[str, Any],
    config: dict[str, Any],
    fit_indices: np.ndarray,
    calibration_indices: np.ndarray,
    common_normal_indices: np.ndarray,
    anomaly_indices: dict[str, np.ndarray],
) -> dict[str, Any]:
    from sklearn.decomposition import PCA

    from scripts.run_iad_pca_baseline import reconstruction_mse

    fitted_values = flatten_native_x10(reference.train_data, fit_indices)
    model = PCA(
        n_components=int(method_config["n_components"]),
        whiten=False,
        svd_solver=str(method_config["svd_solver"]),
        iterated_power=int(method_config["iterated_power"]),
        random_state=task.seed,
    )
    started = time.perf_counter()
    model.fit(fitted_values)
    fit_seconds = time.perf_counter() - started
    batch_size = int(method_config["score_batch_size"])

    def score(data: torch.Tensor, indices: np.ndarray) -> tuple[np.ndarray, float]:
        values = flatten_native_x10(data, indices)
        began = time.perf_counter()
        scores = reconstruction_mse(model, values, batch_size=batch_size)
        return scores, time.perf_counter() - began

    calibration_scores, calibration_seconds = score(
        reference.train_data, calibration_indices
    )
    normal_name = str(config["protocol"]["normal_test_source"]).upper()
    common_normal_scores, common_normal_seconds = score(
        bundles[normal_name].test_data, common_normal_indices
    )
    anomaly_scores: dict[str, np.ndarray] = {}
    anomaly_seconds: dict[str, float] = {}
    for condition in config["protocol"]["anomaly_conditions"]:
        name = str(condition).upper()
        anomaly_scores[name], anomaly_seconds[name] = score(
            bundles[name].test_data, anomaly_indices[name]
        )
    return {
        "implementation_variant": "sklearn_pca64_randomized_native64_mse",
        "preprocessing_profile": method_config["preprocessing_profile"],
        "input_size": int(method_config["input_size"]),
        "fit_info": {
            "n_components": int(model.n_components_),
            "explained_variance_ratio_sum": float(
                np.sum(model.explained_variance_ratio_)
            ),
        },
        "calibration_scores": calibration_scores,
        "common_normal_scores": common_normal_scores,
        "anomaly_scores": anomaly_scores,
        "timing": {
            "initialization_seconds": 0.0,
            "training_wall_seconds": fit_seconds,
            "fit_pipeline_wall_seconds": fit_seconds,
            "calibration_scoring_wall_seconds": calibration_seconds,
            "common_normal_scoring_wall_seconds": common_normal_seconds,
            "anomaly_scoring_wall_seconds": anomaly_seconds,
        },
    }


def build_metrics_and_score_arrays(
    *,
    result: dict[str, Any],
    config: dict[str, Any],
    task: CommonNormalTask,
    fit_indices: np.ndarray,
    calibration_indices: np.ndarray,
    common_normal_indices: np.ndarray,
    anomaly_indices: dict[str, np.ndarray],
    smoke: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    calibration_scores = np.asarray(result["calibration_scores"], dtype=np.float64)
    normal_scores = np.asarray(result["common_normal_scores"], dtype=np.float64)
    arrays: dict[str, np.ndarray] = {
        "fit_indices": fit_indices,
        "calibration_indices": calibration_indices,
        "calibration_scores": calibration_scores,
        "common_normal_indices": common_normal_indices,
        "common_normal_scores": normal_scores,
    }
    metrics: dict[str, Any] = {}
    thresholds: set[float] = set()
    for offset, condition in enumerate(config["protocol"]["anomaly_conditions"]):
        name = str(condition).upper()
        anomaly_scores = np.asarray(result["anomaly_scores"][name], dtype=np.float64)
        labels = np.concatenate(
            (
                np.zeros(normal_scores.size, dtype=np.int64),
                np.ones(anomaly_scores.size, dtype=np.int64),
            )
        )
        scores = np.concatenate((normal_scores, anomaly_scores))
        metrics[name] = evaluate_detection(
            calibration_scores,
            labels,
            scores,
            threshold_quantile=float(config["evaluation"]["threshold_quantile"]),
            bootstrap_samples=(
                20 if smoke else int(config["evaluation"]["bootstrap_samples"])
            ),
            bootstrap_seed=task.seed + offset * 10_007,
        )
        thresholds.add(float(metrics[name]["threshold"]))
        key = name.lower()
        arrays[f"anomaly_indices_{key}"] = anomaly_indices[name]
        arrays[f"anomaly_scores_{key}"] = anomaly_scores
    if len(thresholds) != 1:
        raise RuntimeError("all conditions must share one calibration threshold")
    arrays["threshold"] = np.asarray([next(iter(thresholds))], dtype=np.float64)
    return metrics, arrays


def main() -> None:
    args = parse_args()
    config = load_common_normal_config(args.config)
    tasks = expand_common_normal_tasks(config, args.task_kind)
    if args.task_count:
        print(len(tasks))
        return
    if args.list_tasks:
        print_tasks(tasks)
        return
    if args.task_id is None or not 0 <= args.task_id < len(tasks):
        raise SystemExit(f"--task-id must be in [0, {len(tasks) - 1}]")
    task = tasks[args.task_id]
    if args.print_task_method:
        print(task.method)
        return
    if args.workers < 0:
        raise SystemExit("--workers cannot be negative")
    if args.task_kind == "gpu" and args.device == "cpu":
        raise SystemExit("GPU tasks require a CUDA device")
    if args.task_kind == "cpu" and task.method != "pca_reconstruction":
        raise SystemExit("only PCA is supported by the CPU task matrix")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    randomization = config.get("randomization", {})

    def resolve_seed(name: str) -> int:
        value = randomization.get(name, "task_seed")
        return task.seed if value == "task_seed" else int(value)

    split_seed = resolve_seed("split_seed")
    training_seed = resolve_seed("training_seed")
    execution_task = CommonNormalTask(
        task_id=task.task_id,
        task_kind=task.task_kind,
        method=task.method,
        seed=training_seed,
    )
    set_seed(training_seed)
    dataset_dir = (args.dataset_dir or Path(config["dataset_dir"])).expanduser()
    asset_root = (
        args.asset_root or REPOSITORY_ROOT / ".cache/iad_assets"
    ).expanduser()
    output_dir = args.output_root.expanduser() / task.relative_output_dir
    if args.smoke:
        output_dir = args.output_root.expanduser() / "smoke" / task.relative_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "run.json"
    scores_path = output_dir / "scores.npz"
    checkpoint_path = output_dir / "checkpoint.pt"

    reference, bundles, source_digests = validate_and_load_sources(
        dataset_dir, config, smoke=args.smoke
    )
    identity = {
        "protocol_version": config["protocol_version"],
        "config_sha256": canonical_config_sha256(config),
        "code_snapshot_sha256": code_snapshot_sha256(),
        "task": task.to_dict(),
        "randomization": {
            "split_seed": split_seed,
            "training_seed": training_seed,
        },
        "source_sha256": source_digests,
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

    split = seeded_index_split(
        int(reference.train_data.shape[0]),
        calibration_fraction=float(config["evaluation"]["calibration_fraction"]),
        seed=split_seed,
    )
    fit_indices = split.fit_indices
    calibration_indices = split.calibration_indices
    normal_name = str(config["protocol"]["normal_test_source"]).upper()
    common_normal_indices = np.flatnonzero(bundles[normal_name].test_labels == 0).astype(
        np.int64
    )
    anomaly_indices = {
        str(condition).upper(): np.flatnonzero(
            bundles[str(condition).upper()].test_labels == 1
        ).astype(np.int64)
        for condition in config["protocol"]["anomaly_conditions"]
    }
    method_config = resolve_method_config(config, task.method, asset_root)
    method_config["seed"] = training_seed
    if args.smoke:
        method_config = adapt_smoke_config(task.method, method_config)
        fit_indices = fit_indices[: max(64, 2 * int(method_config.get("batch_size", 32)))]
        calibration_indices = calibration_indices[:32]
        common_normal_indices = common_normal_indices[:16]
        anomaly_indices = {
            name: indices[:16] for name, indices in anomaly_indices.items()
        }

    asset_manifest_value = None
    if task.method == "efficientad":
        manifest_path = asset_root / "ASSET_MANIFEST.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing EfficientAD asset manifest: {manifest_path}")
        asset_manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))

    print(json.dumps({"task": task.to_dict(), "output_dir": str(output_dir)}), flush=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    end_to_end_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if task.method == "cflow":
        method_result = run_cflow(
            execution_task,
            method_config=method_config,
            reference=reference,
            bundles=bundles,
            config=config,
            fit_indices=fit_indices,
            calibration_indices=calibration_indices,
            common_normal_indices=common_normal_indices,
            anomaly_indices=anomaly_indices,
            device=device,
            workers=args.workers,
            checkpoint_path=checkpoint_path,
            identity=identity,
            resume=args.resume,
        )
    elif task.method == "pca_reconstruction":
        method_result = run_pca(
            execution_task,
            method_config=method_config,
            reference=reference,
            bundles=bundles,
            config=config,
            fit_indices=fit_indices,
            calibration_indices=calibration_indices,
            common_normal_indices=common_normal_indices,
            anomaly_indices=anomaly_indices,
        )
    else:
        method_result = run_generic_detector(
            execution_task,
            method_config=method_config,
            reference=reference,
            bundles=bundles,
            config=config,
            fit_indices=fit_indices,
            calibration_indices=calibration_indices,
            common_normal_indices=common_normal_indices,
            anomaly_indices=anomaly_indices,
            device=device,
            workers=args.workers,
        )

    metrics, score_arrays = build_metrics_and_score_arrays(
        result=method_result,
        config=config,
        task=task,
        fit_indices=fit_indices,
        calibration_indices=calibration_indices,
        common_normal_indices=common_normal_indices,
        anomaly_indices=anomaly_indices,
        smoke=args.smoke,
    )
    if bool(config.get("save_scores", True)):
        atomic_scores(scores_path, score_arrays)
    state = git_state()
    end_to_end_seconds = time.perf_counter() - end_to_end_started
    unique_test_samples = int(
        common_normal_indices.size + sum(v.size for v in anomaly_indices.values())
    )
    timing = dict(method_result["timing"])
    timing["unique_test_samples"] = unique_test_samples
    timing["end_to_end_wall_seconds"] = end_to_end_seconds
    timing["definition"] = (
        "One model is fitted on CHIRP reference normals, calibration and CHIRP "
        "test normals are each scored once, and the same frozen model scores all "
        "four anomaly conditions."
    )
    result = {
        "identity": identity,
        "publication_eligible": not args.smoke,
        "method": task.method,
        "seed": task.seed,
        "randomization": {
            "split_seed": split_seed,
            "training_seed": training_seed,
        },
        "protocol": config["protocol"],
        "split": {
            "strategy": "seeded_index_permutation",
            "calibration_fraction": float(
                config["evaluation"]["calibration_fraction"]
            ),
            "fit_samples": int(fit_indices.size),
            "calibration_samples": int(calibration_indices.size),
            "index_overlap": int(
                np.intersect1d(fit_indices, calibration_indices).size
            ),
        },
        "common_test_normal_samples": int(common_normal_indices.size),
        "anomaly_samples": {
            name: int(indices.size) for name, indices in anomaly_indices.items()
        },
        "implementation_variant": method_result["implementation_variant"],
        "preprocessing_profile": method_result["preprocessing_profile"],
        "input_size": int(method_result["input_size"]),
        "config": method_config,
        "fit": method_result["fit_info"],
        "metrics_by_anomaly_condition": metrics,
        "timing": timing,
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else None
        ),
        "scores_file": scores_path.name if bool(config.get("save_scores", True)) else None,
        "score_storage": (
            "Calibration and common-normal scores are stored once; each anomaly "
            "condition has its own indices and scores."
        ),
        "asset_manifest": asset_manifest_value,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "git_revision": state["revision"],
            "git_dirty": state["dirty"],
        },
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(run_path, result)
    macro_auroc = float(
        np.mean([value["auroc"] for value in metrics.values()])
    )
    print(
        f"method={task.method} seed={task.seed} macro_AUROC={macro_auroc:.6f} "
        f"fit={timing['training_wall_seconds']:.3f}s wrote={run_path}",
        flush=True,
    )
    if (
        task.method == "cflow"
        and checkpoint_path.exists()
        and not bool(method_config.get("keep_checkpoint_after_success", False))
    ):
        checkpoint_path.unlink()


if __name__ == "__main__":
    main()
