#!/usr/bin/env python
"""Run a normal-only RF spectral-energy baseline on native IAD spectrograms.

The detector averages every spectrogram over time, robustly standardizes the
resulting frequency-energy profile with fit-normal median/MAD statistics, and
uses the mean squared robust z-score as the image-level anomaly score.  No
anomaly label is consumed until fitting and scoring are complete.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataio.iad import discover_iad_sources, safe_load_iad_pickle  # noqa: E402
from iad_detection.data import build_normal_split  # noqa: E402
from iad_detection.final_matrix import FinalTask, expand_final_tasks  # noqa: E402
from iad_detection.metrics import evaluate_detection  # noqa: E402


METHOD = "rf_spectral_energy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/iad_rf_energy.json",
    )
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--task-count", action="store_true")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "outputs/iad_rf_energy",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"protocol_version", "dataset_dir", "matrix", "evaluation", "rf_energy"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"RF-energy config is missing {sorted(missing)}")
    if config["matrix"].get("methods") != [METHOD]:
        raise ValueError(f"matrix.methods must be ['{METHOD}']")
    rf = config["rf_energy"]
    expected = {
        "input_size": 64,
        "preprocessing_profile": "iad_native_x10",
        "profile_reduction": "mean_over_time",
        "center_estimator": "median",
        "scale_estimator": "mad",
        "score": "mean_squared_robust_z",
    }
    for key, value in expected.items():
        if rf.get(key) != value:
            raise ValueError(f"rf_energy.{key} must equal {value!r}")
    if float(rf["mad_consistency_constant"]) <= 0:
        raise ValueError("mad_consistency_constant must be positive")
    if float(rf["scale_floor"]) <= 0:
        raise ValueError("scale_floor must be positive")
    return config


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_snapshot_sha256() -> str:
    paths = [
        REPOSITORY_ROOT / "scripts/run_iad_rf_energy_baseline.py",
        REPOSITORY_ROOT / "iad_detection/data.py",
        REPOSITORY_ROOT / "iad_detection/metrics.py",
        REPOSITORY_ROOT / "dataio/iad/__init__.py",
    ]
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


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def frequency_profiles(data: Any, indices: np.ndarray) -> np.ndarray:
    """Return time-averaged frequency profiles in the native x10 scale."""

    values = data[indices].detach().cpu().numpy().astype(np.float64, copy=False)
    if values.ndim != 4 or tuple(values.shape[1:]) != (1, 64, 64):
        raise ValueError(f"expected [N,1,64,64], got {values.shape}")
    profiles = values[:, 0].mean(axis=-1) * 10.0
    if not np.isfinite(profiles).all():
        raise RuntimeError("non-finite IAD frequency profiles")
    return np.ascontiguousarray(profiles)


def fit_robust_profile(
    values: np.ndarray, *, consistency_constant: float, scale_floor: float
) -> tuple[np.ndarray, np.ndarray, int]:
    center = np.median(values, axis=0)
    mad = np.median(np.abs(values - center[None, :]), axis=0)
    raw_scale = consistency_constant * mad
    floored_bins = int(np.count_nonzero(raw_scale < scale_floor))
    scale = np.maximum(raw_scale, scale_floor)
    return center, scale, floored_bins


def robust_energy_score(
    values: np.ndarray, *, center: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    z = (values - center[None, :]) / scale[None, :]
    scores = np.mean(np.square(z, dtype=np.float64), axis=1)
    if not np.isfinite(scores).all():
        raise RuntimeError("RF-energy detector produced non-finite scores")
    return scores


def task_output_dir(root: Path, task: FinalTask) -> Path:
    return root / METHOD / task.modulation / f"seed{task.seed}"


def run_task(
    task: FinalTask,
    *,
    config: dict[str, Any],
    dataset_dir: Path,
    output_root: Path,
    resume: bool,
) -> Path:
    sources = {item.modulation: item for item in discover_iad_sources(dataset_dir)}
    if task.modulation not in sources:
        raise FileNotFoundError(
            f"missing {task.modulation}; found {sorted(sources)} in {dataset_dir}"
        )
    source = sources[task.modulation]
    output_dir = task_output_dir(output_root, task)
    run_path = output_dir / "run.json"
    scores_path = output_dir / "scores.npz"
    identity = {
        "protocol_version": config["protocol_version"],
        "config_sha256": canonical_hash(config),
        "code_snapshot_sha256": code_snapshot_sha256(),
        "task": task.to_dict(),
        "source_sha256": file_sha256(source.path),
    }
    if run_path.exists():
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        if previous.get("identity") != identity:
            raise RuntimeError(f"existing result identity mismatch: {run_path}")
        if resume:
            print(f"[resume] task={task.task_id} {task.modulation} seed={task.seed}")
            return run_path
        raise FileExistsError(f"result exists; pass --resume: {run_path}")

    started_utc = datetime.now(timezone.utc).isoformat()
    end_to_end_started = time.perf_counter()
    bundle = safe_load_iad_pickle(source.path)
    split = build_normal_split(
        bundle.train_data,
        calibration_fraction=float(config["evaluation"]["calibration_fraction"]),
        seed=task.seed,
    )
    test_indices = np.arange(bundle.test_data.shape[0], dtype=np.int64)
    fit_values = frequency_profiles(bundle.train_data, split.fit_indices)
    calibration_values = frequency_profiles(
        bundle.train_data, split.calibration_indices
    )
    test_values = frequency_profiles(bundle.test_data, test_indices)
    rf = config["rf_energy"]

    fit_started = time.perf_counter()
    center, scale, floored_bins = fit_robust_profile(
        fit_values,
        consistency_constant=float(rf["mad_consistency_constant"]),
        scale_floor=float(rf["scale_floor"]),
    )
    fit_seconds = time.perf_counter() - fit_started
    calibration_started = time.perf_counter()
    calibration_scores = robust_energy_score(
        calibration_values, center=center, scale=scale
    )
    calibration_seconds = time.perf_counter() - calibration_started
    test_started = time.perf_counter()
    test_scores = robust_energy_score(test_values, center=center, scale=scale)
    test_seconds = time.perf_counter() - test_started

    # Labels are first consumed after fitting and all anomaly scores are fixed.
    test_labels = np.asarray(bundle.test_labels[test_indices], dtype=np.int64)
    metrics = evaluate_detection(
        calibration_scores,
        test_labels,
        test_scores,
        threshold_quantile=float(config["evaluation"]["threshold_quantile"]),
        bootstrap_samples=int(config["evaluation"]["bootstrap_samples"]),
        bootstrap_seed=task.seed,
    )
    if bool(config.get("save_scores", True)):
        atomic_npz(
            scores_path,
            fit_indices=np.asarray(split.fit_indices, dtype=np.int64),
            calibration_indices=np.asarray(split.calibration_indices, dtype=np.int64),
            calibration_scores=calibration_scores,
            test_indices=test_indices,
            test_scores=test_scores,
            test_labels=test_labels,
            threshold=np.asarray([metrics["threshold"]], dtype=np.float64),
            fitted_center=center,
            fitted_scale=scale,
        )
    state = git_state()
    result = {
        "identity": identity,
        "publication_eligible": True,
        "method": METHOD,
        "implementation_variant": "robust_frequency_profile_mad_native64_mean_z2",
        "modulation": bundle.modulation,
        "seed": task.seed,
        "preprocessing_profile": rf["preprocessing_profile"],
        "input_size": rf["input_size"],
        "fit_samples": int(fit_values.shape[0]),
        "calibration_samples": int(calibration_values.shape[0]),
        "test_samples": int(test_values.shape[0]),
        "normal_hash_overlap": len(
            set(split.fit_hashes).intersection(split.calibration_hashes)
        ),
        "config": rf,
        "fit": {
            "profile_bins": int(center.size),
            "scale_floored_bins": floored_bins,
            "scale_min": float(scale.min()),
            "scale_max": float(scale.max()),
        },
        "metrics": metrics,
        "timing": {
            "fit_wall_seconds": fit_seconds,
            "calibration_scoring_wall_seconds": calibration_seconds,
            "test_scoring_wall_seconds": test_seconds,
            "milliseconds_per_test_image": 1000.0 * test_seconds / len(test_values),
            "end_to_end_wall_seconds": time.perf_counter() - end_to_end_started,
            "definition": "CPU wall clock; time-mean frequency profile and robust median/MAD scoring",
        },
        "scores_file": scores_path.name if bool(config.get("save_scores", True)) else None,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "device": "cpu",
            "hostname": socket.gethostname(),
            "git_revision": state["revision"],
            "git_dirty": state["dirty"],
        },
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(run_path, result)
    print(
        f"task={task.task_id:02d} modulation={task.modulation} seed={task.seed} "
        f"AUROC={metrics['auroc']:.6f} AUPRC={metrics['auprc']:.6f} "
        f"wrote={run_path}",
        flush=True,
    )
    return run_path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def mean_sd_ci(values: list[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    critical = float(stats.t.ppf(0.975, array.size - 1))
    half = critical * sd / math.sqrt(array.size)
    return mean, sd, mean - half, mean + half


def summarize(
    *, config: dict[str, Any], tasks: list[FinalTask], output_root: Path
) -> None:
    per_run: list[dict[str, Any]] = []
    for task in tasks:
        path = task_output_dir(output_root, task) / "run.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing completed RF-energy result: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["identity"]["config_sha256"] != canonical_hash(config):
            raise RuntimeError(f"result config mismatch: {path}")
        scores = np.load(path.parent / result["scores_file"])
        reproduced = evaluate_detection(
            scores["calibration_scores"],
            scores["test_labels"],
            scores["test_scores"],
            threshold_quantile=float(config["evaluation"]["threshold_quantile"]),
            bootstrap_samples=0,
        )
        if abs(reproduced["auroc"] - result["metrics"]["auroc"]) > 1e-12:
            raise RuntimeError(f"stored RF-energy AUROC does not reproduce: {path}")
        per_run.append(
            {
                "method": result["method"],
                "modulation": result["modulation"],
                "seed": result["seed"],
                "auroc_pct": 100.0 * result["metrics"]["auroc"],
                "auprc_pct": 100.0 * result["metrics"]["auprc"],
                "test_fpr_pct": 100.0 * result["metrics"]["test_fpr"],
                "test_tpr_pct": 100.0 * result["metrics"]["test_tpr"],
                "fit_wall_seconds": result["timing"]["fit_wall_seconds"],
                "test_ms_per_image": result["timing"]["milliseconds_per_test_image"],
            }
        )
    write_csv(output_root / "per_run.csv", per_run)

    per_modulation: list[dict[str, Any]] = []
    for modulation in config["matrix"]["modulations"]:
        selected = [row for row in per_run if row["modulation"] == modulation]
        auroc = mean_sd_ci([float(row["auroc_pct"]) for row in selected])
        auprc = mean_sd_ci([float(row["auprc_pct"]) for row in selected])
        per_modulation.append(
            {
                "modulation": modulation,
                "n_seeds": len(selected),
                "auroc_mean_pct": auroc[0],
                "auroc_sd_pct": auroc[1],
                "auroc_ci95_low_pct": auroc[2],
                "auroc_ci95_high_pct": auroc[3],
                "auprc_mean_pct": auprc[0],
                "auprc_sd_pct": auprc[1],
                "auprc_ci95_low_pct": auprc[2],
                "auprc_ci95_high_pct": auprc[3],
            }
        )
    write_csv(output_root / "per_modulation_summary.csv", per_modulation)

    macro_auroc: list[float] = []
    macro_auprc: list[float] = []
    macro_tpr: list[float] = []
    macro_fpr: list[float] = []
    for seed in config["matrix"]["seeds"]:
        selected = [row for row in per_run if int(row["seed"]) == int(seed)]
        macro_auroc.append(float(np.mean([row["auroc_pct"] for row in selected])))
        macro_auprc.append(float(np.mean([row["auprc_pct"] for row in selected])))
        macro_tpr.append(float(np.mean([row["test_tpr_pct"] for row in selected])))
        macro_fpr.append(float(np.mean([row["test_fpr_pct"] for row in selected])))
    auroc = mean_sd_ci(macro_auroc)
    auprc = mean_sd_ci(macro_auprc)
    summary = {
        "protocol_version": config["protocol_version"],
        "complete_cells": len(per_run),
        "macro_auroc_pct": {
            "mean": auroc[0],
            "sd": auroc[1],
            "ci95": [auroc[2], auroc[3]],
            "per_seed": macro_auroc,
        },
        "macro_auprc_pct": {
            "mean": auprc[0],
            "sd": auprc[1],
            "ci95": [auprc[2], auprc[3]],
            "per_seed": macro_auprc,
        },
        "macro_tpr_pct_mean": float(np.mean(macro_tpr)),
        "macro_fpr_pct_mean": float(np.mean(macro_fpr)),
        "total_fit_seconds": float(
            np.sum([row["fit_wall_seconds"] for row in per_run])
        ),
    }
    atomic_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    tasks = expand_final_tasks(config)
    if args.task_count:
        print(len(tasks))
        return
    if args.list_tasks:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(tasks[0].to_dict()))
        writer.writeheader()
        writer.writerows(task.to_dict() for task in tasks)
        return
    if args.summarize_only:
        summarize(config=config, tasks=tasks, output_root=args.output_root)
        return
    if args.task_id is not None and args.all:
        raise SystemExit("choose either --task-id or --all")
    if args.task_id is None and not args.all:
        raise SystemExit("pass --all or --task-id")
    if args.task_id is not None and not 0 <= args.task_id < len(tasks):
        raise SystemExit(f"task-id must be in [0, {len(tasks) - 1}]")
    dataset_dir = (args.dataset_dir or Path(config["dataset_dir"])).expanduser()
    selected = tasks if args.all else [tasks[args.task_id]]
    for task in selected:
        run_task(
            task,
            config=config,
            dataset_dir=dataset_dir,
            output_root=args.output_root,
            resume=args.resume,
        )
    if args.all:
        summarize(config=config, tasks=tasks, output_root=args.output_root)


if __name__ == "__main__":
    main()
