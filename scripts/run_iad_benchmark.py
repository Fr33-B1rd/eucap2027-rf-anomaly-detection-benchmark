#!/usr/bin/env python
"""Run one resumable method-by-seed cell with all normal training data fitted.

This protocol has no held-out calibration subset and no calibration-derived
decision threshold.  EfficientAD and official CFLOW still require normal-score
references to construct their final image scores; those references are computed
from the same fitting normals and are recorded as method-internal fitting state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
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
import torch
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from iad_detection.common_normal_protocol import (  # noqa: E402
    CommonNormalTask,
    canonical_config_sha256,
    expand_common_normal_tasks,
)
from scripts.run_iad_common_normal import (  # noqa: E402
    adapt_smoke_config,
    atomic_json,
    atomic_scores,
    resolve_method_config,
    run_cflow,
    run_generic_detector,
    set_seed,
    validate_and_load_sources,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/iad_benchmark.json",
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
        default=REPOSITORY_ROOT / "outputs/iad_benchmark",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_full_training_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "protocol_version",
        "dataset_dir",
        "protocol",
        "matrix",
        "evaluation",
        "methods",
        "assets",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"full-training config is missing {sorted(missing)}")

    protocol = config["protocol"]
    if str(protocol.get("normal_reference_source", "")).upper() != "CHIRP":
        raise ValueError("normal_reference_source must be CHIRP")
    if str(protocol.get("normal_test_source", "")).upper() != "CHIRP":
        raise ValueError("normal_test_source must be CHIRP")
    conditions = [str(value).upper() for value in protocol.get("anomaly_conditions", [])]
    if conditions != ["16QAM", "CHIRP", "GMSK", "QPSK"]:
        raise ValueError("anomaly_conditions must be 16QAM, CHIRP, GMSK, QPSK")
    normal_usage = protocol.get("normal_usage")
    allowed_normal_usage = {
        "all_reference_normals_for_fitting",
        "seeded_nested_reference_subset_for_fitting",
    }
    if normal_usage not in allowed_normal_usage:
        raise ValueError(
            f"normal_usage must be one of {sorted(allowed_normal_usage)}"
        )
    if (
        normal_usage == "seeded_nested_reference_subset_for_fitting"
        and "normal_train_budget" not in protocol
    ):
        raise ValueError("subset fitting requires protocol.normal_train_budget")
    if protocol.get("held_out_calibration") is not False:
        raise ValueError("held_out_calibration must be false")

    evaluation = config["evaluation"]
    forbidden = {"calibration_fraction", "threshold_quantile"}.intersection(evaluation)
    if forbidden:
        raise ValueError(
            "full-training evaluation cannot define calibration settings: "
            f"{sorted(forbidden)}"
        )
    max_fpr = float(evaluation.get("partial_auc_max_fpr", 0.01))
    if not 0.0 < max_fpr < 1.0:
        raise ValueError("partial_auc_max_fpr must be between zero and one")

    matrix = config["matrix"]
    for key in ("gpu_methods", "cpu_methods", "seeds"):
        if key not in matrix:
            raise ValueError(f"matrix.{key} is required")
    all_methods = list(matrix["gpu_methods"]) + list(matrix["cpu_methods"])
    if not all_methods or not matrix["seeds"]:
        raise ValueError("the method matrix and seed list must be non-empty")
    if len(set(all_methods)) != len(all_methods):
        raise ValueError("a method cannot occur in both task lists")
    if set(all_methods).difference(config["methods"]):
        raise ValueError("every matrix method must have a configuration")
    if "pca_reconstruction" in all_methods or "dinov2_knn" in all_methods:
        raise ValueError("PCA and DINOv2-kNN are excluded from this benchmark matrix")
    return config


def select_fit_indices(
    total_samples: int, *, seed: int, budget: int | None
) -> np.ndarray:
    """Return all indices or a deterministic nested permutation prefix."""
    indices = np.arange(total_samples, dtype=np.int64)
    if budget is None:
        return indices
    if not 1 <= int(budget) <= total_samples:
        raise ValueError(
            f"normal_train_budget must be between 1 and {total_samples}, got {budget}"
        )
    subset_rng = np.random.default_rng(seed)
    return np.sort(subset_rng.permutation(indices)[: int(budget)]).astype(np.int64)


def print_tasks(tasks: list[CommonNormalTask]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(tasks[0].to_dict()), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(task.to_dict() for task in tasks)
    print(buffer.getvalue(), end="")


def code_snapshot_sha256() -> str:
    roots = (
        REPOSITORY_ROOT / "core",
        REPOSITORY_ROOT / "dataio" / "iad",
        REPOSITORY_ROOT / "iad_detection",
    )
    paths = [
        REPOSITORY_ROOT / "scripts/run_iad_common_normal.py",
        REPOSITORY_ROOT / "scripts/run_iad_benchmark.py",
    ]
    for root in roots:
        paths.extend(root.rglob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=lambda value: value.as_posix()):
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


def low_fpr_metrics(
    labels: np.ndarray, scores: np.ndarray, *, max_fpr: float
) -> dict[str, float]:
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    within = fpr <= max_fpr + 1e-15
    reaches_95 = tpr >= 0.95 - 1e-15
    return {
        "tpr_at_fpr_limit": float(np.max(tpr[within])),
        "standardized_partial_auroc": float(
            roc_auc_score(labels, scores, max_fpr=max_fpr)
        ),
        "fpr_at_95pct_tpr": float(np.min(fpr[reaches_95])) if reaches_95.any() else 1.0,
    }


def bootstrap_ranking_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    normal = np.flatnonzero(labels == 0)
    anomaly = np.flatnonzero(labels == 1)
    result = {"auroc": [], "auprc": []}
    for _ in range(samples):
        draw = np.concatenate(
            (
                rng.choice(normal, normal.size, replace=True),
                rng.choice(anomaly, anomaly.size, replace=True),
            )
        )
        result["auroc"].append(float(roc_auc_score(labels[draw], scores[draw])))
        result["auprc"].append(
            float(average_precision_score(labels[draw], scores[draw]))
        )
    return result


def evaluate_without_calibration(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    max_fpr: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("test labels and scores must be finite and aligned")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("test labels must contain normal=0 and anomaly=1")
    result: dict[str, Any] = {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        **low_fpr_metrics(labels, scores, max_fpr=max_fpr),
        "partial_auc_max_fpr": float(max_fpr),
        "n_test_normal": int(np.sum(labels == 0)),
        "n_test_anomaly": int(np.sum(labels == 1)),
        "score_direction": "higher_is_more_anomalous",
        "decision_threshold": None,
    }
    if bootstrap_samples > 0:
        draws = bootstrap_ranking_metrics(
            labels,
            scores,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        for metric, values in draws.items():
            low, high = np.quantile(values, (0.025, 0.975))
            result[f"{metric}_ci95"] = [float(low), float(high)]
    return result


def relabel_fit_reference_semantics(method_result: dict[str, Any]) -> dict[str, Any]:
    """Remove held-out-calibration terminology from reused detector adapters."""

    fit_info = dict(method_result["fit_info"])
    if "map_calibration_seconds" in fit_info:
        fit_info["fit_reference_map_normalization_seconds"] = fit_info.pop(
            "map_calibration_seconds"
        )
    if "calibration_log_probability_maxima" in fit_info:
        fit_info["fit_reference_log_probability_maxima"] = fit_info.pop(
            "calibration_log_probability_maxima"
        )
    timing = dict(method_result["timing"])
    timing["fit_reference_scoring_wall_seconds"] = timing.pop(
        "calibration_scoring_wall_seconds"
    )
    variant = str(method_result["implementation_variant"]).replace(
        "iad_calibration_reference_scoring", "iad_fit_reference_scoring"
    )
    return {
        **method_result,
        "implementation_variant": variant,
        "fit_info": fit_info,
        "fit_reference_scores": np.asarray(
            method_result["calibration_scores"], dtype=np.float64
        ),
        "timing": timing,
    }


def build_metrics_and_arrays(
    *,
    method_result: dict[str, Any],
    config: dict[str, Any],
    task: CommonNormalTask,
    fit_indices: np.ndarray,
    common_normal_indices: np.ndarray,
    anomaly_indices: dict[str, np.ndarray],
    smoke: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    normal_scores = np.asarray(
        method_result["common_normal_scores"], dtype=np.float64
    )
    arrays: dict[str, np.ndarray] = {
        "fit_indices": fit_indices,
        "common_normal_indices": common_normal_indices,
        "common_normal_scores": normal_scores,
    }
    if bool(config["evaluation"].get("save_fit_reference_scores", True)):
        arrays["fit_reference_scores"] = np.asarray(
            method_result["fit_reference_scores"], dtype=np.float64
        )
    metrics: dict[str, Any] = {}
    max_fpr = float(config["evaluation"]["partial_auc_max_fpr"])
    bootstrap_samples = (
        20 if smoke else int(config["evaluation"].get("bootstrap_samples", 0))
    )
    for offset, condition in enumerate(config["protocol"]["anomaly_conditions"]):
        name = str(condition).upper()
        anomaly_scores = np.asarray(
            method_result["anomaly_scores"][name], dtype=np.float64
        )
        labels = np.concatenate(
            (
                np.zeros(normal_scores.size, dtype=np.int64),
                np.ones(anomaly_scores.size, dtype=np.int64),
            )
        )
        scores = np.concatenate((normal_scores, anomaly_scores))
        metrics[name] = evaluate_without_calibration(
            labels,
            scores,
            max_fpr=max_fpr,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=task.seed + offset * 10_007,
        )
        key = name.lower()
        arrays[f"anomaly_indices_{key}"] = anomaly_indices[name]
        arrays[f"anomaly_scores_{key}"] = anomaly_scores
    return metrics, arrays


def main() -> None:
    args = parse_args()
    config = load_full_training_config(args.config)
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
    if args.task_kind != "gpu" or args.device == "cpu":
        raise SystemExit("the full-training benchmark matrix contains GPU tasks only")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    set_seed(task.seed)
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

    normal_train_budget = config["protocol"].get("normal_train_budget")
    fit_indices = select_fit_indices(
        int(reference.train_data.shape[0]),
        seed=task.seed,
        budget=(
            int(normal_train_budget) if normal_train_budget is not None else None
        ),
    )
    fit_indices_sha256 = hashlib.sha256(fit_indices.tobytes()).hexdigest()
    normal_name = str(config["protocol"]["normal_test_source"]).upper()
    common_normal_indices = np.flatnonzero(
        bundles[normal_name].test_labels == 0
    ).astype(np.int64)
    anomaly_indices = {
        str(condition).upper(): np.flatnonzero(
            bundles[str(condition).upper()].test_labels == 1
        ).astype(np.int64)
        for condition in config["protocol"]["anomaly_conditions"]
    }
    method_config = resolve_method_config(config, task.method, asset_root)
    method_config["seed"] = task.seed
    if args.smoke:
        method_config = adapt_smoke_config(task.method, method_config)
        fit_indices = fit_indices[: max(64, 2 * int(method_config.get("batch_size", 32)))]
        common_normal_indices = common_normal_indices[:16]
        anomaly_indices = {name: values[:16] for name, values in anomaly_indices.items()}

    asset_manifest_value = None
    if task.method == "efficientad":
        manifest_path = asset_root / "ASSET_MANIFEST.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing EfficientAD asset manifest: {manifest_path}")
        asset_manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))

    print(json.dumps({"task": task.to_dict(), "output_dir": str(output_dir)}), flush=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    end_to_end_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    if task.method == "cflow":
        raw_method_result = run_cflow(
            task,
            method_config=method_config,
            reference=reference,
            bundles=bundles,
            config=config,
            fit_indices=fit_indices,
            calibration_indices=fit_indices,
            common_normal_indices=common_normal_indices,
            anomaly_indices=anomaly_indices,
            device=device,
            workers=args.workers,
            checkpoint_path=checkpoint_path,
            identity=identity,
            resume=args.resume,
        )
    else:
        raw_method_result = run_generic_detector(
            task,
            method_config=method_config,
            reference=reference,
            bundles=bundles,
            config=config,
            fit_indices=fit_indices,
            calibration_indices=fit_indices,
            common_normal_indices=common_normal_indices,
            anomaly_indices=anomaly_indices,
            device=device,
            workers=args.workers,
        )
    method_result = relabel_fit_reference_semantics(raw_method_result)

    metrics, score_arrays = build_metrics_and_arrays(
        method_result=method_result,
        config=config,
        task=task,
        fit_indices=fit_indices,
        common_normal_indices=common_normal_indices,
        anomaly_indices=anomaly_indices,
        smoke=args.smoke,
    )
    if bool(config.get("save_scores", True)):
        atomic_scores(scores_path, score_arrays)

    state = git_state()
    end_to_end_seconds = time.perf_counter() - end_to_end_started
    timing = dict(method_result["timing"])
    timing["unique_test_samples"] = int(
        common_normal_indices.size + sum(values.size for values in anomaly_indices.values())
    )
    timing["end_to_end_wall_seconds"] = end_to_end_seconds
    timing["definition"] = (
        f"One model is fitted using {fit_indices.size} prespecified reference normals. "
        "The same fitting "
        "normals provide only method-internal score references where required; "
        "no held-out calibration subset or calibrated decision threshold is used. "
        "The frozen model scores the common normal test set and all four anomaly conditions."
    )
    result = {
        "identity": identity,
        "publication_eligible": not args.smoke,
        "method": task.method,
        "seed": task.seed,
        "protocol": config["protocol"],
        "normal_usage": {
            "fit_samples": int(fit_indices.size),
            "fit_indices_sha256": fit_indices_sha256,
            "subset_selection": (
                "seeded permutation prefix; subsets are nested across budgets "
                "for the same seed"
                if normal_train_budget is not None
                else "all reference normals"
            ),
            "held_out_calibration_samples": 0,
            "fit_reference_postprocessing_samples": int(fit_indices.size),
            "fit_reference_is_fitting_set": True,
            "fit_reference_role": (
                "method-internal score construction only; not used to select a decision threshold"
            ),
        },
        "common_test_normal_samples": int(common_normal_indices.size),
        "anomaly_samples": {
            name: int(values.size) for name, values in anomaly_indices.items()
        },
        "implementation_variant": method_result["implementation_variant"],
        "preprocessing_profile": method_result["preprocessing_profile"],
        "input_size": int(method_result["input_size"]),
        "config": method_config,
        "fit": method_result["fit_info"],
        "metrics_by_anomaly_condition": metrics,
        "timing": timing,
        "peak_cuda_memory_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2),
        "scores_file": scores_path.name if bool(config.get("save_scores", True)) else None,
        "score_storage": (
            "Fit-reference and common-normal scores are stored once; each anomaly "
            "condition has its own indices and scores. No calibration scores or threshold are stored."
        ),
        "asset_manifest": asset_manifest_value,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
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
    macro_auroc = float(np.mean([value["auroc"] for value in metrics.values()]))
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
