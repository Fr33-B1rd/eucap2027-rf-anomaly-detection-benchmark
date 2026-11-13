#!/usr/bin/env python
"""Validate and summarize the no-calibration full-training IAD matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from iad_detection.common_normal_protocol import expand_common_normal_tasks  # noqa: E402
from scripts.run_iad_benchmark import (  # noqa: E402
    load_full_training_config,
)


DISPLAY = {
    "patchcore": "PatchCore",
    "efficientad": "EfficientAD",
    "cflow": "CFLOW-AD",
    "stfpm": "STFPM",
    "vae_per": "VAE-PER",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/iad_benchmark.json",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=REPOSITORY_ROOT / "outputs/iad_benchmark",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "outputs/iad_benchmark_summary",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def low_fpr_metrics(
    labels: np.ndarray, scores: np.ndarray, *, max_fpr: float
) -> dict[str, float]:
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    within = fpr <= max_fpr + 1e-15
    best_tpr = float(np.max(tpr[within]))
    empirical_fpr = float(
        np.min(fpr[within & np.isclose(tpr, best_tpr, rtol=0.0, atol=1e-15)])
    )
    reaches_95 = tpr >= 0.95 - 1e-15
    return {
        "tpr_at_fpr_1pct": best_tpr,
        "empirical_fpr_at_tpr_at_fpr_1pct": empirical_fpr,
        "standardized_pauroc_fpr_1pct": float(
            roc_auc_score(labels, scores, max_fpr=max_fpr)
        ),
        "fpr_at_95pct_tpr": float(np.min(fpr[reaches_95])) if reaches_95.any() else 1.0,
    }


def mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def main() -> None:
    args = parse_args()
    config = load_full_training_config(args.config)
    expected_tasks = expand_common_normal_tasks(config, "gpu")
    conditions = [
        str(value).upper() for value in config["protocol"]["anomaly_conditions"]
    ]
    expected_fit_count = int(config["protocol"]["expected_counts"]["normal_reference"])
    runs: dict[tuple[str, int], tuple[dict[str, Any], Path]] = {}
    missing: list[str] = []
    for task in expected_tasks:
        run_path = args.result_root / task.relative_output_dir / "run.json"
        if not run_path.is_file():
            missing.append(f"{task.method}/seed{task.seed}")
            continue
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if not bool(run.get("publication_eligible")):
            raise RuntimeError(f"smoke result found in formal result root: {run_path}")
        if run.get("identity", {}).get("protocol_version") != config["protocol_version"]:
            raise RuntimeError(f"protocol mismatch: {run_path}")
        usage = run.get("normal_usage", {})
        if int(usage.get("fit_samples", -1)) != expected_fit_count:
            raise RuntimeError(f"not all reference normals were fitted: {run_path}")
        if int(usage.get("held_out_calibration_samples", -1)) != 0:
            raise RuntimeError(f"unexpected held-out calibration data: {run_path}")
        if set(run["metrics_by_anomaly_condition"]) != set(conditions):
            raise RuntimeError(f"condition mismatch: {run_path}")
        runs[(task.method, task.seed)] = (run, run_path)
    if missing and not args.allow_incomplete:
        raise RuntimeError(
            f"missing {len(missing)} of {len(expected_tasks)} cells: {missing}"
        )

    per_condition: list[dict[str, Any]] = []
    per_seed_macro: list[dict[str, Any]] = []
    fit_reference: np.ndarray | None = None
    common_normal_reference: np.ndarray | None = None
    anomaly_reference: dict[str, np.ndarray] = {}
    max_fpr = float(config["evaluation"]["partial_auc_max_fpr"])
    for (method, seed), (run, run_path) in sorted(runs.items()):
        scores_path = run_path.parent / str(run["scores_file"])
        if not scores_path.is_file():
            raise FileNotFoundError(f"missing scores: {scores_path}")
        with np.load(scores_path, allow_pickle=False) as stored:
            if "calibration_indices" in stored or "calibration_scores" in stored:
                raise RuntimeError(f"calibration arrays found in full-training result: {scores_path}")
            fit_indices = stored["fit_indices"].astype(np.int64)
            if fit_indices.size != expected_fit_count or not np.array_equal(
                fit_indices, np.arange(expected_fit_count, dtype=np.int64)
            ):
                raise RuntimeError(f"fit indices do not cover the full pool: {scores_path}")
            if fit_reference is None:
                fit_reference = fit_indices
            elif not np.array_equal(fit_indices, fit_reference):
                raise RuntimeError(f"fit indices differ across cells: {scores_path}")

            normal_indices = stored["common_normal_indices"].astype(np.int64)
            normal_scores = stored["common_normal_scores"].astype(np.float64)
            if common_normal_reference is None:
                common_normal_reference = normal_indices
            elif not np.array_equal(normal_indices, common_normal_reference):
                raise RuntimeError(f"common-normal indices differ: {scores_path}")

            condition_rows: list[dict[str, Any]] = []
            for condition in conditions:
                key = condition.lower()
                anomaly_indices = stored[f"anomaly_indices_{key}"].astype(np.int64)
                if condition not in anomaly_reference:
                    anomaly_reference[condition] = anomaly_indices
                elif not np.array_equal(anomaly_indices, anomaly_reference[condition]):
                    raise RuntimeError(f"{condition} anomaly indices differ: {scores_path}")
                anomaly_scores = stored[f"anomaly_scores_{key}"].astype(np.float64)
                labels = np.concatenate(
                    (
                        np.zeros(normal_scores.size, dtype=np.int64),
                        np.ones(anomaly_scores.size, dtype=np.int64),
                    )
                )
                scores = np.concatenate((normal_scores, anomaly_scores))
                ranking = low_fpr_metrics(labels, scores, max_fpr=max_fpr)
                stored_metrics = run["metrics_by_anomaly_condition"][condition]
                auroc = float(roc_auc_score(labels, scores))
                auprc = float(average_precision_score(labels, scores))
                if not np.isclose(auroc, float(stored_metrics["auroc"]), atol=1e-12):
                    raise RuntimeError(f"AUROC mismatch: {run_path} {condition}")
                row = {
                    "method": method,
                    "method_display": DISPLAY[method],
                    "seed": seed,
                    "anomaly_condition": condition,
                    "auroc": auroc,
                    "auprc": auprc,
                    **ranking,
                    "n_test_normal": int(normal_scores.size),
                    "n_test_anomaly": int(anomaly_scores.size),
                }
                condition_rows.append(row)
                per_condition.append(row)

        macro = {
            "method": method,
            "method_display": DISPLAY[method],
            "seed": seed,
            "macro_auroc": float(np.mean([row["auroc"] for row in condition_rows])),
            "macro_auprc": float(np.mean([row["auprc"] for row in condition_rows])),
            "macro_tpr_at_fpr_1pct": float(
                np.mean([row["tpr_at_fpr_1pct"] for row in condition_rows])
            ),
            "macro_standardized_pauroc_fpr_1pct": float(
                np.mean(
                    [row["standardized_pauroc_fpr_1pct"] for row in condition_rows]
                )
            ),
            "macro_fpr_at_95pct_tpr": float(
                np.mean([row["fpr_at_95pct_tpr"] for row in condition_rows])
            ),
            "fit_wall_seconds": float(run["timing"]["training_wall_seconds"]),
            "fit_pipeline_wall_seconds": float(run["timing"]["fit_pipeline_wall_seconds"]),
            "end_to_end_wall_seconds": float(run["timing"]["end_to_end_wall_seconds"]),
            "peak_cuda_memory_mb": float(run["peak_cuda_memory_mb"]),
        }
        per_seed_macro.append(macro)

    method_summary: list[dict[str, Any]] = []
    for method in config["matrix"]["gpu_methods"]:
        selected = [row for row in per_seed_macro if row["method"] == method]
        if not selected:
            continue
        summary: dict[str, Any] = {
            "method": method,
            "method_display": DISPLAY[method],
            "n_complete_seeds": len(selected),
        }
        for metric in (
            "macro_auroc",
            "macro_auprc",
            "macro_tpr_at_fpr_1pct",
            "macro_standardized_pauroc_fpr_1pct",
            "macro_fpr_at_95pct_tpr",
            "fit_wall_seconds",
            "fit_pipeline_wall_seconds",
            "end_to_end_wall_seconds",
            "peak_cuda_memory_mb",
        ):
            mean, sd = mean_sd([float(row[metric]) for row in selected])
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_sd"] = sd
        summary["total_fit_device_hours"] = float(
            sum(float(row["fit_wall_seconds"]) for row in selected) / 3600.0
        )
        method_summary.append(summary)

    write_csv(args.output_dir / "per_condition.csv", per_condition)
    write_csv(args.output_dir / "per_seed_macro.csv", per_seed_macro)
    write_csv(args.output_dir / "method_summary.csv", method_summary)
    atomic_json(
        args.output_dir / "summary.json",
        {
            "protocol_version": config["protocol_version"],
            "expected_cells": len(expected_tasks),
            "complete_cells": len(runs),
            "missing_cells": missing,
            "complete": not missing,
            "held_out_calibration": False,
            "conditions": conditions,
            "methods": method_summary,
        },
    )
    print(
        f"complete={not missing} cells={len(runs)}/{len(expected_tasks)} "
        f"wrote={args.output_dir}"
    )


if __name__ == "__main__":
    main()
