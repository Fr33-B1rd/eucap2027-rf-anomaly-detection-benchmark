#!/usr/bin/env python
"""Validate and summarize the formal IAD training-sample matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

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
    "vae_per": "VAE-PER",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-64",
        type=Path,
        default=REPOSITORY_ROOT / "configs/iad_training_samples_n64.json",
    )
    parser.add_argument(
        "--config-256",
        type=Path,
        default=REPOSITORY_ROOT / "configs/iad_training_samples_n256.json",
    )
    parser.add_argument(
        "--configs",
        type=Path,
        nargs="+",
        help=(
            "Config files to collect. If omitted, the original N=64 and N=256 "
            "formal configs are used."
        ),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=REPOSITORY_ROOT / "outputs/iad_training_samples",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "outputs/iad_training_samples_summary",
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


def mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(array.mean()),
        float(array.std(ddof=1)) if array.size > 1 else 0.0,
    )


def main() -> None:
    args = parse_args()
    configs: dict[int, tuple[dict[str, Any], Path]] = {}
    config_paths = args.configs or [args.config_64, args.config_256]
    for path in config_paths:
        config = load_full_training_config(path)
        budget = int(config["protocol"]["normal_train_budget"])
        if budget in configs:
            raise RuntimeError(f"duplicate budget config: {budget}")
        configs[budget] = (config, path)
    if not configs:
        raise RuntimeError("at least one budget config is required")

    per_condition: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    missing: list[str] = []
    fit_indices_by_budget_seed: dict[tuple[int, int], np.ndarray] = {}
    common_normal_reference: np.ndarray | None = None
    anomaly_references: dict[str, np.ndarray] = {}
    expected_cells = 0

    for budget, (config, _path) in sorted(configs.items()):
        tasks = expand_common_normal_tasks(config, "gpu")
        if len(tasks) != 20:
            raise RuntimeError(f"budget {budget}: expected 20 tasks, got {len(tasks)}")
        expected_cells += len(tasks)
        conditions = [
            str(value).upper() for value in config["protocol"]["anomaly_conditions"]
        ]
        expected_normal = int(config["protocol"]["expected_counts"]["normal_test"])
        expected_anomaly = int(
            config["protocol"]["expected_counts"]["anomaly_per_condition"]
        )

        for task in tasks:
            run_path = (
                args.result_root
                / f"n{budget}"
                / task.relative_output_dir
                / "run.json"
            )
            if not run_path.is_file():
                missing.append(f"n{budget}/{task.method}/seed{task.seed}")
                continue
            run = json.loads(run_path.read_text(encoding="utf-8"))
            if not bool(run.get("publication_eligible")):
                raise RuntimeError(f"smoke result in formal root: {run_path}")
            if run.get("identity", {}).get("protocol_version") != config["protocol_version"]:
                raise RuntimeError(f"protocol mismatch: {run_path}")
            usage = run.get("normal_usage", {})
            if int(usage.get("fit_samples", -1)) != budget:
                raise RuntimeError(f"fit budget mismatch: {run_path}")
            if int(usage.get("held_out_calibration_samples", -1)) != 0:
                raise RuntimeError(f"unexpected held-out calibration: {run_path}")

            scores_path = run_path.parent / str(run["scores_file"])
            if not scores_path.is_file():
                raise FileNotFoundError(f"missing scores: {scores_path}")
            with np.load(scores_path, allow_pickle=False) as stored:
                if "calibration_indices" in stored or "calibration_scores" in stored:
                    raise RuntimeError(f"calibration arrays found: {scores_path}")
                fit_indices = stored["fit_indices"].astype(np.int64)
                if fit_indices.size != budget or np.unique(fit_indices).size != budget:
                    raise RuntimeError(f"invalid fit indices: {scores_path}")
                digest = hashlib.sha256(fit_indices.tobytes()).hexdigest()
                if digest != usage.get("fit_indices_sha256"):
                    raise RuntimeError(f"fit-index digest mismatch: {scores_path}")
                subset_key = (budget, int(task.seed))
                previous = fit_indices_by_budget_seed.get(subset_key)
                if previous is None:
                    fit_indices_by_budget_seed[subset_key] = fit_indices
                elif not np.array_equal(previous, fit_indices):
                    raise RuntimeError(
                        f"methods use different subsets for budget/seed {subset_key}"
                    )

                normal_indices = stored["common_normal_indices"].astype(np.int64)
                normal_scores = stored["common_normal_scores"].astype(np.float64)
                if normal_indices.size != expected_normal:
                    raise RuntimeError(f"normal-test count mismatch: {scores_path}")
                if common_normal_reference is None:
                    common_normal_reference = normal_indices
                elif not np.array_equal(normal_indices, common_normal_reference):
                    raise RuntimeError(f"common-normal indices differ: {scores_path}")
                if not np.isfinite(normal_scores).all():
                    raise RuntimeError(f"non-finite normal scores: {scores_path}")

                condition_rows: list[dict[str, Any]] = []
                for condition in conditions:
                    key = condition.lower()
                    anomaly_indices = stored[f"anomaly_indices_{key}"].astype(np.int64)
                    anomaly_scores = stored[f"anomaly_scores_{key}"].astype(np.float64)
                    if anomaly_indices.size != expected_anomaly:
                        raise RuntimeError(f"anomaly count mismatch: {scores_path} {condition}")
                    if not np.isfinite(anomaly_scores).all():
                        raise RuntimeError(f"non-finite anomaly scores: {scores_path} {condition}")
                    if condition not in anomaly_references:
                        anomaly_references[condition] = anomaly_indices
                    elif not np.array_equal(anomaly_indices, anomaly_references[condition]):
                        raise RuntimeError(f"anomaly indices differ: {scores_path} {condition}")

                    labels = np.concatenate(
                        (
                            np.zeros(normal_scores.size, dtype=np.int64),
                            np.ones(anomaly_scores.size, dtype=np.int64),
                        )
                    )
                    scores = np.concatenate((normal_scores, anomaly_scores))
                    auroc = float(roc_auc_score(labels, scores))
                    stored_auroc = float(
                        run["metrics_by_anomaly_condition"][condition]["auroc"]
                    )
                    if not np.isclose(auroc, stored_auroc, atol=1e-12):
                        raise RuntimeError(f"AUROC mismatch: {run_path} {condition}")
                    row = {
                        "budget": budget,
                        "method": task.method,
                        "method_display": DISPLAY[task.method],
                        "seed": int(task.seed),
                        "anomaly_condition": condition,
                        "auroc": auroc,
                        "n_test_normal": int(normal_scores.size),
                        "n_test_anomaly": int(anomaly_scores.size),
                    }
                    condition_rows.append(row)
                    per_condition.append(row)

            per_seed.append(
                {
                    "budget": budget,
                    "method": task.method,
                    "method_display": DISPLAY[task.method],
                    "seed": int(task.seed),
                    "macro_auroc": float(
                        np.mean([row["auroc"] for row in condition_rows])
                    ),
                    "fit_wall_seconds": float(run["timing"]["training_wall_seconds"]),
                    "end_to_end_wall_seconds": float(
                        run["timing"]["end_to_end_wall_seconds"]
                    ),
                    "peak_cuda_memory_mb": float(run["peak_cuda_memory_mb"]),
                }
            )

    if missing and not args.allow_incomplete:
        raise RuntimeError(f"missing {len(missing)} of {expected_cells} cells: {missing}")

    budgets = sorted(configs)
    for seed in (2026, 2027, 2028, 2029, 2030):
        for smaller_budget, larger_budget in zip(budgets, budgets[1:]):
            small = fit_indices_by_budget_seed.get((smaller_budget, seed))
            large = fit_indices_by_budget_seed.get((larger_budget, seed))
            if small is None or large is None:
                continue
            if not set(small.tolist()).issubset(set(large.tolist())):
                raise RuntimeError(
                    f"subsets are not nested for seed {seed}: "
                    f"N={smaller_budget} is not contained in N={larger_budget}"
                )

    method_summary: list[dict[str, Any]] = []
    for budget in budgets:
        for method in configs[budget][0]["matrix"]["gpu_methods"]:
            selected = [
                row
                for row in per_seed
                if row["budget"] == budget and row["method"] == method
            ]
            if not selected:
                continue
            auroc_mean, auroc_sd = mean_sd(
                [float(row["macro_auroc"]) for row in selected]
            )
            fit_mean, fit_sd = mean_sd(
                [float(row["fit_wall_seconds"]) for row in selected]
            )
            method_summary.append(
                {
                    "budget": budget,
                    "method": method,
                    "method_display": DISPLAY[method],
                    "n_seeds": len(selected),
                    "macro_auroc_mean": auroc_mean,
                    "macro_auroc_sd": auroc_sd,
                    "fit_wall_seconds_mean": fit_mean,
                    "fit_wall_seconds_sd": fit_sd,
                }
            )

    write_csv(args.output_dir / "per_condition.csv", per_condition)
    write_csv(args.output_dir / "per_seed_macro.csv", per_seed)
    write_csv(args.output_dir / "method_budget_summary.csv", method_summary)
    atomic_json(
        args.output_dir / "summary.json",
        {
            "expected_cells": expected_cells,
            "complete_cells": len(per_seed),
            "complete": not missing,
            "missing_cells": missing,
            "budgets": budgets,
            "nested_subsets_verified": not missing,
            "methods": method_summary,
        },
    )
    print(
        f"complete={not missing} cells={len(per_seed)}/{expected_cells} "
        f"summary={args.output_dir}"
    )


if __name__ == "__main__":
    main()
