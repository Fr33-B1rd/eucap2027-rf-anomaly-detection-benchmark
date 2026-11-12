#!/usr/bin/env python
"""Validate and aggregate completed official-CFLOW matrix cells."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from iad_detection.cflow_matrix import (  # noqa: E402
    canonical_config_hash,
    expand_tasks,
    load_matrix_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/iad_cflow_official_matrix.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "outputs/iad_cflow_official_matrix",
    )
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    config = load_matrix_config(args.config)
    tasks = expand_tasks(config)
    args.output_root.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    missing: list[int] = []
    expected_config_hash = canonical_config_hash(config)
    for task in tasks:
        run_path = args.output_root / task.relative_output_dir / "run.json"
        if not run_path.exists():
            missing.append(task.task_id)
            continue
        result = json.loads(run_path.read_text(encoding="utf-8"))
        if result.get("identity", {}).get("config_sha256") != expected_config_hash:
            raise RuntimeError(f"config hash mismatch in {run_path}")
        if result.get("task") != task.to_dict():
            raise RuntimeError(f"task identity mismatch in {run_path}")
        if not result.get("publication_eligible", False):
            raise RuntimeError(f"smoke result found in formal matrix: {run_path}")
        completed.append(result)

    if args.require_complete and missing:
        raise SystemExit(f"matrix is incomplete; missing task IDs: {missing}")

    fields = [
        "task_id",
        "input_size",
        "modulation",
        "seed",
        "auroc",
        "auprc",
        "test_tpr",
        "test_fpr",
        "threshold",
        "fit_seconds",
        "scoring_seconds",
        "peak_cuda_memory_mb",
        "gpu",
        "git_revision",
    ]
    summary_path = args.output_root / "summary.csv"
    temporary_csv = summary_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in sorted(completed, key=lambda item: item["task"]["task_id"]):
            metrics = result["metrics"]
            task = result["task"]
            writer.writerow(
                {
                    "task_id": task["task_id"],
                    "input_size": task["input_size"],
                    "modulation": task["modulation"],
                    "seed": task["seed"],
                    "auroc": metrics["auroc"],
                    "auprc": metrics["auprc"],
                    "test_tpr": metrics["test_tpr"],
                    "test_fpr": metrics["test_fpr"],
                    "threshold": metrics["threshold"],
                    "fit_seconds": result["fit_seconds"],
                    "scoring_seconds": result["scoring_seconds"],
                    "peak_cuda_memory_mb": result["peak_cuda_memory_mb"],
                    "gpu": result["environment"]["gpu"],
                    "git_revision": result["environment"]["git_revision"],
                }
            )
    os.replace(temporary_csv, summary_path)

    per_input_modulation: list[dict[str, Any]] = []
    for input_size in config["matrix"]["input_sizes"]:
        for modulation in config["matrix"]["modulations"]:
            rows = [
                item
                for item in completed
                if item["task"]["input_size"] == int(input_size)
                and item["task"]["modulation"] == modulation
            ]
            values = [float(item["metrics"]["auroc"]) for item in rows]
            per_input_modulation.append(
                {
                    "input_size": int(input_size),
                    "modulation": modulation,
                    "seeds": sorted(item["task"]["seed"] for item in rows),
                    "n": len(values),
                    "auroc_mean": statistics.mean(values) if values else None,
                    "auroc_std": statistics.stdev(values) if len(values) > 1 else None,
                }
            )

    per_input_seed: list[dict[str, Any]] = []
    expected_modulations = set(config["matrix"]["modulations"])
    for input_size in config["matrix"]["input_sizes"]:
        for seed in config["matrix"]["seeds"]:
            rows = [
                item
                for item in completed
                if item["task"]["input_size"] == int(input_size)
                and item["task"]["seed"] == int(seed)
            ]
            modulations = {item["task"]["modulation"] for item in rows}
            per_input_seed.append(
                {
                    "input_size": int(input_size),
                    "seed": int(seed),
                    "modulations": sorted(modulations),
                    "complete": modulations == expected_modulations,
                    "macro_auroc": (
                        statistics.mean(
                            float(item["metrics"]["auroc"]) for item in rows
                        )
                        if rows
                        else None
                    ),
                }
            )

    per_input: list[dict[str, Any]] = []
    for input_size in config["matrix"]["input_sizes"]:
        rows = [
            item
            for item in per_input_seed
            if item["input_size"] == int(input_size) and item["complete"]
        ]
        values = [float(item["macro_auroc"]) for item in rows]
        per_input.append(
            {
                "input_size": int(input_size),
                "complete_seeds": [item["seed"] for item in rows],
                "macro_auroc_mean": statistics.mean(values) if values else None,
                "macro_auroc_std": (
                    statistics.stdev(values) if len(values) > 1 else None
                ),
            }
        )

    aggregate = {
        "protocol_version": config["protocol_version"],
        "expected_tasks": len(tasks),
        "completed_tasks": len(completed),
        "missing_task_ids": missing,
        "per_input_modulation": per_input_modulation,
        "per_input_seed": per_input_seed,
        "per_input": per_input,
    }
    atomic_json(args.output_root / "aggregate.json", aggregate)
    atomic_json(
        args.output_root / "status.json",
        {
            "expected_tasks": len(tasks),
            "completed_tasks": len(completed),
            "missing_task_ids": missing,
        },
    )
    print(
        f"completed {len(completed)}/{len(tasks)}; wrote {summary_path} and "
        f"{args.output_root / 'aggregate.json'}"
    )


if __name__ == "__main__":
    main()
