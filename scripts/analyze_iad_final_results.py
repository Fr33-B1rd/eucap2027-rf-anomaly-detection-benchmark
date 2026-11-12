#!/usr/bin/env python
"""Validate, aggregate, and compare the final five-method IAD results."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


DEFAULT_METHODS = ("patchcore", "efficientad", "cflow", "stfpm", "vae_per")
DEFAULT_MODULATIONS = ("16QAM", "CHIRP", "GMSK", "QPSK")
DEFAULT_SEEDS = (2026, 2027, 2028, 2029, 2030)
LOCKED_INPUT_SIZES = {
    "patchcore": 64,
    "efficientad": 256,
    "cflow": 64,
    "stfpm": 64,
    "vae_per": 64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--cflow-root", type=Path, required=True)
    parser.add_argument("--cflow-input-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--allow-mixed-environments", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def t_interval(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    mean = statistics.mean(values)
    if statistics.stdev(values) == 0.0:
        return mean, mean
    low, high = stats.t.interval(
        0.95,
        len(values) - 1,
        loc=mean,
        scale=stats.sem(values),
    )
    return float(low), float(high)


def exact_sign_flip_pvalue(differences: list[float]) -> float:
    """Exact two-sided paired randomization p-value over all 2^n sign flips."""

    observed = abs(statistics.mean(differences))
    null_values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        null_values.append(
            abs(statistics.mean(s * value for s, value in zip(signs, differences)))
        )
    tolerance = 1e-15
    return sum(value >= observed - tolerance for value in null_values) / len(
        null_values
    )


def holm_adjust(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [math.nan] * len(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_t_pvalue(differences: list[float]) -> float | None:
    if len(differences) < 2 or statistics.stdev(differences) == 0.0:
        return None
    value = float(stats.ttest_1samp(differences, 0.0).pvalue)
    return value if math.isfinite(value) else None


def add_holm_column(
    rows: list[dict[str, Any]], raw_key: str, adjusted_key: str
) -> None:
    valid = [
        (index, float(row[raw_key]))
        for index, row in enumerate(rows)
        if row[raw_key] is not None and math.isfinite(float(row[raw_key]))
    ]
    adjusted = holm_adjust([value for _, value in valid])
    for row in rows:
        row[adjusted_key] = None
    for (index, _), value in zip(valid, adjusted, strict=True):
        rows[index][adjusted_key] = value


def normalize_result(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    task = result.get("identity", {}).get("task", result.get("task", {}))
    method = str(result.get("method", task.get("method", "")))
    modulation = str(result.get("modulation", task.get("modulation", ""))).upper()
    seed = int(result.get("seed", task.get("seed")))
    timing = result.get("timing")
    if timing is None:
        timing = {
            "initialization_seconds": result["initialization_seconds"],
            "training_wall_seconds": result["fit_seconds"],
            "fit_pipeline_wall_seconds": result["fit_seconds"],
            "calibration_scoring_wall_seconds": result.get(
                "calibration_scoring_seconds", 0.0
            ),
            "test_scoring_wall_seconds": result.get("test_scoring_seconds", 0.0),
            "end_to_end_wall_seconds": (
                result["initialization_seconds"]
                + result["fit_seconds"]
                + result.get("scoring_seconds", 0.0)
            ),
        }
    scores_path = path.with_name(str(result.get("scores_file") or "scores.npz"))
    training_seconds = timing.get(
        "training_wall_seconds",
        timing.get("fit_pipeline_wall_seconds", timing.get("fit_wall_seconds")),
    )
    fit_pipeline_seconds = timing.get(
        "fit_pipeline_wall_seconds",
        timing.get("fit_wall_seconds", training_seconds),
    )
    if training_seconds is None or fit_pipeline_seconds is None:
        raise KeyError(f"missing training timing in {path}")
    return {
        "path": path,
        "scores_path": scores_path,
        "result": result,
        "method": method,
        "modulation": modulation,
        "seed": seed,
        "input_size": int(result.get("input_size", task.get("input_size"))),
        "auroc": float(result["metrics"]["auroc"]),
        "auprc": float(result["metrics"]["auprc"]),
        "auroc_ci95": result["metrics"].get("auroc_ci95"),
        "auprc_ci95": result["metrics"].get("auprc_ci95"),
        "tpr": float(result["metrics"]["test_tpr"]),
        "fpr": float(result["metrics"]["test_fpr"]),
        "training_seconds": float(training_seconds),
        "fit_pipeline_seconds": float(fit_pipeline_seconds),
        "initialization_seconds": float(timing["initialization_seconds"]),
        "test_scoring_seconds": float(timing["test_scoring_wall_seconds"]),
        "end_to_end_seconds": float(timing["end_to_end_wall_seconds"]),
        "peak_cuda_memory_mb": float(result["peak_cuda_memory_mb"]),
        "gpu": result["environment"].get("gpu"),
        "torch": result["environment"].get("torch"),
        "torchvision": result["environment"].get("torchvision"),
        "python": result["environment"].get("python"),
        "git_revision": result["environment"].get("git_revision"),
        "protocol_version": result.get("identity", {}).get("protocol_version"),
        "config_sha256": result.get("identity", {}).get("config_sha256"),
        "source_sha256": result.get("identity", {}).get("source_sha256"),
        "code_snapshot_sha256": result.get("identity", {}).get(
            "code_snapshot_sha256"
        ),
    }


def load_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in args.final_root.glob("*/*/seed*/run.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("publication_eligible"):
            runs.append(normalize_result(path, result))
    cflow_pattern = f"input{args.cflow_input_size}/**/run.json"
    for path in args.cflow_root.glob(cflow_pattern):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("publication_eligible"):
            runs.append(normalize_result(path, result))
    return [run for run in runs if run["method"] in args.methods]


def validate_scores(runs: list[dict[str, Any]]) -> None:
    references: dict[
        tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    for run in runs:
        path = run["scores_path"]
        if not path.is_file():
            raise FileNotFoundError(f"missing score artifact: {path}")
        with np.load(path) as values:
            test_indices = np.asarray(values["test_indices"], dtype=np.int64)
            test_labels = np.asarray(values["test_labels"], dtype=np.int64)
            calibration_indices = np.asarray(
                values["calibration_indices"], dtype=np.int64
            )
            test_scores = np.asarray(values["test_scores"], dtype=np.float64)
            calibration_scores = np.asarray(
                values["calibration_scores"], dtype=np.float64
            )
        if not np.isfinite(test_scores).all() or not np.isfinite(
            calibration_scores
        ).all():
            raise ValueError(f"non-finite scores in {path}")
        reproduced = roc_auc_score(test_labels, test_scores)
        if abs(reproduced - run["auroc"]) > 1e-12:
            raise ValueError(
                f"AUROC mismatch in {path}: {reproduced} != {run['auroc']}"
            )
        key = (run["modulation"], run["seed"])
        if key in references:
            expected_indices, expected_labels, expected_calibration = references[key]
            if (
                not np.array_equal(test_indices, expected_indices)
                or not np.array_equal(test_labels, expected_labels)
                or not np.array_equal(calibration_indices, expected_calibration)
            ):
                raise ValueError(
                    f"unpaired test/calibration samples for {key} in {path}"
                )
        else:
            references[key] = (test_indices, test_labels, calibration_indices)


def validate_protocol(runs: list[dict[str, Any]]) -> None:
    for run in runs:
        expected_size = LOCKED_INPUT_SIZES.get(run["method"])
        if expected_size is not None and run["input_size"] != expected_size:
            raise ValueError(
                f"unexpected input size for {run['method']}: "
                f"{run['input_size']} != {expected_size}"
            )

    for modulation in DEFAULT_MODULATIONS:
        hashes = {
            run["source_sha256"]
            for run in runs
            if run["modulation"] == modulation
        }
        if len(hashes) > 1:
            raise ValueError(f"dataset hash drift for {modulation}: {sorted(hashes)}")

    for method in set(run["method"] for run in runs):
        hashes = {
            run["config_sha256"] for run in runs if run["method"] == method
        }
        if len(hashes) > 1:
            raise ValueError(f"configuration drift for {method}: {sorted(hashes)}")

    final_code_hashes = {
        run["code_snapshot_sha256"]
        for run in runs
        if run["method"] != "cflow"
    }
    if len(final_code_hashes) > 1:
        raise ValueError(f"code drift in final reruns: {sorted(final_code_hashes)}")


def environment_identity(run: dict[str, Any]) -> tuple[Any, ...]:
    """Hardware/framework identity; Python patch releases are metadata only."""

    python_major_minor = ".".join(str(run["python"]).split(".")[:2])
    return run["gpu"], run["torch"], run["torchvision"], python_major_minor


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args)
    identities = [(r["method"], r["modulation"], r["seed"]) for r in runs]
    if len(identities) != len(set(identities)):
        raise RuntimeError("duplicate method/modulation/seed result cells")
    expected = {
        (method, modulation, seed)
        for method in args.methods
        for modulation in DEFAULT_MODULATIONS
        for seed in DEFAULT_SEEDS
    }
    observed = set(identities)
    missing = sorted(expected.difference(observed))
    unexpected = sorted(observed.difference(expected))
    if args.require_complete and (missing or unexpected):
        raise SystemExit(f"incomplete matrix; missing={missing}, unexpected={unexpected}")
    validate_protocol(runs)
    validate_scores(runs)

    environments = sorted({environment_identity(run) for run in runs})
    if len(environments) != 1 and not args.allow_mixed_environments:
        raise RuntimeError(
            f"mixed execution environments found: {environments}; use "
            "--allow-mixed-environments only for exploratory analysis"
        )

    per_run_rows = []
    for run in sorted(runs, key=lambda r: identities.index((r["method"], r["modulation"], r["seed"]))):
        per_run_rows.append(
            {
                "method": run["method"],
                "modulation": run["modulation"],
                "seed": run["seed"],
                "input_size": run["input_size"],
                "auroc": run["auroc"],
                "auroc_bootstrap_ci95_low": (
                    run["auroc_ci95"][0] if run["auroc_ci95"] else None
                ),
                "auroc_bootstrap_ci95_high": (
                    run["auroc_ci95"][1] if run["auroc_ci95"] else None
                ),
                "auprc": run["auprc"],
                "auprc_bootstrap_ci95_low": (
                    run["auprc_ci95"][0] if run["auprc_ci95"] else None
                ),
                "auprc_bootstrap_ci95_high": (
                    run["auprc_ci95"][1] if run["auprc_ci95"] else None
                ),
                "test_tpr": run["tpr"],
                "test_fpr": run["fpr"],
                "training_seconds": run["training_seconds"],
                "fit_pipeline_seconds": run["fit_pipeline_seconds"],
                "initialization_seconds": run["initialization_seconds"],
                "test_scoring_seconds": run["test_scoring_seconds"],
                "end_to_end_seconds": run["end_to_end_seconds"],
                "peak_cuda_memory_mb": run["peak_cuda_memory_mb"],
                "gpu": run["gpu"],
                "torch": run["torch"],
                "torchvision": run["torchvision"],
                "python": run["python"],
                "git_revision": run["git_revision"],
                "protocol_version": run["protocol_version"],
                "config_sha256": run["config_sha256"],
                "source_sha256": run["source_sha256"],
                "code_snapshot_sha256": run["code_snapshot_sha256"],
                "run_json": str(run["path"]),
                "scores_npz": str(run["scores_path"]),
            }
        )
    atomic_csv(
        args.output_dir / "per_run.csv",
        list(per_run_rows[0]) if per_run_rows else [],
        per_run_rows,
    )

    per_modulation_rows: list[dict[str, Any]] = []
    macro_by_method_seed: dict[tuple[str, int], dict[str, float]] = {}
    for method in args.methods:
        for modulation in DEFAULT_MODULATIONS:
            cells = [
                run
                for run in runs
                if run["method"] == method and run["modulation"] == modulation
            ]
            aurocs = [100.0 * run["auroc"] for run in cells]
            auprcs = [100.0 * run["auprc"] for run in cells]
            low, high = t_interval(aurocs)
            per_modulation_rows.append(
                {
                    "method": method,
                    "modulation": modulation,
                    "n_seeds": len(cells),
                    "auroc_mean_pct": statistics.mean(aurocs) if aurocs else None,
                    "auroc_sd_pct": statistics.stdev(aurocs)
                    if len(aurocs) > 1
                    else None,
                    "auroc_t_ci95_low_pct": low,
                    "auroc_t_ci95_high_pct": high,
                    "auprc_mean_pct": statistics.mean(auprcs) if auprcs else None,
                    "auprc_sd_pct": statistics.stdev(auprcs)
                    if len(auprcs) > 1
                    else None,
                }
            )
        for seed in DEFAULT_SEEDS:
            cells = [
                run
                for run in runs
                if run["method"] == method and run["seed"] == seed
            ]
            if {run["modulation"] for run in cells} == set(DEFAULT_MODULATIONS):
                macro_by_method_seed[(method, seed)] = {
                    "auroc": statistics.mean(100.0 * run["auroc"] for run in cells),
                    "auprc": statistics.mean(100.0 * run["auprc"] for run in cells),
                }
    atomic_csv(
        args.output_dir / "per_modulation_summary.csv",
        list(per_modulation_rows[0]),
        per_modulation_rows,
    )

    method_rows: list[dict[str, Any]] = []
    for method in args.methods:
        seed_rows = [
            macro_by_method_seed[(method, seed)]
            for seed in DEFAULT_SEEDS
            if (method, seed) in macro_by_method_seed
        ]
        aurocs = [row["auroc"] for row in seed_rows]
        auprcs = [row["auprc"] for row in seed_rows]
        cells = [run for run in runs if run["method"] == method]
        training = [run["training_seconds"] for run in cells]
        fit_pipeline = [run["fit_pipeline_seconds"] for run in cells]
        low, high = t_interval(aurocs)
        method_rows.append(
            {
                "method": method,
                "n_complete_seeds": len(seed_rows),
                "macro_auroc_mean_pct": statistics.mean(aurocs) if aurocs else None,
                "macro_auroc_sd_pct": statistics.stdev(aurocs)
                if len(aurocs) > 1
                else None,
                "macro_auroc_t_ci95_low_pct": low,
                "macro_auroc_t_ci95_high_pct": high,
                "macro_auprc_mean_pct": statistics.mean(auprcs) if auprcs else None,
                "macro_auprc_sd_pct": statistics.stdev(auprcs)
                if len(auprcs) > 1
                else None,
                "training_time_mean_seconds": statistics.mean(training)
                if training
                else None,
                "training_time_sd_seconds": statistics.stdev(training)
                if len(training) > 1
                else None,
                "training_time_median_seconds": statistics.median(training)
                if training
                else None,
                "training_time_q25_seconds": float(np.quantile(training, 0.25))
                if training
                else None,
                "training_time_q75_seconds": float(np.quantile(training, 0.75))
                if training
                else None,
                "training_time_total_gpu_hours": sum(training) / 3600.0,
                "fit_pipeline_time_mean_seconds": statistics.mean(fit_pipeline)
                if fit_pipeline
                else None,
                "peak_cuda_memory_mean_mb": statistics.mean(
                    run["peak_cuda_memory_mb"] for run in cells
                )
                if cells
                else None,
            }
        )
    atomic_csv(
        args.output_dir / "method_summary.csv",
        list(method_rows[0]),
        method_rows,
    )

    pairwise_rows: list[dict[str, Any]] = []
    for method_a, method_b in itertools.combinations(args.methods, 2):
        seeds = [
            seed
            for seed in DEFAULT_SEEDS
            if (method_a, seed) in macro_by_method_seed
            and (method_b, seed) in macro_by_method_seed
        ]
        differences = [
            macro_by_method_seed[(method_a, seed)]["auroc"]
            - macro_by_method_seed[(method_b, seed)]["auroc"]
            for seed in seeds
        ]
        low, high = t_interval(differences)
        t_p = paired_t_pvalue(differences)
        exact_p = exact_sign_flip_pvalue(differences) if differences else None
        if not differences:
            continue
        pairwise_rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "difference_direction": "method_a_minus_method_b_macro_auroc_pct",
                "n_paired_seeds": len(seeds),
                "mean_difference_pct_points": statistics.mean(differences)
                if differences
                else None,
                "sd_difference_pct_points": statistics.stdev(differences)
                if len(differences) > 1
                else None,
                "t_ci95_low_pct_points": low,
                "t_ci95_high_pct_points": high,
                "cohen_dz": statistics.mean(differences)
                / statistics.stdev(differences)
                if len(differences) > 1 and statistics.stdev(differences) > 0
                else None,
                "paired_t_p_raw": t_p,
                "exact_sign_flip_p_raw": exact_p,
                "paired_seed_values": json.dumps(seeds),
            }
        )
    add_holm_column(pairwise_rows, "paired_t_p_raw", "paired_t_p_holm")
    add_holm_column(
        pairwise_rows,
        "exact_sign_flip_p_raw",
        "exact_sign_flip_p_holm",
    )
    atomic_csv(
        args.output_dir / "pairwise_macro_comparisons.csv",
        list(pairwise_rows[0]) if pairwise_rows else [],
        pairwise_rows,
    )

    analysis = {
        "methods": args.methods,
        "modulations": list(DEFAULT_MODULATIONS),
        "seeds": list(DEFAULT_SEEDS),
        "expected_cells": len(expected),
        "observed_cells": len(observed),
        "missing_cells": missing,
        "unexpected_cells": unexpected,
        "environments": environments,
        "inference_unit": (
            "per-seed macro average across the four independently trained "
            "modulation datasets"
        ),
        "primary_uncertainty": "sample SD and Student-t 95% CI across five seeds",
        "single_run_uncertainty": (
            "stratified nonparametric bootstrap over normal/anomaly test samples; "
            "these intervals condition on the fitted model"
        ),
        "pairwise_testing": (
            "same-seed paired macro differences; two-sided paired t-test and "
            "exact 2^5 sign-flip randomization test; Holm correction over all pairs"
        ),
        "small_n_warning": (
            "with five seeds, a two-sided exact sign-flip p-value cannot be below "
            "0.0625; emphasize effect sizes and confidence intervals"
        ),
        "scope_warning": (
            "seed-level inference quantifies split/optimization repeatability "
            "conditional on these four IAD modulation datasets; seeds are not "
            "independent dataset replicates and do not establish generalization "
            "to unseen modulation families"
        ),
        "score_integrity": "all stored AUROCs reproduced from scores.npz",
    }
    atomic_json(args.output_dir / "analysis.json", analysis)
    print(
        f"validated {len(observed)}/{len(expected)} cells; wrote analysis to "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
