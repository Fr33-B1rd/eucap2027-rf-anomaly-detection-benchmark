#!/usr/bin/env python
"""Summarize the local STFPM split/full-training diagnostic matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2026, 2027, 2028, 2029, 2030)
CONDITIONS = ("16QAM", "CHIRP", "GMSK", "QPSK")
OUTPUT_DIR = ROOT / "outputs" / "iad_stfpm_local_diagnostic_summary"


def run_paths() -> dict[str, dict[int, Path]]:
    split = ROOT / "outputs" / "iad_stfpm_local_diagnostic_split50" / "stfpm"
    full40 = ROOT / "outputs" / "iad_stfpm_local_diagnostic_40epochs" / "stfpm"
    full50_a = ROOT / "outputs" / "iad_stfpm_local_diagnostic_50epochs" / "stfpm"
    full50_b = (
        ROOT / "outputs" / "iad_stfpm_local_diagnostic_50epochs_remaining" / "stfpm"
    )
    full50_wd1e3 = (
        ROOT / "outputs" / "iad_stfpm_local_diagnostic_wd1e3_50epochs" / "stfpm"
    )
    return {
        "split50": {seed: split / f"seed{seed}" for seed in SEEDS},
        "full40": {seed: full40 / f"seed{seed}" for seed in SEEDS},
        "full50": {
            2026: full50_b / "seed2026",
            2027: full50_a / "seed2027",
            2028: full50_b / "seed2028",
            2029: full50_b / "seed2029",
            2030: full50_a / "seed2030",
        },
        "full50_wd1e3": {
            seed: full50_wd1e3 / f"seed{seed}" for seed in SEEDS
        },
    }


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, maximum: float) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    eligible = tpr[fpr <= maximum + 1e-12]
    return float(eligible.max()) if eligible.size else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    paths = run_paths()
    seed_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []

    for protocol, by_seed in paths.items():
        for seed, run_dir in by_seed.items():
            run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            with np.load(run_dir / "scores.npz", allow_pickle=False) as stored:
                normal = stored["common_normal_scores"].astype(np.float64)
                metrics: list[tuple[float, float, float]] = []
                for condition in CONDITIONS:
                    anomaly = stored[f"anomaly_scores_{condition.lower()}"].astype(
                        np.float64
                    )
                    labels = np.concatenate(
                        (np.zeros(normal.size, dtype=np.int8), np.ones(anomaly.size, dtype=np.int8))
                    )
                    scores = np.concatenate((normal, anomaly))
                    auroc = float(roc_auc_score(labels, scores))
                    tpr1 = tpr_at_fpr(labels, scores, 0.01)
                    spauc1 = float(roc_auc_score(labels, scores, max_fpr=0.01))
                    metrics.append((auroc, tpr1, spauc1))
                    condition_rows.append(
                        {
                            "protocol": protocol,
                            "seed": seed,
                            "condition": condition,
                            "auroc": auroc,
                            "tpr_at_fpr_1pct": tpr1,
                            "standardized_pauroc_fpr_1pct": spauc1,
                        }
                    )

            state = run["fit"]["final_state_summary"]
            seed_rows.append(
                {
                    "protocol": protocol,
                    "seed": seed,
                    "fit_samples": int(
                        run.get("normal_usage", {}).get(
                            "fit_samples", run.get("split", {}).get("fit_samples")
                        )
                    ),
                    "epochs": int(run["fit"]["epochs"]),
                    "optimizer_steps": int(run["fit"]["optimizer_steps"]),
                    "macro_auroc": float(np.mean([value[0] for value in metrics])),
                    "macro_tpr_at_fpr_1pct": float(
                        np.mean([value[1] for value in metrics])
                    ),
                    "macro_standardized_pauroc_fpr_1pct": float(
                        np.mean([value[2] for value in metrics])
                    ),
                    "train_loss_last": float(run["fit"]["train_loss_last"]),
                    "student_parameter_l2": float(state["student_parameter_l2"]),
                    "batchnorm_running_mean_rms": float(
                        state["batchnorm_running_mean_rms"]
                    ),
                    "batchnorm_running_var_mean": float(
                        state["batchnorm_running_var_mean"]
                    ),
                    "fit_wall_seconds": float(run["timing"]["training_wall_seconds"]),
                    "python": run["environment"]["python"],
                    "torch": run["environment"]["torch"],
                    "gpu": run["environment"]["gpu"],
                }
            )
            for entry in run["fit"]["epoch_history"]:
                history_rows.append(
                    {
                        "protocol": protocol,
                        "seed": seed,
                        **entry,
                    }
                )

    summary_rows: list[dict[str, Any]] = []
    for protocol in paths:
        group = [row for row in seed_rows if row["protocol"] == protocol]
        summary: dict[str, Any] = {
            "protocol": protocol,
            "n_seeds": len(group),
            "fit_samples": group[0]["fit_samples"],
            "epochs": group[0]["epochs"],
            "optimizer_steps": group[0]["optimizer_steps"],
        }
        for key in (
            "macro_auroc",
            "macro_tpr_at_fpr_1pct",
            "macro_standardized_pauroc_fpr_1pct",
            "train_loss_last",
            "student_parameter_l2",
            "batchnorm_running_mean_rms",
            "batchnorm_running_var_mean",
            "fit_wall_seconds",
        ):
            values = np.asarray([float(row[key]) for row in group])
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_sd"] = float(values.std(ddof=1))
        summary_rows.append(summary)

    correlation_rows: list[dict[str, Any]] = []
    auroc = np.asarray([float(row["macro_auroc"]) for row in seed_rows])
    for key in (
        "train_loss_last",
        "student_parameter_l2",
        "batchnorm_running_mean_rms",
        "batchnorm_running_var_mean",
    ):
        values = np.asarray([float(row[key]) for row in seed_rows])
        correlation_rows.append(
            {
                "diagnostic": key,
                "pearson_with_macro_auroc": float(stats.pearsonr(values, auroc).statistic),
                "spearman_with_macro_auroc": float(stats.spearmanr(values, auroc).statistic),
            }
        )

    prefix_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        history40 = [
            row
            for row in history_rows
            if row["protocol"] == "full40" and row["seed"] == seed
        ]
        history50 = [
            row
            for row in history_rows
            if row["protocol"] == "full50" and row["seed"] == seed
        ][:40]
        prefix_rows.append(
            {
                "seed": seed,
                "max_abs_train_loss_difference_first_40_epochs": max(
                    abs(float(a["train_loss"]) - float(b["train_loss"]))
                    for a, b in zip(history40, history50, strict=True)
                ),
                "max_abs_bn_mean_rms_difference_first_40_epochs": max(
                    abs(
                        float(a["batchnorm_running_mean_rms"])
                        - float(b["batchnorm_running_mean_rms"])
                    )
                    for a, b in zip(history40, history50, strict=True)
                ),
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "per_seed.csv", seed_rows)
    write_csv(OUTPUT_DIR / "per_condition.csv", condition_rows)
    write_csv(OUTPUT_DIR / "epoch_history.csv", history_rows)
    write_csv(OUTPUT_DIR / "protocol_summary.csv", summary_rows)
    write_csv(OUTPUT_DIR / "diagnostic_correlations.csv", correlation_rows)
    write_csv(OUTPUT_DIR / "full40_full50_prefix_check.csv", prefix_rows)

    weight_decay_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        baseline = next(
            row
            for row in seed_rows
            if row["protocol"] == "full50" and row["seed"] == seed
        )
        stronger = next(
            row
            for row in seed_rows
            if row["protocol"] == "full50_wd1e3" and row["seed"] == seed
        )
        weight_decay_rows.append(
            {
                "seed": seed,
                "macro_auroc_wd1e4": baseline["macro_auroc"],
                "macro_auroc_wd1e3": stronger["macro_auroc"],
                "macro_auroc_delta_wd1e3_minus_wd1e4": (
                    float(stronger["macro_auroc"])
                    - float(baseline["macro_auroc"])
                ),
                "train_loss_last_wd1e4": baseline["train_loss_last"],
                "train_loss_last_wd1e3": stronger["train_loss_last"],
            }
        )
    write_csv(OUTPUT_DIR / "weight_decay_comparison.csv", weight_decay_rows)

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    colors = {
        "split50": "#0072B2",
        "full40": "#E69F00",
        "full50": "#D55E00",
        "full50_wd1e3": "#009E73",
    }
    labels = {
        "split50": "3,510 fit / 50 epochs",
        "full40": "4,388 fit / 40 epochs",
        "full50": "4,388 fit / 50 epochs",
        "full50_wd1e3": "4,388 fit / 50 epochs / wd=1e-3",
    }
    for protocol in paths:
        group = sorted(
            (row for row in seed_rows if row["protocol"] == protocol),
            key=lambda row: int(row["seed"]),
        )
        axes[0].plot(
            [int(row["seed"]) for row in group],
            [100.0 * float(row["macro_auroc"]) for row in group],
            marker="o",
            linewidth=1.2,
            color=colors[protocol],
            label=labels[protocol],
        )
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("Macro AUROC (%)")
    axes[0].set_xticks(SEEDS)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=6.8)

    for protocol in paths:
        for seed in SEEDS:
            group = [
                row
                for row in history_rows
                if row["protocol"] == protocol and row["seed"] == seed
            ]
            axes[1].plot(
                [int(row["epoch"]) for row in group],
                [float(row["train_loss"]) for row in group],
                color=colors[protocol],
                alpha=0.45,
                linewidth=0.8,
            )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Training feature-matching loss")
    axes[1].grid(alpha=0.25)
    axes[1].set_title("All losses decrease smoothly")
    fig.tight_layout(pad=0.6)
    fig.savefig(OUTPUT_DIR / "stfpm_diagnostic_matrix.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "stfpm_diagnostic_matrix.pdf", bbox_inches="tight")
    plt.close(fig)

    payload = {
        "status": "ok",
        "protocol_summary": summary_rows,
        "diagnostic_correlations": correlation_rows,
        "full40_is_exact_prefix_of_full50": all(
            row["max_abs_train_loss_difference_first_40_epochs"] == 0.0
            and row["max_abs_bn_mean_rms_difference_first_40_epochs"] == 0.0
            for row in prefix_rows
        ),
        "interpretation_boundary": (
            "Metrics diagnose instability; they must not be used to select a final "
            "STFPM configuration without a prespecified normal-only rule."
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
