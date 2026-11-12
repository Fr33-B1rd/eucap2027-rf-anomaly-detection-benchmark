#!/usr/bin/env python
"""Analyze STFPM split-seed versus training-seed diagnostic experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2026, 2027, 2028, 2029, 2030)
CONDITIONS = ("16QAM", "CHIRP", "GMSK", "QPSK")
OUTPUT = ROOT / "outputs" / "iad_stfpm_seed_decoupling_summary"


def tpr_at_one_percent(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    eligible = tpr[fpr <= 0.01 + 1e-12]
    return float(eligible.max()) if eligible.size else 0.0


def calculate(run_dir: Path) -> dict[str, float]:
    with np.load(run_dir / "scores.npz", allow_pickle=False) as stored:
        normal = stored["common_normal_scores"].astype(np.float64)
        values: list[tuple[float, float, float]] = []
        for condition in CONDITIONS:
            anomaly = stored[f"anomaly_scores_{condition.lower()}"].astype(np.float64)
            labels = np.concatenate(
                (np.zeros(normal.size, dtype=np.int8), np.ones(anomaly.size, dtype=np.int8))
            )
            scores = np.concatenate((normal, anomaly))
            values.append(
                (
                    float(roc_auc_score(labels, scores)),
                    tpr_at_one_percent(labels, scores),
                    float(roc_auc_score(labels, scores, max_fpr=0.01)),
                )
            )
    return {
        "macro_auroc": float(np.mean([value[0] for value in values])),
        "macro_tpr_at_fpr_1pct": float(np.mean([value[1] for value in values])),
        "macro_spauc_fpr_1pct": float(np.mean([value[2] for value in values])),
    }


def full50_path(seed: int) -> Path:
    name = (
        "iad_stfpm_local_diagnostic_50epochs"
        if seed in {2027, 2030}
        else "iad_stfpm_local_diagnostic_50epochs_remaining"
    )
    return ROOT / "outputs" / name / "stfpm" / f"seed{seed}"


def paths() -> dict[str, dict[int, tuple[Path, int, int]]]:
    split = ROOT / "outputs" / "iad_stfpm_local_diagnostic_split50" / "stfpm"
    fixed_split = ROOT / "outputs" / "iad_stfpm_seed_decoupling_fixed_split" / "stfpm"
    fixed_training = ROOT / "outputs" / "iad_stfpm_seed_decoupling_fixed_training" / "stfpm"
    baseline = split / "seed2026"
    return {
        "fixed_split": {
            seed: (
                baseline if seed == 2026 else fixed_split / f"seed{seed}",
                2026,
                seed,
            )
            for seed in SEEDS
        },
        "fixed_training": {
            seed: (
                baseline if seed == 2026 else fixed_training / f"seed{seed}",
                seed,
                2026,
            )
            for seed in SEEDS
        },
        "coupled_split": {
            seed: (split / f"seed{seed}", seed, seed) for seed in SEEDS
        },
        "full_training": {
            seed: (full50_path(seed), -1, seed) for seed in SEEDS
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    groups = paths()
    rows: list[dict[str, Any]] = []
    for group, runs in groups.items():
        for axis_seed, (run_dir, split_seed, training_seed) in runs.items():
            run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            rows.append(
                {
                    "group": group,
                    "axis_seed": axis_seed,
                    "split_seed": split_seed,
                    "training_seed": training_seed,
                    **calculate(run_dir),
                    "train_loss_last": float(run["fit"]["train_loss_last"]),
                    "fit_samples": int(
                        run.get("split", run.get("normal_usage", {}))["fit_samples"]
                    ),
                    "run_dir": str(run_dir.relative_to(ROOT)),
                }
            )

    summaries: list[dict[str, Any]] = []
    for group in groups:
        selected = [row for row in rows if row["group"] == group]
        summary: dict[str, Any] = {
            "group": group,
            "n": len(selected),
            "fit_samples": selected[0]["fit_samples"],
        }
        for metric in (
            "macro_auroc",
            "macro_tpr_at_fpr_1pct",
            "macro_spauc_fpr_1pct",
            "train_loss_last",
        ):
            values = np.asarray([float(row[metric]) for row in selected])
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_sd"] = float(values.std(ddof=1))
            summary[f"{metric}_range"] = float(values.max() - values.min())
        summaries.append(summary)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "per_run.csv", rows)
    write_csv(OUTPUT / "group_summary.csv", summaries)

    colors = {
        "fixed_split": "#0072B2",
        "fixed_training": "#009E73",
        "coupled_split": "#E69F00",
        "full_training": "#D55E00",
    }
    labels = {
        "fixed_split": "Fixed split; vary training seed",
        "fixed_training": "Fixed training seed; vary split",
        "coupled_split": "Original coupled split protocol",
        "full_training": "All 4,388 normals",
    }
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for group in groups:
        selected = sorted(
            (row for row in rows if row["group"] == group),
            key=lambda row: int(row["axis_seed"]),
        )
        ax.plot(
            [int(row["axis_seed"]) for row in selected],
            [100.0 * float(row["macro_auroc"]) for row in selected],
            marker="o",
            linewidth=1.3,
            color=colors[group],
            label=labels[group],
        )
    ax.set_xlabel("Varied seed")
    ax.set_ylabel("Macro AUROC (%)")
    ax.set_xticks(SEEDS)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUTPUT / "stfpm_seed_decoupling.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUTPUT / "stfpm_seed_decoupling.pdf", bbox_inches="tight")
    plt.close(fig)

    payload = {"status": "ok", "group_summary": summaries}
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
