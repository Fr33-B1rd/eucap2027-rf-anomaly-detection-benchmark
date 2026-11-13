#!/usr/bin/env python
"""Plot q99 confusion matrices for the common-normal calibration protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("efficientad", "patchcore", "cflow", "vae_per")
SEEDS = (2026, 2027, 2028, 2029, 2030)
CONDITIONS = ("16qam", "chirp", "gmsk", "qpsk")
DISPLAY = {
    "efficientad": "EfficientAD",
    "patchcore": "PatchCore",
    "cflow": "CFLOW-AD",
    "vae_per": "VAE-PER",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=ROOT / "outputs/iad_common_normal",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/figure_previews",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=ROOT / "outputs/iad_common_normal_summary/q99_confusion_counts.csv",
    )
    return parser.parse_args()


def load_counts(result_root: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for method in METHODS:
        for seed in SEEDS:
            run_dir = result_root / method / f"seed{seed}"
            run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            if not run.get("publication_eligible"):
                raise RuntimeError(f"non-publication run: {run_dir}")
            with np.load(run_dir / run["scores_file"], allow_pickle=False) as stored:
                calibration = stored["calibration_scores"].astype(np.float64)
                normal = stored["common_normal_scores"].astype(np.float64)
                anomaly = np.concatenate(
                    [stored[f"anomaly_scores_{name}"].astype(np.float64) for name in CONDITIONS]
                )
                threshold = float(stored["threshold"].reshape(-1)[0])
            expected = float(np.quantile(calibration, 0.99, method="higher"))
            if not np.isclose(threshold, expected, rtol=0.0, atol=1e-12):
                raise RuntimeError(
                    f"stored threshold is not calibration q99: {run_dir}: "
                    f"{threshold} != {expected}"
                )
            if normal.shape != (788,) or anomaly.shape != (4 * 774,):
                raise RuntimeError(f"unexpected test shape: {run_dir}")
            predicted_normal = normal >= threshold
            predicted_anomaly = anomaly >= threshold
            tn = int(np.sum(~predicted_normal))
            fp = int(np.sum(predicted_normal))
            fn = int(np.sum(~predicted_anomaly))
            tp = int(np.sum(predicted_anomaly))
            rows.append(
                {
                    "method": method,
                    "method_display": DISPLAY[method],
                    "seed": seed,
                    "threshold": threshold,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                    "normal_total": len(normal),
                    "anomaly_total": len(anomaly),
                    "fpr": fp / len(normal),
                    "tpr": tp / len(anomaly),
                }
            )
    return rows


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = load_counts(args.result_root)
    write_rows(args.summary_csv, rows)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        1,
        len(METHODS),
        figsize=(7.12, 1.95),
        layout="constrained",
    )
    image = None
    for axis, method in zip(axes, METHODS, strict=True):
        selected = [row for row in rows if row["method"] == method]
        matrices = np.asarray(
            [[[row["tn"], row["fp"]], [row["fn"], row["tp"]]] for row in selected],
            dtype=np.float64,
        )
        means = np.mean(matrices, axis=0)
        sds = np.std(matrices, axis=0, ddof=1)
        rates = means / np.asarray([[788.0], [4.0 * 774.0]]) * 100.0
        image = axis.imshow(rates, vmin=0.0, vmax=100.0, cmap="Blues")
        for row_index in range(2):
            for column_index in range(2):
                rate = rates[row_index, column_index]
                color = "white" if rate >= 55.0 else "black"
                label = (("TN", "FP"), ("FN", "TP"))[row_index][column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{label}\n{means[row_index, column_index]:.1f} $\\pm$ {sds[row_index, column_index]:.1f}\n({rate:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=6.3,
                    color=color,
                    linespacing=1.05,
                )
        axis.set_title(DISPLAY[method], pad=3)
        axis.set_xticks([0, 1], ["Normal", "Anomaly"])
        axis.set_yticks([0, 1], ["Normal", "Anomaly"])
        axis.set_xlabel("Predicted", fontsize=7)
        axis.tick_params(labelsize=6.5, length=0)
    axes[0].set_ylabel("True class", fontsize=7)
    for axis in axes[1:]:
        axis.set_yticklabels([])
    if image is None:
        raise RuntimeError("no confusion matrices generated")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.018, pad=0.02)
    colorbar.set_label("Within-class rate (%)", fontsize=6.8)
    colorbar.ax.tick_params(labelsize=6)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "iad_common_normal_q99_confusion_matrices_preview"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
