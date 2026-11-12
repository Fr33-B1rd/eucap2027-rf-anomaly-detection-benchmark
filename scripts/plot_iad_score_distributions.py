#!/usr/bin/env python
"""Draw reproducible per-method IAD score histograms and KDE curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402


DEFAULT_METHODS = ("patchcore", "efficientad", "cflow", "vae_per")
DEFAULT_MODULATIONS = ("16QAM", "CHIRP", "GMSK", "QPSK")
DISPLAY_NAMES = {
    "patchcore": "PatchCore",
    "efficientad": "EfficientAD",
    "cflow": "CFLOW",
    "stfpm": "STFPM",
    "vae_per": "VAE-PER",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--cflow-root", type=Path, required=True)
    parser.add_argument("--cflow-input-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--modulations", nargs="+", default=list(DEFAULT_MODULATIONS))
    parser.add_argument(
        "--seed",
        "--seeds",
        dest="seeds",
        type=int,
        nargs="+",
        default=[2026],
    )
    parser.add_argument("--kind", choices=("hist", "kde", "both"), default="both")
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--bw-method", default="scott")
    parser.add_argument("--include-calibration", action="store_true")
    parser.add_argument("--allow-smoke", action="store_true")
    return parser.parse_args()


def locate_run(
    args: argparse.Namespace, method: str, modulation: str, seed: int
) -> tuple[Path, dict]:
    if method == "cflow":
        path = (
            args.cflow_root
            / f"input{args.cflow_input_size}"
            / modulation
            / f"seed{seed}"
            / "run.json"
        )
    else:
        path = args.final_root / method / modulation / f"seed{seed}" / "run.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing run for plot: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not result.get("publication_eligible") and not args.allow_smoke:
        raise ValueError(
            f"refusing smoke-only result without --allow-smoke: {path}"
        )
    return path, result


def write_plot_data(
    path: Path,
    calibration: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("split", "label", "score", "threshold")
        )
        writer.writeheader()
        for value in calibration:
            writer.writerow(
                {
                    "split": "calibration",
                    "label": 0,
                    "score": value,
                    "threshold": threshold,
                }
            )
        for value, label in zip(scores, labels, strict=True):
            writer.writerow(
                {
                    "split": "test",
                    "label": int(label),
                    "score": value,
                    "threshold": threshold,
                }
            )


def kde_curve(values: np.ndarray, grid: np.ndarray, bandwidth: str):
    if values.size < 2 or float(np.std(values)) == 0.0:
        return None
    return gaussian_kde(values, bw_method=bandwidth)(grid)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for method in args.methods:
            for modulation in (value.upper() for value in args.modulations):
                plot_one(args, method, modulation, seed)


def plot_one(args: argparse.Namespace, method: str, modulation: str, seed: int) -> None:
    run_path, result = locate_run(args, method, modulation, seed)
    scores_path = run_path.with_name(str(result.get("scores_file") or "scores.npz"))
    with np.load(scores_path) as values:
        calibration = np.asarray(values["calibration_scores"], dtype=np.float64)
        scores = np.asarray(values["test_scores"], dtype=np.float64)
        labels = np.asarray(values["test_labels"], dtype=np.int64)
        threshold = float(
            values["threshold"][0]
            if "threshold" in values
            else result["metrics"]["threshold"]
        )
    normal = scores[labels == 0]
    anomaly = scores[labels == 1]
    if normal.size == 0 or anomaly.size == 0:
        raise ValueError(f"both test classes are required: {scores_path}")
    if not np.isfinite(np.concatenate((calibration, scores))).all():
        raise ValueError(f"non-finite score found: {scores_path}")
    combined = np.concatenate((normal, anomaly, calibration))
    low, high = float(combined.min()), float(combined.max())
    margin = 0.02 * max(high - low, 1e-12)
    grid = np.linspace(low - margin, high + margin, 512)

    fig, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    if args.kind in {"hist", "both"}:
        axis.hist(
            normal,
            bins=args.bins,
            density=True,
            alpha=0.35,
            color="#1f77b4",
            label=f"Test normal (n={normal.size})",
        )
        axis.hist(
            anomaly,
            bins=args.bins,
            density=True,
            alpha=0.35,
            color="#d62728",
            label=f"Test anomaly (n={anomaly.size})",
        )
    if args.kind in {"kde", "both"}:
        for values, color, label in (
            (normal, "#1f77b4", "Test normal KDE"),
            (anomaly, "#d62728", "Test anomaly KDE"),
        ):
            density = kde_curve(values, grid, args.bw_method)
            if density is not None:
                axis.plot(grid, density, color=color, linewidth=2, label=label)
        if args.include_calibration:
            density = kde_curve(calibration, grid, args.bw_method)
            if density is not None:
                axis.plot(
                    grid,
                    density,
                    color="#555555",
                    linewidth=1.5,
                    linestyle=":",
                    label="Calibration normal KDE",
                )
    axis.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="95% calibration threshold",
    )
    axis.set_xlabel("Anomaly score (higher = more anomalous)")
    axis.set_ylabel("Density")
    axis.set_title(
        f"{DISPLAY_NAMES.get(method, method)} — {modulation} — seed {seed}\n"
        f"AUROC={100.0 * result['metrics']['auroc']:.2f}%"
    )
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.15)
    stem = f"{method}__{modulation}__seed{seed}__scores"
    fig.savefig(args.output_dir / f"{stem}.png", dpi=300)
    fig.savefig(args.output_dir / f"{stem}.pdf")
    plt.close(fig)
    write_plot_data(
        args.output_dir / f"{stem}.csv",
        calibration,
        scores,
        labels,
        threshold,
    )
    print(f"wrote {stem}.png/.pdf/.csv")


if __name__ == "__main__":
    main()
