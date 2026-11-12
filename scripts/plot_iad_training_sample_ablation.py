#!/usr/bin/env python
"""Plot the formal IAD normal-training-sample ablation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("efficientad", "patchcore", "cflow", "vae_per")
DISPLAY = {
    "efficientad": "EfficientAD",
    "patchcore": "PatchCore",
    "cflow": "CFLOW-AD",
    "vae_per": "VAE-PER",
}
COLORS = {
    "efficientad": "#0072B2",
    "patchcore": "#E69F00",
    "cflow": "#CC79A7",
    "vae_per": "#D55E00",
}
MARKERS = {
    "efficientad": "o",
    "patchcore": "s",
    "cflow": "^",
    "vae_per": "D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation-summary",
        type=Path,
        default=ROOT / "outputs/iad_training_samples_summary/method_budget_summary.csv",
    )
    parser.add_argument(
        "--full-summary",
        type=Path,
        default=ROOT / "outputs/iad_full_training_summary/method_summary.csv",
    )
    parser.add_argument(
        "--additional-ablation-summary",
        type=Path,
        action="append",
        default=[],
        help="Additional method_budget_summary.csv files to merge by budget.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/figure_previews",
    )
    parser.add_argument(
        "--no-inset",
        action="store_true",
        help="Omit the near-ceiling inset and save a separate clean preview.",
    )
    parser.add_argument(
        "--min-budget",
        type=int,
        help="Plot only fitting budgets greater than or equal to this value.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_values(args: argparse.Namespace) -> dict[str, dict[int, tuple[float, float]]]:
    values: dict[str, dict[int, tuple[float, float]]] = {
        method: {} for method in METHODS
    }
    for summary in [args.ablation_summary, *args.additional_ablation_summary]:
        for row in read_rows(summary):
            method = row["method"]
            if method not in values:
                continue
            budget = int(row["budget"])
            current = (
                100.0 * float(row["macro_auroc_mean"]),
                100.0 * float(row["macro_auroc_sd"]),
            )
            previous = values[method].get(budget)
            if previous is not None and not np.allclose(previous, current, atol=1e-10):
                raise RuntimeError(
                    f"conflicting values for {method} at budget {budget}: "
                    f"{previous} versus {current}"
                )
            values[method][budget] = current
    for row in read_rows(args.full_summary):
        method = row["method"]
        if method not in values:
            continue
        values[method][4388] = (
            100.0 * float(row["macro_auroc_mean"]),
            100.0 * float(row["macro_auroc_sd"]),
        )
    expected_budgets = set(values[METHODS[0]])
    for method in METHODS:
        if set(values[method]) != expected_budgets:
            raise RuntimeError(f"incomplete budgets for {method}: {values[method]}")
    if args.min_budget is not None:
        values = {
            method: {
                budget: result
                for budget, result in method_values.items()
                if budget >= args.min_budget
            }
            for method, method_values in values.items()
        }
        if not values[METHODS[0]]:
            raise RuntimeError(f"no budgets remain at or above {args.min_budget}")
    return values


def draw_curves(
    axis: plt.Axes,
    values: dict[str, dict[int, tuple[float, float]]],
    *,
    labels: bool,
) -> None:
    budgets = np.asarray(sorted(values[METHODS[0]]), dtype=np.float64)
    for method in METHODS:
        means = np.asarray([values[method][int(n)][0] for n in budgets])
        sds = np.asarray([values[method][int(n)][1] for n in budgets])
        axis.errorbar(
            budgets,
            means,
            yerr=sds,
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=4.2,
            markeredgecolor="white",
            markeredgewidth=0.45,
            linewidth=1.25,
            elinewidth=0.75,
            capsize=2.0,
            label=DISPLAY[method] if labels else None,
            zorder=3,
        )
    axis.set_xscale("log")
    axis.set_xlim(float(budgets[0]) * 0.78, float(budgets[-1]) * 1.22)
    axis.set_xticks(
        budgets,
        [f"{int(value):,}" for value in budgets],
    )
    axis.grid(alpha=0.22, linewidth=0.45)
    axis.tick_params(labelsize=7)


def main() -> None:
    args = parse_args()
    values = load_values(args)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axis = plt.subplots(figsize=(4.1, 2.75))
    draw_curves(axis, values, labels=True)
    lower_values = [
        mean - sd
        for method_values in values.values()
        for mean, sd in method_values.values()
    ]
    lower_limit = 5.0 * np.floor((min(lower_values) - 1.0) / 5.0)
    axis.set_ylim(max(0.0, lower_limit), 100.35)
    axis.set_xlabel("Normal fitting samples")
    axis.set_ylabel("Macro AUROC (%)")
    axis.legend(
        frameon=False,
        ncol=2,
        loc="lower right",
        handlelength=1.8,
        columnspacing=1.0,
    )

    if not args.no_inset:
        inset = axis.inset_axes([0.49, 0.44, 0.47, 0.35])
        draw_curves(inset, values, labels=False)
        inset.set_ylim(97.0, 100.05)
        inset.set_yticks([97, 98, 99, 100])
        inset.set_title("Near-ceiling region", fontsize=6.8, pad=1.5)
        inset.tick_params(labelsize=5.8, pad=1)
        for spine in inset.spines.values():
            spine.set_linewidth(0.65)

    fig.tight_layout(pad=0.45)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extended = bool(args.additional_ablation_summary)
    if args.min_budget is not None:
        stem_name = f"iad_training_sample_ablation_from_n{args.min_budget}"
        if args.no_inset:
            stem_name += "_no_inset"
        stem_name += "_preview"
    else:
        stem_name = (
            "iad_training_sample_ablation_extended_no_inset_preview"
            if extended and args.no_inset
            else "iad_training_sample_ablation_extended_preview"
            if extended
            else "iad_training_sample_ablation_no_inset_preview"
            if args.no_inset
            else "iad_training_sample_ablation_preview"
        )
    stem = args.output_dir / stem_name
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
