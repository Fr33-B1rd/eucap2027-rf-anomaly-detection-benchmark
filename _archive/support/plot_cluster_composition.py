import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot cluster composition by modulation for selected clustering outputs"
    )
    parser.add_argument(
        "--base-dir",
        default="outputs/feature_mode_comparison_v1",
        help="Base directory containing mean/gram/mean_plus_gram best_config_artifacts",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["mean", "gram", "mean_plus_gram"],
        help="Feature modes to plot",
    )
    return parser.parse_args()


def plot_mode(base_dir: str, mode: str) -> str:
    assign_path = os.path.join(base_dir, mode, "best_config_artifacts", "cluster_assignments.csv")
    if not os.path.exists(assign_path):
        raise FileNotFoundError(assign_path)

    df = pd.read_csv(assign_path)
    clustered = df[df["cluster_label"] != -1].copy()
    noise = df[df["cluster_label"] == -1].copy()

    cluster_ids = sorted(clustered["cluster_label"].unique().tolist())
    dataset_order = ["16QAM", "CHIRP", "GMSK", "QPSK"]
    colors = {
        "16QAM": "#1f77b4",
        "CHIRP": "#ff7f0e",
        "GMSK": "#2ca02c",
        "QPSK": "#d62728",
    }

    pivot = (
        clustered.groupby(["cluster_label", "dataset_name"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=cluster_ids, columns=dataset_order, fill_value=0)
    )

    x = np.arange(len(cluster_ids))
    width = 0.72
    bottom = np.zeros(len(cluster_ids), dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    for dataset_name in dataset_order:
        values = pivot[dataset_name].to_numpy(dtype=float)
        ax.bar(
            x,
            values,
            width=width,
            bottom=bottom,
            label=dataset_name,
            color=colors[dataset_name],
            edgecolor="white",
            linewidth=0.8,
        )
        bottom += values

    total_counts = pivot.sum(axis=1).to_numpy()
    for xi, total in zip(x, total_counts):
        ax.text(xi, total + 8, str(int(total)), ha="center", va="bottom", fontsize=10)

    noise_counts = (
        noise["dataset_name"].value_counts().reindex(dataset_order, fill_value=0).astype(int)
    )
    noise_text = "Noise samples: " + str(int(len(noise))) + "\n" + ", ".join(
        f"{name}={int(noise_counts[name])}" for name in dataset_order
    )

    ax.text(
        0.98,
        0.98,
        noise_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.92, "edgecolor": "#444444"},
    )

    ax.set_xticks(x, [f"Cluster {cluster_id}" for cluster_id in cluster_ids])
    ax.set_ylabel("Sample Count")
    ax.set_xlabel("Cluster")
    ax.set_title(f"Cluster Composition by Modulation: {mode.replace('_plus_', ' + ').upper()}")
    ax.legend(title="Modulation", loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()

    output_path = os.path.join(base_dir, mode, "best_config_artifacts", "cluster_composition_by_modulation.png")
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    args = parse_args()
    for mode in args.modes:
        output_path = plot_mode(args.base_dir, mode)
        print(output_path, flush=True)


if __name__ == "__main__":
    main()
