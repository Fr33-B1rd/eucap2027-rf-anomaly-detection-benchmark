#!/usr/bin/env python
"""Plot one representative normal and four IAD anomaly spectrograms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT.parent / "Datasets" / "IAD",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=ROOT / "configs" / "iad_formal_heatmap_selection.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "figure_previews",
    )
    parser.add_argument(
        "--match-normal-background",
        action="store_true",
        help=(
            "Replace the typical normal with the labeled-normal test sample "
            "having the smallest anomaly-robust background distance to the "
            "four displayed anomaly samples."
        ),
    )
    parser.add_argument(
        "--match-anomalies-to-normal",
        action="store_true",
        help=(
            "Keep the representative normal and select one labeled anomaly "
            "per type by anomaly-robust background distance to that normal."
        ),
    )
    return parser.parse_args()


def select_background_matched_normal(
    normal_images: np.ndarray,
    anomaly_images: np.ndarray,
    *,
    retained_fraction: float = 0.80,
) -> tuple[int, float]:
    """Select a normal by trimmed pixel distance to the anomaly backgrounds.

    The largest pairwise differences are excluded so that the injected signal
    regions have limited influence on the background comparison.
    """

    pixels = int(normal_images.shape[-2] * normal_images.shape[-1])
    retained = max(1, int(round(retained_fraction * pixels)))
    distances = np.zeros(normal_images.shape[0], dtype=np.float64)
    for anomaly in anomaly_images:
        difference = np.abs(normal_images - anomaly[None, ...]).reshape(
            normal_images.shape[0], -1
        )
        distances += np.partition(difference, retained - 1, axis=1)[
            :, :retained
        ].mean(axis=1)
    distances /= anomaly_images.shape[0]
    position = int(np.argmin(distances))
    return position, float(distances[position])


def select_background_matched_candidate(
    reference: np.ndarray,
    candidates: np.ndarray,
    *,
    retained_fraction: float = 0.80,
) -> tuple[int, float]:
    pixels = int(reference.shape[-2] * reference.shape[-1])
    retained = max(1, int(round(retained_fraction * pixels)))
    difference = np.abs(candidates - reference[None, ...]).reshape(
        candidates.shape[0], -1
    )
    distances = np.partition(difference, retained - 1, axis=1)[
        :, :retained
    ].mean(axis=1)
    position = int(np.argmin(distances))
    return position, float(distances[position])


def main() -> None:
    args = parse_args()

    from dataio.iad.pickle_io import discover_iad_sources, safe_load_iad_pickle

    sources = {
        item.modulation: item.path for item in discover_iad_sources(args.dataset_dir)
    }
    bundles = {
        name: safe_load_iad_pickle(path, modulation=name)
        for name, path in sources.items()
    }
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    requested = ["Typical normal", "16QAM", "CHIRP", "GMSK", "QPSK"]
    by_label = {str(item["label"]): item for item in selection["samples"]}

    images: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for label in requested:
        item = by_label[label]
        source = str(item["source"]).upper()
        index = int(item["source_index"])
        expected_label = int(item["expected_label"])
        bundle = bundles[source]
        actual_label = int(bundle.test_labels[index])
        if actual_label != expected_label:
            raise RuntimeError(
                f"label mismatch for {label}: expected {expected_label}, "
                f"found {actual_label}"
            )
        image = bundle.test_data[index, 0].float().numpy() * 10.0
        images.append(image)
        metadata.append(
            {
                "display_label": "Normal" if label == "Typical normal" else label,
                "source": source,
                "source_index": index,
                "label_value": actual_label,
                "selection_basis": (
                    "median consensus normal rank"
                    if label == "Typical normal"
                    else "median consensus anomaly rank"
                ),
            }
        )

    background_match: dict[str, Any] | None = None
    if args.match_normal_background and args.match_anomalies_to_normal:
        raise ValueError("choose only one background matching mode")
    if args.match_normal_background:
        normal_bundle = bundles["CHIRP"]
        normal_indices = np.flatnonzero(normal_bundle.test_labels == 0).astype(
            np.int64
        )
        normal_images = (
            normal_bundle.test_data[normal_indices, 0].float().numpy() * 10.0
        )
        anomaly_images = np.stack(images[1:])
        position, distance = select_background_matched_normal(
            normal_images, anomaly_images
        )
        source_index = int(normal_indices[position])
        images[0] = normal_images[position]
        metadata[0] = {
            "display_label": "Normal",
            "source": "CHIRP",
            "source_index": source_index,
            "label_value": 0,
            "selection_basis": (
                "minimum mean trimmed pixel distance to the four displayed "
                "anomalies; largest 20% pairwise differences excluded"
            ),
        }
        background_match = {
            "candidate_count": int(normal_indices.size),
            "retained_pixel_fraction": 0.80,
            "selected_mean_trimmed_distance": distance,
        }
    elif args.match_anomalies_to_normal:
        matches: list[dict[str, Any]] = []
        reference = images[0]
        for position, condition in enumerate(("16QAM", "CHIRP", "GMSK", "QPSK"), start=1):
            bundle = bundles[condition]
            anomaly_indices = np.flatnonzero(bundle.test_labels == 1).astype(
                np.int64
            )
            candidates = (
                bundle.test_data[anomaly_indices, 0].float().numpy() * 10.0
            )
            candidate_position, distance = select_background_matched_candidate(
                reference, candidates
            )
            source_index = int(anomaly_indices[candidate_position])
            images[position] = candidates[candidate_position]
            metadata[position] = {
                "display_label": condition,
                "source": condition,
                "source_index": source_index,
                "label_value": 1,
                "selection_basis": (
                    "minimum trimmed pixel distance to the displayed normal; "
                    "largest 20% pairwise differences excluded"
                ),
            }
            matches.append(
                {
                    "condition": condition,
                    "candidate_count": int(anomaly_indices.size),
                    "source_index": source_index,
                    "trimmed_distance": distance,
                }
            )
        background_match = {
            "mode": "anomalies matched to representative normal",
            "retained_pixel_fraction": 0.80,
            "matches": matches,
        }

    stacked = np.stack(images)
    display_min, display_max = np.quantile(stacked, [0.01, 0.995])
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 5, figsize=(7.15, 1.62))
    for axis, image, item in zip(axes, images, metadata):
        axis.imshow(
            image,
            cmap="gray",
            vmin=float(display_min),
            vmax=float(display_max),
            origin="lower",
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(item["display_label"], pad=3.0)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.55)

    figure.supxlabel("Frequency", fontsize=8, y=0.015)
    figure.supylabel("Time", fontsize=8, x=0.012)
    figure.subplots_adjust(left=0.045, right=0.997, bottom=0.16, top=0.84, wspace=0.08)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / (
        "iad_normal_four_anomalies_1x5_shared_background"
        if args.match_anomalies_to_normal
        else
        "iad_normal_four_anomalies_1x5_background_matched"
        if args.match_normal_background
        else "iad_normal_four_anomalies_1x5"
    )
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    figure.savefig(
        stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    plt.close(figure)

    manifest = {
        "figure": "one normal and four IAD anomaly spectrograms",
        "selection_version": selection["selection_version"],
        "selection_basis": selection["selection_basis"],
        "display": {
            "input_scale": "stored spectrogram multiplied by 10",
            "shared_vmin_quantile": 0.01,
            "shared_vmax_quantile": 0.995,
            "shared_vmin": float(display_min),
            "shared_vmax": float(display_max),
            "origin": "lower",
            "horizontal_axis": "frequency",
            "vertical_axis": "time",
        },
        "samples": metadata,
        "background_match": background_match,
    }
    stem.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
