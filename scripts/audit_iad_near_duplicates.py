#!/usr/bin/env python3
"""Audit within-modulation IAD train/test near duplicates.

The exact-hash audit remains the authoritative test for byte-identical samples.
This complementary screen searches every test image against every training image
from the same modulation using two transparent criteria:

* raw-pixel RMSE, which catches numerically near-identical copies; and
* centered cosine correlation, which catches the same spatial pattern after a
  possible global offset or gain change.

The released tensors are compared at their native 64x64 resolution.  No test
labels are used by either distance calculation; labels are retained only for
stratified reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataio.iad import discover_iad_sources, safe_load_iad_pickle  # noqa: E402


DEFAULT_RMSE_THRESHOLD = 1e-3
DEFAULT_CORRELATION_THRESHOLD = 0.99
QUANTILES = (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--rmse-threshold", type=float, default=DEFAULT_RMSE_THRESHOLD
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=DEFAULT_CORRELATION_THRESHOLD,
    )
    return parser


def nearest_train_matches(
    train: torch.Tensor,
    test: torch.Tensor,
    *,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return nearest raw-RMSE and maximum-correlation train matches."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if train.ndim < 2 or test.ndim != train.ndim:
        raise ValueError("train and test must have matching sample dimensions")
    if tuple(train.shape[1:]) != tuple(test.shape[1:]):
        raise ValueError("train and test sample shapes differ")
    if len(train) == 0 or len(test) == 0:
        raise ValueError("train and test must both be non-empty")

    train_flat = train.detach().cpu().to(torch.float64).reshape(len(train), -1)
    test_flat = test.detach().cpu().to(torch.float64).reshape(len(test), -1)
    feature_count = train_flat.shape[1]

    train_squared_norm = torch.sum(train_flat * train_flat, dim=1)
    train_centered = train_flat - torch.mean(train_flat, dim=1, keepdim=True)
    train_centered /= torch.clamp(
        torch.linalg.vector_norm(train_centered, dim=1, keepdim=True),
        min=1e-15,
    )

    best_rmse = torch.empty(len(test_flat), dtype=torch.float64)
    best_rmse_index = torch.empty(len(test_flat), dtype=torch.int64)
    best_correlation = torch.empty(len(test_flat), dtype=torch.float64)
    best_correlation_index = torch.empty(len(test_flat), dtype=torch.int64)

    for start in range(0, len(test_flat), block_size):
        block = test_flat[start : start + block_size]
        squared_distance = torch.clamp(
            torch.sum(block * block, dim=1)[:, None]
            + train_squared_norm[None, :]
            - 2.0 * block @ train_flat.T,
            min=0.0,
        )
        raw_minimum, raw_index = torch.min(squared_distance, dim=1)
        best_rmse[start : start + len(block)] = torch.sqrt(
            raw_minimum / feature_count
        )
        best_rmse_index[start : start + len(block)] = raw_index

        centered = block - torch.mean(block, dim=1, keepdim=True)
        centered /= torch.clamp(
            torch.linalg.vector_norm(centered, dim=1, keepdim=True),
            min=1e-15,
        )
        correlation, correlation_index = torch.max(centered @ train_centered.T, dim=1)
        best_correlation[start : start + len(block)] = correlation
        best_correlation_index[start : start + len(block)] = correlation_index

    return (
        best_rmse,
        best_rmse_index,
        best_correlation,
        best_correlation_index,
    )


def _quantile_map(values: torch.Tensor) -> dict[str, float]:
    probabilities = torch.tensor(QUANTILES, dtype=values.dtype)
    estimates = torch.quantile(values, probabilities).tolist()
    return {
        f"q{int(round(probability * 100)):02d}": float(estimate)
        for probability, estimate in zip(QUANTILES, estimates, strict=True)
    }


def _finite_float(value: float, *, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.threads <= 0:
        raise SystemExit("--threads must be positive")
    rmse_threshold = _finite_float(args.rmse_threshold, name="rmse threshold")
    correlation_threshold = _finite_float(
        args.correlation_threshold, name="correlation threshold"
    )
    if rmse_threshold < 0:
        raise SystemExit("--rmse-threshold must be non-negative")
    if not -1.0 <= correlation_threshold <= 1.0:
        raise SystemExit("--correlation-threshold must be in [-1, 1]")

    torch.set_num_threads(args.threads)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    modulation_summaries: list[dict[str, Any]] = []
    for source in discover_iad_sources(args.dataset_dir):
        bundle = safe_load_iad_pickle(source.path, modulation=source.modulation)
        rmse, rmse_index, correlation, correlation_index = nearest_train_matches(
            bundle.train_data,
            bundle.test_data,
            block_size=args.block_size,
        )
        labels = torch.as_tensor(np.asarray(bundle.test_labels), dtype=torch.int64)
        rmse_flag = rmse <= rmse_threshold
        correlation_flag = correlation >= correlation_threshold
        near_flag = rmse_flag | correlation_flag

        strata: dict[str, Any] = {}
        for label, label_name in ((0, "normal"), (1, "anomaly")):
            mask = labels == label
            strata[label_name] = {
                "count": int(mask.sum()),
                "nearest_raw_rmse": _quantile_map(rmse[mask]),
                "maximum_centered_correlation": _quantile_map(correlation[mask]),
                "rmse_flag_count": int((rmse_flag & mask).sum()),
                "correlation_flag_count": int((correlation_flag & mask).sum()),
                "near_duplicate_flag_count": int((near_flag & mask).sum()),
            }

        modulation_summaries.append(
            {
                "modulation": bundle.modulation,
                "train_count": int(len(bundle.train_data)),
                "test_count": int(len(bundle.test_data)),
                "near_duplicate_flag_count": int(near_flag.sum()),
                "rmse_flag_count": int(rmse_flag.sum()),
                "correlation_flag_count": int(correlation_flag.sum()),
                "strata": strata,
            }
        )

        for test_index in range(len(bundle.test_data)):
            rows.append(
                {
                    "modulation": bundle.modulation,
                    "test_index": test_index,
                    "test_label": int(labels[test_index]),
                    "nearest_rmse_train_index": int(rmse_index[test_index]),
                    "nearest_raw_rmse": float(rmse[test_index]),
                    "maximum_correlation_train_index": int(
                        correlation_index[test_index]
                    ),
                    "maximum_centered_correlation": float(correlation[test_index]),
                    "rmse_flag": int(rmse_flag[test_index]),
                    "correlation_flag": int(correlation_flag[test_index]),
                    "near_duplicate_flag": int(near_flag[test_index]),
                }
            )

        print(
            f"{bundle.modulation}: {int(near_flag.sum())}/{len(bundle.test_data)} "
            "test samples flagged",
            flush=True,
        )

    csv_path = output_dir / "iad_near_duplicate_nearest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "protocol_version": "iad-near-duplicate-audit-v1",
        "scope": "exhaustive train-test comparison within each modulation",
        "sample_representation": "released native 64x64 float64 tensors",
        "criteria": {
            "near_duplicate_if": "raw_rmse <= threshold OR centered_correlation >= threshold",
            "raw_rmse_threshold": rmse_threshold,
            "centered_correlation_threshold": correlation_threshold,
        },
        "total_test_samples_compared": len(rows),
        "total_near_duplicate_flag_count": sum(
            int(row["near_duplicate_flag"]) for row in rows
        ),
        "modulations": modulation_summaries,
        "nearest_pairs_csv": csv_path.name,
    }
    summary_path = output_dir / "iad_near_duplicate_audit.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
