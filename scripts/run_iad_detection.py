#!/usr/bin/env python
"""Run the locked, per-modulation IAD anomaly-detection benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataio.iad import discover_iad_sources, safe_load_iad_pickle  # noqa: E402
from iad_detection.data import (  # noqa: E402
    IADTensorDataset,
    IndexedLabelDataset,
    build_normal_split,
)
from iad_detection.metrics import evaluate_detection  # noqa: E402
from iad_detection.methods import build_detector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs/iad_detection_v2.json"
    )
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--modulations", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=REPOSITORY_ROOT / "outputs/iad_detection_v2"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke-only wiring check on small subsets and one epoch",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed per-run JSON files and continue the remaining matrix",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def loader(
    dataset: IADTensorDataset,
    batch_size: int,
    workers: int,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        IndexedLabelDataset(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def smoke_indices(labels: np.ndarray) -> np.ndarray:
    normal = np.flatnonzero(labels == 0)[:16]
    anomaly = np.flatnonzero(labels == 1)[:16]
    return np.sort(np.concatenate((normal, anomaly))).astype(np.int64)


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute pre-registered per-seed modulation macro averages."""
    per_seed: list[dict[str, Any]] = []
    keys = sorted({(item["method"], item["seed"]) for item in results})
    for method, seed in keys:
        rows = [
            item
            for item in results
            if item["method"] == method and item["seed"] == seed
        ]
        per_seed.append(
            {
                "method": method,
                "seed": seed,
                "modulations": sorted(item["modulation"] for item in rows),
                "macro_auroc": float(
                    np.mean([item["metrics"]["auroc"] for item in rows])
                ),
                "macro_auprc": float(
                    np.mean([item["metrics"]["auprc"] for item in rows])
                ),
                "macro_test_tpr": float(
                    np.mean([item["metrics"]["test_tpr"] for item in rows])
                ),
                "macro_test_fpr": float(
                    np.mean([item["metrics"]["test_fpr"] for item in rows])
                ),
            }
        )
    across_seeds: list[dict[str, Any]] = []
    for method in sorted({item["method"] for item in per_seed}):
        rows = [item for item in per_seed if item["method"] == method]
        across_seeds.append(
            {
                "method": method,
                "seeds": [item["seed"] for item in rows],
                "macro_auroc_mean": float(
                    np.mean([item["macro_auroc"] for item in rows])
                ),
                "macro_auroc_std": float(
                    np.std([item["macro_auroc"] for item in rows], ddof=1)
                )
                if len(rows) > 1
                else None,
                "macro_auprc_mean": float(
                    np.mean([item["macro_auprc"] for item in rows])
                ),
                "macro_auprc_std": float(
                    np.std([item["macro_auprc"] for item in rows], ddof=1)
                )
                if len(rows) > 1
                else None,
            }
        )
    return {"per_seed": per_seed, "across_seeds": across_seeds}


def run_one(
    *,
    method: str,
    method_config: dict[str, Any],
    bundle: Any,
    seed: int,
    protocol: dict[str, Any],
    device: torch.device,
    workers: int,
    smoke: bool,
) -> dict[str, Any]:
    set_seed(seed)
    split = build_normal_split(
        bundle.train_data,
        calibration_fraction=float(protocol["calibration_fraction"]),
        seed=seed,
    )
    fit_indices, calibration_indices = split.fit_indices, split.calibration_indices
    test_indices = np.arange(bundle.test_data.shape[0], dtype=np.int64)
    config = dict(method_config)
    config["seed"] = seed
    if smoke:
        fit_indices = fit_indices[:32]
        calibration_indices = calibration_indices[:16]
        test_indices = smoke_indices(bundle.test_labels)
        if "epochs" in config:
            config["epochs"] = 1
        if "max_steps" in config:
            config["max_steps"] = 2
        for key, value in (
            ("candidate_pool_size", 256),
            ("max_memory_bank_size", 64),
            ("sample_patches_per_image", 4),
            ("patches_per_image", 4),
            ("patches_per_batch", 256),
            ("bootstrap_samples", 20),
        ):
            if key in config or key == "bootstrap_samples":
                config[key] = value

    detector = build_detector(method, config, device)
    fit_dataset = IADTensorDataset(
        bundle.train_data,
        fit_indices,
        profile=detector.preprocessing_profile,
        input_size=detector.input_size,
    )
    calibration_dataset = IADTensorDataset(
        bundle.train_data,
        calibration_indices,
        profile=detector.preprocessing_profile,
        input_size=detector.input_size,
    )
    test_dataset = IADTensorDataset(
        bundle.test_data,
        test_indices,
        profile=detector.preprocessing_profile,
        input_size=detector.input_size,
    )
    batch_size = int(config.get("batch_size", 32))
    fit_loader = loader(fit_dataset, batch_size, workers, shuffle=True, seed=seed)
    calibration_loader = loader(
        calibration_dataset, batch_size, workers, shuffle=False, seed=seed
    )
    test_loader = loader(test_dataset, batch_size, workers, shuffle=False, seed=seed)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    fit_info = detector.fit(fit_loader, calibration_loader)
    fit_seconds = time.perf_counter() - started
    calibration_scores = detector.score(calibration_loader)
    inference_started = time.perf_counter()
    test_scores = detector.score(test_loader)
    inference_seconds = time.perf_counter() - inference_started
    # Test labels are first exposed here, after the detector and scores are fixed.
    final_labels = bundle.test_labels[test_indices]
    metrics = evaluate_detection(
        calibration_scores,
        final_labels,
        test_scores,
        threshold_quantile=float(protocol["threshold_quantile"]),
        bootstrap_samples=20 if smoke else int(protocol["bootstrap_samples"]),
        bootstrap_seed=seed,
    )
    return {
        "protocol_version": protocol["protocol_version"],
        "publication_eligible": not smoke,
        "method": method,
        "implementation_variant": detector.implementation_variant,
        "modulation": bundle.modulation,
        "seed": seed,
        "preprocessing_profile": detector.preprocessing_profile,
        "input_size": detector.input_size,
        "fit_samples": int(len(fit_dataset)),
        "calibration_samples": int(len(calibration_dataset)),
        "test_samples": int(len(test_dataset)),
        "normal_hash_overlap": len(
            set(split.fit_hashes).intersection(split.calibration_hashes)
        ),
        "config": config,
        "fit": fit_info,
        "metrics": metrics,
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "milliseconds_per_test_image": 1000.0 * inference_seconds / len(test_dataset),
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else None
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "git_revision": git_revision(),
        },
    }


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    methods = args.methods or list(protocol["methods"])
    seeds = args.seeds or protocol["seeds"]
    sources = discover_iad_sources(protocol["dataset_dir"])
    wanted = {value.upper() for value in args.modulations} if args.modulations else None
    sources = [
        source for source in sources if wanted is None or source.modulation in wanted
    ]
    unknown = set(methods).difference(protocol["methods"])
    if unknown:
        raise SystemExit(f"methods absent from locked config: {sorted(unknown)}")
    if not sources:
        raise SystemExit("no requested modulation found")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for source in sources:
        bundle = safe_load_iad_pickle(source.path)
        for method in methods:
            for seed in seeds:
                name = f"{method}__{bundle.modulation}__seed{seed}{'__smoke' if args.smoke else ''}.json"
                run_path = args.output_dir / name
                if args.resume and run_path.exists():
                    previous = json.loads(run_path.read_text(encoding="utf-8"))
                    identity = (
                        previous.get("protocol_version"),
                        previous.get("method"),
                        previous.get("modulation"),
                        previous.get("seed"),
                        previous.get("publication_eligible"),
                    )
                    expected = (
                        protocol["protocol_version"],
                        method,
                        bundle.modulation,
                        seed,
                        not args.smoke,
                    )
                    if identity != expected:
                        raise RuntimeError(
                            f"resume file identity mismatch for {run_path}: "
                            f"found {identity}, expected {expected}"
                        )
                    results.append(previous)
                    print(
                        f"[resume] {method} {bundle.modulation} seed={seed}",
                        flush=True,
                    )
                    continue
                print(
                    f"[{method}] {bundle.modulation} seed={seed} smoke={args.smoke}",
                    flush=True,
                )
                result = run_one(
                    method=method,
                    method_config=protocol["methods"][method],
                    bundle=bundle,
                    seed=seed,
                    protocol=protocol,
                    device=device,
                    workers=args.workers,
                    smoke=args.smoke,
                )
                results.append(result)
                run_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(
                    f"  AUROC={result['metrics']['auroc']:.4f} AUPRC={result['metrics']['auprc']:.4f}",
                    flush=True,
                )

    summary_json = args.output_dir / (
        "summary_smoke.json" if args.smoke else "summary.json"
    )
    summary_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    aggregate_json = args.output_dir / (
        "aggregate_smoke.json" if args.smoke else "aggregate.json"
    )
    aggregate_json.write_text(
        json.dumps(aggregate_results(results), indent=2), encoding="utf-8"
    )
    summary_csv = args.output_dir / (
        "summary_smoke.csv" if args.smoke else "summary.csv"
    )
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "method",
            "implementation_variant",
            "modulation",
            "seed",
            "input_size",
            "publication_eligible",
            "auroc",
            "auprc",
            "test_tpr",
            "test_fpr",
            "fit_seconds",
            "milliseconds_per_test_image",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {key: result.get(key, result["metrics"].get(key)) for key in fields}
            )
    print(f"Wrote {summary_json}, {summary_csv}, and {aggregate_json}")


if __name__ == "__main__":
    main()
