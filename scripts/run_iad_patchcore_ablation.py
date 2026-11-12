#!/usr/bin/env python
"""Run controlled PatchCore ablations between the thesis and v2 protocols.

This is a diagnostic runner.  It reports image AUROC/AUPRC only and never uses
test labels for model construction, candidate selection, or threshold tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.random_projection import SparseRandomProjection
from torch.utils.data import DataLoader, Dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from core.patchcore import PatchCore  # noqa: E402
from core.utils import log  # noqa: E402
from dataio.iad import discover_iad_sources, safe_load_iad_pickle  # noqa: E402
from iad_detection.data import build_normal_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/iad_patchcore_ablation.json",
    )
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--modulations", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "outputs/iad_patchcore_ablation_20260803",
    )
    parser.add_argument("--resume", action="store_true")
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


class AblationDataset(Dataset):
    """IAD preprocessing with explicit legacy and v2 profiles."""

    def __init__(
        self,
        data: torch.Tensor,
        labels: np.ndarray,
        indices: np.ndarray,
        *,
        preprocessing: str,
        input_size: int,
        train_mean: float,
        train_std: float,
        clip_z: float = 5.0,
    ) -> None:
        self.data = data
        self.labels = np.asarray(labels, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.preprocessing = preprocessing
        self.input_size = int(input_size)
        self.train_mean = float(train_mean)
        self.train_std = float(train_std)
        self.clip_z = float(clip_z)
        if preprocessing not in {"legacy_global_zscore", "imagenet_x10"}:
            raise ValueError(f"unsupported preprocessing: {preprocessing}")

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        x = self.data[index].float()
        if self.preprocessing == "legacy_global_zscore":
            x = (x - self.train_mean) / self.train_std
            x = x.clamp(-self.clip_z, self.clip_z)
        else:
            x = x * 10.0
        if tuple(x.shape[-2:]) != (self.input_size, self.input_size):
            x = F.interpolate(
                x.unsqueeze(0),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        if self.preprocessing == "legacy_global_zscore":
            x = x.repeat(3, 1, 1)
        else:
            x = x.clamp(0.0, 1.0).repeat(3, 1, 1)
            mean = x.new_tensor((0.485, 0.456, 0.406))[:, None, None]
            std = x.new_tensor((0.229, 0.224, 0.225))[:, None, None]
            x = (x - mean) / std
        return x, int(self.labels[index]), index


class DiagnosticPatchCore(PatchCore):
    """PatchCore with explicit candidate and coreset semantics for ablations."""

    def __init__(
        self,
        *,
        candidate_mode: str,
        coreset_ratio_base: str,
        candidate_storage: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if candidate_mode not in {"global_uniform", "per_image"}:
            raise ValueError(f"unsupported candidate_mode: {candidate_mode}")
        if coreset_ratio_base not in {"total_patches", "candidate_pool"}:
            raise ValueError(f"unsupported coreset_ratio_base: {coreset_ratio_base}")
        if candidate_storage not in {"float16", "float32"}:
            raise ValueError(f"unsupported candidate_storage: {candidate_storage}")
        self.candidate_mode = candidate_mode
        self.coreset_ratio_base = coreset_ratio_base
        self.candidate_storage = candidate_storage
        self.candidate_count = 0

    def _dedicated_kcenter(
        self, candidate_embeddings: torch.Tensor, num_select: int
    ) -> torch.Tensor:
        candidate_embeddings = candidate_embeddings.to(self.device, non_blocking=True)
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        first_index = int(
            torch.randint(
                0,
                candidate_embeddings.shape[0],
                (1,),
                generator=generator,
                device=self.device,
            ).item()
        )
        selected = torch.empty(num_select, dtype=torch.long, device=self.device)
        selected[0] = first_index
        center = candidate_embeddings[first_index]
        min_distances = ((candidate_embeddings - center) ** 2).sum(dim=1)
        for position in range(1, num_select):
            next_index = torch.argmax(min_distances)
            selected[position] = next_index
            center = candidate_embeddings[next_index]
            distances = ((candidate_embeddings - center) ** 2).sum(dim=1)
            min_distances = torch.minimum(min_distances, distances)
            if position % 250 == 0 or position == num_select - 1:
                log(f"k-center progress: {position + 1}/{num_select}")
        return selected.cpu()

    def fit(self, train_loader: DataLoader) -> None:
        stored_dtype = torch.float16 if self.candidate_storage == "float16" else torch.float32
        candidate_batches: list[torch.Tensor] = []
        selected_global: np.ndarray | None = None
        selected_cursor = 0
        traversed_patches = 0
        total_batches = len(train_loader)

        with torch.no_grad():
            for batch_number, (images, _, _) in enumerate(train_loader, start=1):
                images = images.to(self.device, non_blocking=True)
                embeddings = self._compute_patch_embeddings(images)
                batch_size, patches_per_image, embedding_dim = embeddings.shape
                batch_patch_count = int(batch_size * patches_per_image)
                if selected_global is None and self.candidate_mode == "global_uniform":
                    total_patch_count = int(len(train_loader.dataset) * patches_per_image)
                    pool_size = min(self.candidate_pool_size, total_patch_count)
                    rng = np.random.default_rng(self.seed)
                    selected_global = np.sort(
                        rng.choice(total_patch_count, size=pool_size, replace=False)
                    )
                flat = embeddings.reshape(-1, embedding_dim)
                if self.candidate_mode == "global_uniform":
                    assert selected_global is not None
                    upper = traversed_patches + batch_patch_count
                    end_cursor = int(np.searchsorted(selected_global, upper, side="left"))
                    local = selected_global[selected_cursor:end_cursor] - traversed_patches
                    if local.size:
                        candidate_batches.append(
                            flat[torch.from_numpy(local).to(flat.device)].cpu().to(stored_dtype)
                        )
                    selected_cursor = end_cursor
                else:
                    sampled = self._sample_candidate_patches(embeddings)
                    candidate_batches.append(sampled.cpu().to(stored_dtype))
                traversed_patches += batch_patch_count
                if batch_number == 1 or batch_number % 40 == 0 or batch_number == total_batches:
                    log(f"Collected ablation candidates from batch {batch_number}/{total_batches}")

        self.total_train_patches = traversed_patches
        candidates = torch.cat(candidate_batches, dim=0).float()
        if self.candidate_mode == "per_image" and candidates.shape[0] > self.candidate_pool_size:
            generator = torch.Generator().manual_seed(self.seed)
            keep = torch.randperm(candidates.shape[0], generator=generator)[
                : self.candidate_pool_size
            ]
            candidates = candidates[keep]
        self.candidate_count = int(candidates.shape[0])

        ratio_count = (
            self.total_train_patches
            if self.coreset_ratio_base == "total_patches"
            else self.candidate_count
        )
        target_size = max(1, int(math.ceil(ratio_count * self.coreset_ratio)))
        target_size = min(target_size, self.max_memory_bank_size, self.candidate_count)
        if self.projection_dim > 0 and candidates.shape[1] > self.projection_dim:
            projector = SparseRandomProjection(
                n_components=self.projection_dim, random_state=self.seed
            )
            projected = torch.from_numpy(
                np.asarray(projector.fit_transform(candidates.numpy()), dtype=np.float32)
            )
        else:
            projected = candidates
        log(
            f"Ablation k-center: total={self.total_train_patches}, "
            f"candidates={self.candidate_count}, memory={target_size}"
        )
        selected = self._dedicated_kcenter(projected, target_size)
        self.memory_bank = candidates[selected].to(self.device, non_blocking=True)
        self.memory_bank_norms = (self.memory_bank**2).sum(dim=1)


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
    )


def write_summary(output_dir: Path, results: list[dict[str, Any]]) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    fields = [
        "variant",
        "modulation",
        "seed",
        "auroc",
        "auprc",
        "preprocessing",
        "input_size",
        "fit_samples",
        "patch_grid",
        "total_train_patches",
        "candidate_count",
        "memory_bank_size",
        "fit_seconds",
        "predict_seconds",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in fields})


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(protocol["seed"])
    variants = args.variants or list(protocol["variants"])
    missing = set(variants).difference(protocol["variants"])
    if missing:
        raise SystemExit(f"unknown variants: {sorted(missing)}")
    wanted = {item.upper() for item in args.modulations} if args.modulations else None
    sources = [
        source
        for source in discover_iad_sources(protocol["dataset_dir"])
        if wanted is None or source.modulation in wanted
    ]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for source in sources:
        bundle = safe_load_iad_pickle(source.path)
        train_mean = float(bundle.train_data.mean())
        train_std = float(bundle.train_data.std())
        split = build_normal_split(
            bundle.train_data,
            calibration_fraction=float(protocol["calibration_fraction"]),
            seed=seed,
        )
        for variant in variants:
            run_path = args.output_dir / f"{variant}__{bundle.modulation}__seed{seed}.json"
            if args.resume and run_path.exists():
                result = json.loads(run_path.read_text(encoding="utf-8"))
                results.append(result)
                print(f"[resume] {variant} {bundle.modulation}", flush=True)
                continue
            config = dict(protocol["variants"][variant])
            set_seed(seed)
            fit_indices = (
                np.arange(bundle.train_data.shape[0], dtype=np.int64)
                if float(config["fit_fraction"]) == 1.0
                else split.fit_indices
            )
            train_dataset = AblationDataset(
                bundle.train_data,
                bundle.train_labels,
                fit_indices,
                preprocessing=config["preprocessing"],
                input_size=int(config["input_size"]),
                train_mean=train_mean,
                train_std=train_std,
            )
            test_dataset = AblationDataset(
                bundle.test_data,
                bundle.test_labels,
                np.arange(bundle.test_data.shape[0], dtype=np.int64),
                preprocessing=config["preprocessing"],
                input_size=int(config["input_size"]),
                train_mean=train_mean,
                train_std=train_std,
            )
            train_loader = make_loader(
                train_dataset,
                batch_size=int(config["batch_size"]),
                workers=args.workers,
                shuffle=bool(config["loader_shuffle"]),
                seed=seed,
            )
            test_loader = make_loader(
                test_dataset,
                batch_size=int(config["batch_size"]),
                workers=args.workers,
                shuffle=False,
                seed=seed,
            )
            model = DiagnosticPatchCore(
                backbone="wide_resnet50_2",
                layers=["layer2", "layer3"],
                device=device,
                sample_patches_per_image=int(config["sample_patches_per_image"]),
                candidate_pool_size=int(config["candidate_pool_size"]),
                coreset_ratio=float(config["coreset_ratio"]),
                max_memory_bank_size=int(config["max_memory_bank_size"]),
                projection_dim=int(config["projection_dim"]),
                query_chunk_size=4096,
                seed=seed,
                pretrained=True,
                candidate_mode=config["candidate_mode"],
                coreset_ratio_base=config["coreset_ratio_base"],
                candidate_storage=config["candidate_storage"],
            )
            print(f"[{variant}] {bundle.modulation}", flush=True)
            started = time.perf_counter()
            model.fit(train_loader)
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            scores, labels, _, _ = model.predict(test_loader)
            predict_seconds = time.perf_counter() - started
            result = {
                "protocol_version": protocol["protocol_version"],
                "diagnostic_only": True,
                "variant": variant,
                "modulation": bundle.modulation,
                "seed": seed,
                "auroc": float(roc_auc_score(labels, scores)),
                "auprc": float(average_precision_score(labels, scores)),
                "preprocessing": config["preprocessing"],
                "input_size": int(config["input_size"]),
                "fit_samples": int(len(train_dataset)),
                "patch_grid": "x".join(str(value) for value in model.patch_grid_size),
                "total_train_patches": int(model.total_train_patches),
                "candidate_count": int(model.candidate_count),
                "memory_bank_size": int(model.memory_bank.shape[0]),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "train_mean": train_mean,
                "train_std": train_std,
                "config": config,
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "device": str(device),
                    "git_revision": git_revision(),
                },
            }
            run_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            results.append(result)
            write_summary(args.output_dir, results)
            print(
                f"  AUROC={result['auroc']:.6f} memory={result['memory_bank_size']}",
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_summary(args.output_dir, results)
    print(f"Wrote {args.output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
