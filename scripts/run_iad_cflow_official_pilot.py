#!/usr/bin/env python
"""Timing and compatibility pilot for the official CFLOW-AD architecture.

The decoder and training loop follow gudovskiy/cflow-ad.  This is explicitly
not a formal result: input 64 requires image batch 64 so that the layer4
2x2 grid supplies one complete official-size (N=256) fiber batch.
"""

from __future__ import annotations

import argparse
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

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataio.iad import discover_iad_sources, safe_load_iad_pickle  # noqa: E402
from iad_detection.data import (  # noqa: E402
    IADTensorDataset,
    IndexedLabelDataset,
    build_normal_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/iad_cflow_official64_pilot.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "outputs/iad_cflow_official64_pilot",
    )
    parser.add_argument("--device", default="cuda")
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


def positional_encoding_2d(
    dimension: int, height: int, width: int, device: torch.device
) -> torch.Tensor:
    if dimension % 4:
        raise ValueError("condition vector dimension must be divisible by four")
    encoding = torch.zeros(dimension, height, width, device=device)
    half = dimension // 2
    scale = torch.exp(
        torch.arange(0.0, half, 2, device=device) * -(math.log(1e4) / half)
    )
    position_w = torch.arange(width, device=device).unsqueeze(1)
    position_h = torch.arange(height, device=device).unsqueeze(1)
    encoding[0:half:2] = (
        torch.sin(position_w * scale).T.unsqueeze(1).repeat(1, height, 1)
    )
    encoding[1:half:2] = (
        torch.cos(position_w * scale).T.unsqueeze(1).repeat(1, height, 1)
    )
    encoding[half::2] = (
        torch.sin(position_h * scale).T.unsqueeze(2).repeat(1, 1, width)
    )
    encoding[half + 1 :: 2] = (
        torch.cos(position_h * scale).T.unsqueeze(2).repeat(1, 1, width)
    )
    return encoding


def subnet_fc(dims_in: int, dims_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dims_in, 2 * dims_in),
        nn.ReLU(),
        nn.Linear(2 * dims_in, dims_out),
    )


def make_decoder(
    feature_dim: int,
    *,
    condition_dim: int,
    coupling_blocks: int,
    clamp_alpha: float,
) -> Ff.SequenceINN:
    decoder = Ff.SequenceINN(feature_dim)
    for _ in range(coupling_blocks):
        decoder.append(
            Fm.AllInOneBlock,
            cond=0,
            cond_shape=(condition_dim,),
            subnet_constructor=subnet_fc,
            affine_clamping=clamp_alpha,
            global_affine_type="SOFTPLUS",
            permute_soft=True,
        )
    return decoder


class OfficialWideResNetFeatures(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return [layer2, layer3, layer4]


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    set_seed(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    sources = {
        source.modulation: source
        for source in discover_iad_sources(config["dataset_dir"])
    }
    source = sources[str(config["modulation"]).upper()]
    bundle = safe_load_iad_pickle(source.path)
    split = build_normal_split(
        bundle.train_data,
        calibration_fraction=float(config["calibration_fraction"]),
        seed=seed,
    )
    dataset = IndexedLabelDataset(
        IADTensorDataset(
            bundle.train_data,
            split.fit_indices,
            profile="imagenet_x10",
            input_size=int(config["input_size"]),
        )
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(config["workers"]),
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )

    encoder = OfficialWideResNetFeatures().to(device).eval()
    feature_dims = [512, 1024, 2048]
    decoders = nn.ModuleList(
        [
            make_decoder(
                dimension,
                condition_dim=int(config["condition_vector_dim"]),
                coupling_blocks=int(config["coupling_blocks"]),
                clamp_alpha=float(config["clamp_alpha"]),
            )
            for dimension in feature_dims
        ]
    ).to(device)
    optimizer = torch.optim.Adam(decoders.parameters(), lr=float(config["lr"]))
    log_sigmoid = nn.LogSigmoid()
    fiber_size = int(config["fiber_batch_size"])
    optimizer_steps = 0
    loss_sum = 0.0
    loss_count = 0
    observed_grids: list[list[int]] | None = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    total_passes = int(config["meta_epochs"]) * int(config["sub_epochs"])
    total_batches = total_passes * len(loader)
    completed_batches = 0

    for _ in range(int(config["meta_epochs"])):
        for _ in range(int(config["sub_epochs"])):
            for images, _, _ in loader:
                images = images.to(device, non_blocking=True)
                with torch.no_grad():
                    features = encoder(images)
                if observed_grids is None:
                    observed_grids = [
                        [int(value) for value in feature.shape[-2:]]
                        for feature in features
                    ]
                for decoder, feature in zip(decoders, features, strict=True):
                    batch, channels, height, width = feature.shape
                    patch_count = batch * height * width
                    if patch_count < fiber_size:
                        raise RuntimeError(
                            f"official fiber size {fiber_size} exceeds layer patch "
                            f"count {patch_count} for grid {height}x{width}; increase "
                            "image batch size or reduce fiber size"
                        )
                    position = positional_encoding_2d(
                        int(config["condition_vector_dim"]), height, width, device
                    )
                    condition = (
                        position.unsqueeze(0)
                        .repeat(batch, 1, 1, 1)
                        .reshape(batch, int(config["condition_vector_dim"]), -1)
                        .transpose(1, 2)
                        .reshape(patch_count, int(config["condition_vector_dim"]))
                    )
                    values = (
                        feature.reshape(batch, channels, -1)
                        .transpose(1, 2)
                        .reshape(patch_count, channels)
                    )
                    permutation = torch.randperm(patch_count, device=device)
                    complete_fibers = patch_count // fiber_size
                    for fiber_index in range(complete_fibers):
                        indices = permutation[
                            fiber_index * fiber_size : (fiber_index + 1) * fiber_size
                        ]
                        latent, log_jacobian = decoder(
                            values[indices], [condition[indices]]
                        )
                        log_probability = (
                            -0.5 * channels * math.log(2.0 * math.pi)
                            - 0.5 * latent.square().sum(1)
                            + log_jacobian
                        ) / channels
                        loss = -log_sigmoid(log_probability).mean()
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        optimizer_steps += 1
                        loss_sum += float(loss.detach())
                        loss_count += 1
                completed_batches += 1
                if completed_batches == 1 or completed_batches % 10 == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"batch {completed_batches}/{total_batches}, "
                        f"optimizer_steps={optimizer_steps}, elapsed={elapsed:.1f}s",
                        flush=True,
                    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    fit_seconds = time.perf_counter() - started

    # One no-gradient batch verifies that all three trained decoders are finite.
    images, _, _ = next(iter(loader))
    images = images.to(device, non_blocking=True)
    finite_probe = True
    with torch.no_grad():
        for decoder, feature in zip(decoders, encoder(images), strict=True):
            batch, channels, height, width = feature.shape
            values = feature.permute(0, 2, 3, 1).reshape(-1, channels)
            position = positional_encoding_2d(
                int(config["condition_vector_dim"]), height, width, device
            )
            condition = (
                position.permute(1, 2, 0)
                .reshape(-1, int(config["condition_vector_dim"]))
                .repeat(batch, 1)
            )
            latent, log_jacobian = decoder(values[:fiber_size], [condition[:fiber_size]])
            finite_probe = finite_probe and bool(
                torch.isfinite(latent).all() and torch.isfinite(log_jacobian).all()
            )

    result: dict[str, Any] = {
        "protocol_version": config["protocol_version"],
        "diagnostic_only": True,
        "official_source": "https://github.com/gudovskiy/cflow-ad",
        "adaptation": "batch_size_64_for_input64_layer4_fiber_compatibility",
        "modulation": bundle.modulation,
        "seed": seed,
        "fit_samples": int(len(dataset)),
        "loader_batches_per_sub_epoch": int(len(loader)),
        "feature_dims": feature_dims,
        "feature_grids": observed_grids,
        "optimizer_steps": optimizer_steps,
        "mean_train_loss": loss_sum / max(loss_count, 1),
        "finite_decoder_probe": finite_probe,
        "fit_seconds": fit_seconds,
        "estimated_default_25x8_seconds": fit_seconds * 25 * 8,
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else None
        ),
        "config": config,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "git_revision": git_revision(),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "pilot.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
