"""CFLOW-AD's official WRN50-2/FrEIA architecture adapted to IAD.

The encoder, conditional normalizing-flow heads, likelihood objective, fiber
batching, cosine schedule, and two-epoch warm-up follow gudovskiy/cflow-ad
(commit b2ebf9e673a0aa46992a3b18367ec066a57bba89).  Dataset handling and final
evaluation deliberately follow the leakage-free IAD protocol: test labels are
not used for model selection, and normal calibration data fixes score scaling
and the operating threshold.

CFLOW-derived portions: Copyright (c) 2021 Panasonic AI Lab of Panasonic
Corporation of North America, BSD-3-Clause; see berzelius/CFLOW_LICENSE.txt.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

try:
    import FrEIA.framework as Ff
    import FrEIA.modules as Fm
except ModuleNotFoundError:  # Optional until an official CFLOW model is built.
    Ff = None
    Fm = None


OFFICIAL_REPOSITORY = "https://github.com/gudovskiy/cflow-ad"
OFFICIAL_COMMIT = "b2ebf9e673a0aa46992a3b18367ec066a57bba89"
GAUSSIAN_LOG_CONSTANT = -0.9189385332046727


def positional_encoding_2d(
    dimension: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the official sinusoidal D x H x W conditional encoding."""

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
    if Ff is None or Fm is None:
        raise ModuleNotFoundError(
            "Official CFLOW requires FrEIA; install the repository requirements "
            "or run `python -m pip install FrEIA==0.2`."
        )
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


class WideResNet50Features(nn.Module):
    """Frozen ImageNet WRN50-2 layer2/layer3/layer4 feature extractor."""

    feature_dims = (512, 1024, 2048)

    def __init__(self, pool_layers: int = 3) -> None:
        super().__init__()
        if pool_layers not in {1, 2, 3}:
            raise ValueError("pool_layers must be one of 1, 2, or 3")
        base = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.pool_layers = pool_layers
        for parameter in self.parameters():
            parameter.requires_grad = False

    @property
    def output_dims(self) -> tuple[int, ...]:
        return self.feature_dims[-self.pool_layers :]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return [layer2, layer3, layer4][-self.pool_layers :]


@dataclass(frozen=True)
class CFLOWTrainStats:
    mean_loss: float
    optimizer_steps: int
    samples_seen: int
    seconds: float
    learning_rate: float


class OfficialCFLOW:
    """Train and score the official conditional-flow architecture."""

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = dict(config)
        self.device = device
        self.condition_dim = int(config["condition_vector_dim"])
        self.fiber_size = int(config["fiber_batch_size"])
        self.meta_epochs = int(config["meta_epochs"])
        self.sub_epochs = int(config["sub_epochs"])
        self.base_lr = float(config["lr"])
        self.lr_decay_rate = float(config.get("lr_decay_rate", 0.1))
        self.lr_warm_epochs = int(config.get("lr_warm_epochs", 2))

        self.encoder = WideResNet50Features(int(config["pool_layers"])).to(device)
        self.encoder.eval()
        self.feature_dims = self.encoder.output_dims
        self.decoders = nn.ModuleList(
            [
                make_decoder(
                    dimension,
                    condition_dim=self.condition_dim,
                    coupling_blocks=int(config["coupling_blocks"]),
                    clamp_alpha=float(config["clamp_alpha"]),
                )
                for dimension in self.feature_dims
            ]
        ).to(device)
        self.optimizer = torch.optim.Adam(self.decoders.parameters(), lr=self.base_lr)
        self.log_sigmoid = nn.LogSigmoid()

    def _meta_learning_rate(self, epoch: int) -> float:
        eta_min = self.base_lr * self.lr_decay_rate**3
        return eta_min + (self.base_lr - eta_min) * (
            1.0 + math.cos(math.pi * epoch / self.meta_epochs)
        ) / 2.0

    def _warmup_target(self) -> float:
        eta_min = self.base_lr * self.lr_decay_rate**3
        return eta_min + (self.base_lr - eta_min) * (
            1.0 + math.cos(math.pi * self.lr_warm_epochs / self.meta_epochs)
        ) / 2.0

    def _set_learning_rate(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = value

    def _batch_learning_rate(
        self,
        *,
        epoch: int,
        batch_in_meta_epoch: int,
        total_batches: int,
    ) -> float:
        if epoch >= self.lr_warm_epochs:
            return float(self.optimizer.param_groups[0]["lr"])
        progress = (batch_in_meta_epoch + epoch * total_batches) / (
            self.lr_warm_epochs * total_batches
        )
        value = self.base_lr / 10.0 + progress * (
            self._warmup_target() - self.base_lr / 10.0
        )
        self._set_learning_rate(value)
        return value

    def train_meta_epoch(self, epoch: int, loader: DataLoader) -> CFLOWTrainStats:
        """Run the official number of sub-epochs for one meta epoch."""

        self.encoder.eval()
        self.decoders.train()
        self._set_learning_rate(self._meta_learning_rate(epoch))
        loss_sum = 0.0
        loss_count = 0
        optimizer_steps = 0
        samples_seen = 0
        total_batches = len(loader) * self.sub_epochs
        started = time.perf_counter()
        last_lr = float(self.optimizer.param_groups[0]["lr"])

        for sub_epoch in range(self.sub_epochs):
            sub_loss_sum = 0.0
            sub_loss_count = 0
            for batch_index, (images, _, _) in enumerate(loader):
                batch_in_meta = batch_index + sub_epoch * len(loader)
                last_lr = self._batch_learning_rate(
                    epoch=epoch,
                    batch_in_meta_epoch=batch_in_meta,
                    total_batches=total_batches,
                )
                images = images.to(self.device, non_blocking=True)
                samples_seen += int(images.shape[0])
                with torch.no_grad():
                    features = self.encoder(images)
                for decoder, feature in zip(self.decoders, features, strict=True):
                    batch, channels, height, width = feature.shape
                    patch_count = batch * height * width
                    complete_fibers = patch_count // self.fiber_size
                    if complete_fibers <= 0:
                        raise RuntimeError(
                            f"fiber size {self.fiber_size} exceeds layer patch count "
                            f"{patch_count} for batch={batch}, grid={height}x{width}"
                        )
                    position = positional_encoding_2d(
                        self.condition_dim, height, width, self.device
                    )
                    condition = (
                        position.unsqueeze(0)
                        .repeat(batch, 1, 1, 1)
                        .reshape(batch, self.condition_dim, -1)
                        .transpose(1, 2)
                        .reshape(patch_count, self.condition_dim)
                    )
                    values = (
                        feature.reshape(batch, channels, -1)
                        .transpose(1, 2)
                        .reshape(patch_count, channels)
                    )
                    permutation = torch.randperm(patch_count, device=self.device)
                    for fiber_index in range(complete_fibers):
                        indices = permutation[
                            fiber_index
                            * self.fiber_size : (fiber_index + 1)
                            * self.fiber_size
                        ]
                        latent, log_jacobian = decoder(
                            values[indices], [condition[indices]]
                        )
                        log_probability = (
                            channels * GAUSSIAN_LOG_CONSTANT
                            - 0.5 * latent.square().sum(1)
                            + log_jacobian
                        ) / channels
                        loss = -self.log_sigmoid(log_probability).mean()
                        self.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        self.optimizer.step()
                        value = float(loss.detach())
                        loss_sum += value * self.fiber_size
                        loss_count += self.fiber_size
                        sub_loss_sum += value * self.fiber_size
                        sub_loss_count += self.fiber_size
                        optimizer_steps += 1
            print(
                f"meta={epoch + 1}/{self.meta_epochs} "
                f"sub={sub_epoch + 1}/{self.sub_epochs} "
                f"loss={sub_loss_sum / max(sub_loss_count, 1):.6f} "
                f"lr={last_lr:.8f}",
                flush=True,
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return CFLOWTrainStats(
            mean_loss=loss_sum / max(loss_count, 1),
            optimizer_steps=optimizer_steps,
            samples_seen=samples_seen,
            seconds=time.perf_counter() - started,
            learning_rate=last_lr,
        )

    def collect_log_probability_maps(
        self, loader: DataLoader
    ) -> tuple[list[torch.Tensor], list[list[int]]]:
        """Collect one raw per-patch log-likelihood map per feature scale."""

        self.encoder.eval()
        self.decoders.eval()
        batches: list[list[torch.Tensor]] = [list() for _ in self.feature_dims]
        grids: list[list[int]] | None = None
        with torch.inference_mode():
            for images, _, _ in loader:
                images = images.to(self.device, non_blocking=True)
                features = self.encoder(images)
                if grids is None:
                    grids = [list(map(int, feature.shape[-2:])) for feature in features]
                for layer_index, (decoder, feature) in enumerate(
                    zip(self.decoders, features, strict=True)
                ):
                    batch, channels, height, width = feature.shape
                    patch_count = batch * height * width
                    values = feature.permute(0, 2, 3, 1).reshape(
                        patch_count, channels
                    )
                    position = positional_encoding_2d(
                        self.condition_dim, height, width, self.device
                    )
                    condition = (
                        position.permute(1, 2, 0)
                        .reshape(-1, self.condition_dim)
                        .repeat(batch, 1)
                    )
                    parts: list[torch.Tensor] = []
                    for start in range(0, patch_count, self.fiber_size):
                        stop = min(start + self.fiber_size, patch_count)
                        latent, log_jacobian = decoder(
                            values[start:stop], [condition[start:stop]]
                        )
                        log_probability = (
                            channels * GAUSSIAN_LOG_CONSTANT
                            - 0.5 * latent.square().sum(1)
                            + log_jacobian
                        ) / channels
                        parts.append(log_probability.float().cpu())
                    layer_map = torch.cat(parts).reshape(batch, height, width)
                    batches[layer_index].append(layer_map)
        if grids is None:
            raise ValueError("cannot score an empty data loader")
        return [torch.cat(values) for values in batches], grids

    @staticmethod
    def calibration_references(
        calibration_maps: list[torch.Tensor],
    ) -> list[float]:
        return [float(layer.max()) for layer in calibration_maps]

    @staticmethod
    def image_scores(
        maps: list[torch.Tensor],
        references: list[float],
        *,
        input_size: int,
    ) -> np.ndarray:
        """Aggregate official probability maps using fixed normal references.

        The upstream code subtracts a maximum computed from the test set.  To
        avoid test-distribution leakage, IAD calibration maxima are fixed first
        and reused unchanged for calibration and test samples.  Values above a
        normal reference are clipped to probability one.
        """

        if len(maps) != len(references):
            raise ValueError("one calibration reference is required per feature map")
        normality: torch.Tensor | None = None
        for layer_map, reference in zip(maps, references, strict=True):
            probability = torch.exp(
                torch.clamp(layer_map - reference, min=-60.0, max=0.0)
            )
            upsampled = F.interpolate(
                probability.unsqueeze(1),
                size=(input_size, input_size),
                mode="bilinear",
                align_corners=True,
            ).squeeze(1)
            normality = upsampled if normality is None else normality + upsampled
        assert normality is not None
        # An additive constant does not affect AUROC or a calibration threshold.
        return (-normality.amin(dim=(1, 2))).numpy().astype(np.float64, copy=False)

    def checkpoint_state(
        self,
        *,
        next_meta_epoch: int,
        identity: dict[str, Any],
        train_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "format_version": 1,
            "identity": identity,
            "next_meta_epoch": int(next_meta_epoch),
            "decoders": self.decoders.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_history": train_history,
        }

    def load_checkpoint(
        self,
        checkpoint: dict[str, Any],
        *,
        expected_identity: dict[str, Any],
    ) -> tuple[int, list[dict[str, Any]]]:
        if checkpoint.get("format_version") != 1:
            raise RuntimeError("unsupported CFLOW checkpoint format")
        if checkpoint.get("identity") != expected_identity:
            raise RuntimeError(
                "checkpoint identity does not match this matrix cell: "
                f"{checkpoint.get('identity')} != {expected_identity}"
            )
        self.decoders.load_state_dict(checkpoint["decoders"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        return int(checkpoint["next_meta_epoch"]), list(
            checkpoint.get("train_history", [])
        )
