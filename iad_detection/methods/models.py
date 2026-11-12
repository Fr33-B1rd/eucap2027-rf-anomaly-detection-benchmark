"""Auditable detector implementations used by the IAD benchmark.

The VAE-PER and PatchCore adapters follow their cited reference code closely.
EfficientAD-M ports the PDN/autoencoder/loss recipe and verified teacher weights
from Anomalib's Apache-2.0 reference. CFLOW remains a dependency-light
replication; every result records an explicit implementation variant.
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import (
    ResNet18_Weights,
    Wide_ResNet50_2_Weights,
    resnet18,
    wide_resnet50_2,
)
from torchvision.transforms import functional as vision_functional


def _images(batch: Any, device: torch.device) -> torch.Tensor:
    return batch[0].to(device, non_blocking=True)


def _score_batches(detector: Any, loader: DataLoader) -> np.ndarray:
    values: list[np.ndarray] = []
    detector.model_eval()
    with torch.no_grad():
        for batch in loader:
            values.append(
                detector.score_tensor(_images(batch, detector.device)).cpu().numpy()
            )
    return np.concatenate(values).astype(np.float64, copy=False)


class Detector:
    implementation_variant = "unspecified"
    preprocessing_profile = "imagenet_x10"
    input_size = 256

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device

    def fit(
        self, fit_loader: DataLoader, calibration_loader: DataLoader
    ) -> dict[str, Any]:
        raise NotImplementedError

    def score(self, loader: DataLoader) -> np.ndarray:
        return _score_batches(self, loader)

    def model_eval(self) -> None:
        raise NotImplementedError


class ConvVAE(nn.Module):
    """Device-safe port of github.com/YonghaoXu/vae_ism_ano/model.py."""

    def __init__(self, latent_dim: int = 75) -> None:
        super().__init__()
        channels = [1, 64, 128, 256, 512]
        blocks: list[nn.Module] = []
        for source, target in zip(channels[:-1], channels[1:]):
            blocks += [nn.Conv2d(source, target, 4, 2, 1), nn.LeakyReLU(0.2)]
        blocks.append(nn.Conv2d(512, 512, 4, 4, 0))
        self.encoder = nn.Sequential(*blocks)
        self.fc = nn.Sequential(nn.Linear(512, 4096), nn.LeakyReLU(0.2))
        self.to_mean = nn.Linear(4096, latent_dim)
        self.to_logvar = nn.Linear(4096, latent_dim)
        self.from_latent = nn.Sequential(nn.Linear(latent_dim, 512), nn.LeakyReLU(0.2))
        decoder: list[nn.Module] = [nn.ConvTranspose2d(512, 512, 4, 4, 0)]
        for source, target in ((512, 256), (256, 128), (128, 64), (64, 32)):
            decoder += [nn.ConvTranspose2d(source, target, 4, 2, 1), nn.LeakyReLU(0.2)]
        decoder += [nn.Conv2d(32, 1, 1), nn.Sigmoid()]
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.fc(self.encoder(x).flatten(1))
        return self.to_mean(hidden), self.to_logvar(hidden)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.from_latent(z).reshape(-1, 512, 1, 1))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encode(x)
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return self.decode(z), mean, logvar


def per_score(x: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    """Positive PER reconstruction-error score; higher means more anomalous.

    The upstream IAD implementation reports ``-PER`` because its public pickle
    used inverse labels.  This repository validates normal=0/anomaly=1, so the
    physically meaningful positive reconstruction error is used here.
    """

    difference = (x - reconstruction).abs()
    background = F.max_pool2d(x, kernel_size=3, stride=1, padding=1) < 0.05
    local_minimum = -F.max_pool2d(-difference, kernel_size=3, stride=1, padding=1)
    background_values = (local_minimum * background).flatten(1)
    signal_values = (local_minimum * ~background).flatten(1)
    return 2.0 * torch.quantile(background_values, 0.90, dim=1) + torch.quantile(
        signal_values, 0.99, dim=1
    )


class VAEPERDetector(Detector):
    implementation_variant = "iad_official_conv_vae_per_device_safe"
    preprocessing_profile = "iad_native_x10"
    input_size = 64

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        super().__init__(config, device)
        self.model = ConvVAE(int(config.get("latent_dim", 75))).to(device)

    def fit(
        self, fit_loader: DataLoader, calibration_loader: DataLoader
    ) -> dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(self.config.get("lr", 1e-4))
        )
        epochs = int(self.config.get("epochs", 100))
        history: list[float] = []
        for _ in range(epochs):
            self.model.train()
            total = 0.0
            count = 0
            for batch in fit_loader:
                x = _images(batch, self.device)
                reconstruction, mean, logvar = self.model(x)
                mse = F.mse_loss(reconstruction, x, reduction="sum")
                kl = -0.5 * torch.sum(1 + logvar - mean.square() - logvar.exp())
                loss = (mse + float(self.config.get("kl_weight", 0.1)) * kl) / x.shape[
                    0
                ]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * x.shape[0]
                count += x.shape[0]
            history.append(total / max(count, 1))
        return {"train_loss_last": history[-1], "epochs": epochs}

    def model_eval(self) -> None:
        self.model.eval()

    def score_tensor(self, x: torch.Tensor) -> torch.Tensor:
        mean, _ = self.model.encode(x)
        return per_score(x, self.model.decode(mean))


class PatchCoreDetector(Detector):
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        super().__init__(config, device)
        self.input_size = int(config.get("input_size", 256))
        backbone = str(config.get("backbone", "wide_resnet50_2"))
        layers = list(config.get("layers", ["layer2", "layer3"]))
        self.implementation_variant = (
            f"repository_patchcore_{backbone}_{'_'.join(layers)}"
        )
        from core.patchcore import PatchCore

        self.model = PatchCore(
            backbone=backbone,
            layers=layers,
            device=device,
            sample_patches_per_image=int(config.get("sample_patches_per_image", 32)),
            candidate_pool_size=int(config.get("candidate_pool_size", 30000)),
            coreset_ratio=float(config.get("coreset_ratio", 0.1)),
            max_memory_bank_size=int(config.get("max_memory_bank_size", 3000)),
            projection_dim=int(config.get("projection_dim", 128)),
            query_chunk_size=int(config.get("query_chunk_size", 4096)),
            seed=int(config.get("seed", 2026)),
            pretrained=True,
        )

    def fit(
        self, fit_loader: DataLoader, calibration_loader: DataLoader
    ) -> dict[str, Any]:
        self.model.fit(fit_loader)
        return {"memory_bank_size": int(self.model.memory_bank.shape[0])}

    def score(self, loader: DataLoader) -> np.ndarray:
        scores, _, _, _ = self.model.predict(loader)
        return np.asarray(scores, dtype=np.float64)

    def model_eval(self) -> None:
        self.model.feature_extractor.eval()


class ResNetFeatures(nn.Module):
    def __init__(
        self, *, pretrained: bool, layers: tuple[str, ...], backbone: str = "resnet18"
    ) -> None:
        super().__init__()
        if backbone == "resnet18":
            base = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            self.channels = {"layer1": 64, "layer2": 128, "layer3": 256}
        elif backbone == "wide_resnet50_2":
            base = wide_resnet50_2(
                weights=Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
            )
            self.channels = {"layer1": 256, "layer2": 512, "layer3": 1024}
        else:
            raise ValueError(f"unsupported ResNet backbone: {backbone}")
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1, self.layer2, self.layer3 = base.layer1, base.layer2, base.layer3
        self.layers = layers

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        x = self.stem(x)
        x = self.layer1(x)
        if "layer1" in self.layers:
            result["layer1"] = x
        x = self.layer2(x)
        if "layer2" in self.layers:
            result["layer2"] = x
        x = self.layer3(x)
        if "layer3" in self.layers:
            result["layer3"] = x
        return result


def _normalized_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (F.normalize(a, dim=1) - F.normalize(b, dim=1)).square().sum(1)


class STFPMDetector(Detector):
    implementation_variant = "stfpm_resnet18_feature_pyramid"

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        super().__init__(config, device)
        self.input_size = int(config.get("input_size", 256))
        layers = tuple(config.get("layers", ["layer1", "layer2", "layer3"]))
        self.teacher = ResNetFeatures(pretrained=True, layers=layers).to(device).eval()
        self.student = ResNetFeatures(pretrained=False, layers=layers).to(device)
        self.layers = layers
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False

    def fit(
        self, fit_loader: DataLoader, calibration_loader: DataLoader
    ) -> dict[str, Any]:
        optimizer = torch.optim.SGD(
            self.student.parameters(),
            lr=float(self.config.get("lr", 0.4)),
            momentum=0.9,
            weight_decay=float(self.config.get("weight_decay", 1e-4)),
        )
        epochs = int(self.config.get("epochs", 50))
        record_diagnostics = bool(self.config.get("record_diagnostics", False))
        history: list[dict[str, float | int]] = []

        def state_summary() -> dict[str, float | int]:
            parameter_sq = 0.0
            for parameter in self.student.parameters():
                parameter_sq += float(parameter.detach().float().square().sum())
            running_mean_sq = 0.0
            running_mean_count = 0
            running_var_sum = 0.0
            running_var_count = 0
            batchnorm_count = 0
            for module in self.student.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    batchnorm_count += 1
                    if module.running_mean is not None:
                        values = module.running_mean.detach().float()
                        running_mean_sq += float(values.square().sum())
                        running_mean_count += values.numel()
                    if module.running_var is not None:
                        values = module.running_var.detach().float()
                        running_var_sum += float(values.sum())
                        running_var_count += values.numel()
            return {
                "student_parameter_l2": math.sqrt(parameter_sq),
                "batchnorm_layers": batchnorm_count,
                "batchnorm_running_mean_rms": math.sqrt(
                    running_mean_sq / max(running_mean_count, 1)
                ),
                "batchnorm_running_var_mean": (
                    running_var_sum / max(running_var_count, 1)
                ),
            }

        last = math.nan
        optimizer_steps = 0
        for epoch in range(epochs):
            self.student.train()
            total, count = 0.0, 0
            for batch in fit_loader:
                x = _images(batch, self.device)
                with torch.no_grad():
                    teacher = self.teacher(x)
                student = self.student(x)
                loss = sum(
                    _normalized_difference(teacher[k], student[k]).mean()
                    for k in self.layers
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
                total += float(loss.detach()) * x.shape[0]
                count += x.shape[0]
            last = total / max(count, 1)
            if record_diagnostics:
                history.append(
                    {
                        "epoch": epoch + 1,
                        "train_loss": last,
                        **state_summary(),
                    }
                )
        result: dict[str, Any] = {"train_loss_last": last, "epochs": epochs}
        if record_diagnostics:
            result.update(
                {
                    "optimizer_steps": optimizer_steps,
                    "epoch_history": history,
                    "final_state_summary": state_summary(),
                }
            )
        return result

    def model_eval(self) -> None:
        self.teacher.eval()
        self.student.eval()

    def score_tensor(self, x: torch.Tensor) -> torch.Tensor:
        teacher, student = self.teacher(x), self.student(x)
        maps = [
            F.interpolate(
                _normalized_difference(teacher[k], student[k])[:, None],
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            for k in self.layers
        ]
        anomaly_map = torch.stack(maps).prod(0).pow(1.0 / len(maps))
        return torch.quantile(anomaly_map.flatten(1), 0.99, dim=1)


def _imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = x.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    return (x - mean) / std


class MediumPatchDescriptionNetwork(nn.Module):
    """EfficientAD-M PDN ported from Anomalib (Apache-2.0)."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=4)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3)
        self.conv5 = nn.Conv2d(512, out_channels, kernel_size=4)
        self.conv6 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.avgpool1 = nn.AvgPool2d(2, 2)
        self.avgpool2 = nn.AvgPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _imagenet_normalize(x)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return self.conv6(x)


class EfficientADAutoencoder(nn.Module):
    """Official EfficientAD image-to-teacher-feature autoencoder."""

    def __init__(self, out_channels: int = 384) -> None:
        super().__init__()
        self.encoder = nn.ModuleList(
            [
                nn.Conv2d(3, 32, 4, 2, 1),
                nn.Conv2d(32, 32, 4, 2, 1),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.Conv2d(64, 64, 4, 2, 1),
                nn.Conv2d(64, 64, 4, 2, 1),
                nn.Conv2d(64, 64, 8),
            ]
        )
        self.decoder = nn.ModuleList(
            [
                nn.Conv2d(64, 64, 4, padding=2),
                nn.Conv2d(64, 64, 4, padding=2),
                nn.Conv2d(64, 64, 4, padding=2),
                nn.Conv2d(64, 64, 4, padding=2),
                nn.Conv2d(64, 64, 4, padding=2),
                nn.Conv2d(64, 64, 4, padding=2),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.Conv2d(64, out_channels, 3, padding=1),
            ]
        )
        self.dropouts = nn.ModuleList(nn.Dropout(0.2) for _ in range(6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image_size = x.shape[-2:]
        x = _imagenet_normalize(x)
        for layer in self.encoder[:-1]:
            x = F.relu(layer(x))
        x = self.encoder[-1](x)
        sizes = [
            (image_size[0] // 64 - 1, image_size[1] // 64 - 1),
            (image_size[0] // 32, image_size[1] // 32),
            (image_size[0] // 16 - 1, image_size[1] // 16 - 1),
            (image_size[0] // 8, image_size[1] // 8),
            (image_size[0] // 4 - 1, image_size[1] // 4 - 1),
            (image_size[0] // 2 - 1, image_size[1] // 2 - 1),
            (math.ceil(image_size[0] / 4) - 8, math.ceil(image_size[1] / 4) - 8),
        ]
        for index in range(6):
            x = F.interpolate(x, size=sizes[index], mode="bilinear")
            x = self.dropouts[index](F.relu(self.decoder[index](x)))
        x = F.interpolate(x, size=sizes[6], mode="bilinear")
        x = F.relu(self.decoder[6](x))
        return self.decoder[7](x)


class EfficientADDetector(Detector):
    """EfficientAD-M with official PDN/AE/loss and Anomalib teacher weights."""

    implementation_variant = "efficientad_pdn_medium_anomalib_reference"
    preprocessing_profile = "rgb_unit_x10"

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        super().__init__(config, device)
        self.input_size = int(config.get("input_size", 256))
        if str(config.get("model_size", "medium")) != "medium":
            raise ValueError("this adapter implements EfficientAD PDN-medium only")
        if self.input_size != 256:
            raise ValueError(
                "official EfficientAD-M configuration requires 256x256 input"
            )
        self.channels = int(config.get("teacher_out_channels", 384))
        self.teacher = MediumPatchDescriptionNetwork(self.channels).to(device).eval()
        self.student = MediumPatchDescriptionNetwork(self.channels * 2).to(device)
        self.autoencoder = EfficientADAutoencoder(self.channels).to(device)
        weights = Path(str(config["teacher_weights"])).expanduser()
        if not weights.is_file():
            raise FileNotFoundError(
                f"EfficientAD-M teacher weights not found: {weights}"
            )
        expected_sha256 = str(config.get("teacher_weights_sha256", "")).lower()
        if expected_sha256:
            actual_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    "EfficientAD-M teacher weight checksum mismatch: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.teacher.load_state_dict(state)
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        self.teacher_mean = torch.zeros(1, self.channels, 1, 1, device=device)
        self.teacher_std = torch.ones(1, self.channels, 1, 1, device=device)
        self.map_quantiles: dict[str, torch.Tensor] | None = None

    def _normalized_teacher(self, x: torch.Tensor) -> torch.Tensor:
        return (self.teacher(x) - self.teacher_mean) / self.teacher_std

    @torch.no_grad()
    def _set_teacher_statistics(self, loader: DataLoader) -> None:
        channel_sum = torch.zeros(self.channels, device=self.device)
        channel_square_sum = torch.zeros(self.channels, device=self.device)
        count = 0
        for batch in loader:
            output = self.teacher(_images(batch, self.device))
            channel_sum += output.sum((0, 2, 3))
            channel_square_sum += output.square().sum((0, 2, 3))
            count += output.shape[0] * output.shape[2] * output.shape[3]
        mean = channel_sum / count
        variance = channel_square_sum / count - mean.square()
        self.teacher_mean = mean[None, :, None, None]
        self.teacher_std = variance.clamp_min(1e-12).sqrt()[None, :, None, None]

    @staticmethod
    def _augment(x: torch.Tensor) -> torch.Tensor:
        functions = (
            vision_functional.adjust_brightness,
            vision_functional.adjust_contrast,
            vision_functional.adjust_saturation,
        )
        coefficient = float(torch.empty(1).uniform_(0.8, 1.2).item())
        function = functions[int(torch.randint(0, len(functions), (1,)).item())]
        return function(x, coefficient)

    def _raw_maps(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            teacher = self._normalized_teacher(x)
            student = self.student(x)
            ae_output = self.autoencoder(x)
            map_st = (
                (teacher - student[:, : self.channels]).square().mean(1, keepdim=True)
            )
            map_ae = (
                (ae_output - student[:, self.channels :]).square().mean(1, keepdim=True)
            )
            map_st = F.interpolate(
                F.pad(map_st, (4, 4, 4, 4)), size=x.shape[-2:], mode="bilinear"
            )
            map_ae = F.interpolate(
                F.pad(map_ae, (4, 4, 4, 4)), size=x.shape[-2:], mode="bilinear"
            )
        return map_st, map_ae

    @torch.no_grad()
    def _set_map_quantiles(self, loader: DataLoader) -> None:
        maps_st: list[torch.Tensor] = []
        maps_ae: list[torch.Tensor] = []
        self.model_eval()
        max_per_image = max(1, (2**24) // max(len(loader.dataset), 1))
        generator = torch.Generator().manual_seed(int(self.config.get("seed", 2026)))
        for batch in loader:
            map_st, map_ae = self._raw_maps(_images(batch, self.device))
            for st_image, ae_image in zip(map_st.cpu(), map_ae.cpu(), strict=True):
                st_flat = st_image.flatten()
                ae_flat = ae_image.flatten()
                if st_flat.numel() > max_per_image:
                    selection = torch.randperm(st_flat.numel(), generator=generator)[
                        :max_per_image
                    ]
                    st_flat = st_flat[selection]
                    ae_flat = ae_flat[selection]
                maps_st.append(st_flat)
                maps_ae.append(ae_flat)
        st = torch.cat(maps_st).flatten()
        ae = torch.cat(maps_ae).flatten()
        self.map_quantiles = {
            "qa_st": torch.quantile(st, 0.90).to(self.device),
            "qb_st": torch.quantile(st, 0.995).to(self.device),
            "qa_ae": torch.quantile(ae, 0.90).to(self.device),
            "qb_ae": torch.quantile(ae, 0.995).to(self.device),
        }

    def fit(
        self, fit_loader: DataLoader, calibration_loader: DataLoader
    ) -> dict[str, Any]:
        imagenette_root = Path(str(self.config["imagenette_dir"])).expanduser()
        train_root = imagenette_root / "train"
        if not train_root.is_dir():
            raise FileNotFoundError(
                f"ImageNette training directory not found: {train_root}"
            )
        transform = transforms.Compose(
            [
                transforms.Resize((self.input_size * 2, self.input_size * 2)),
                transforms.RandomGrayscale(p=0.3),
                transforms.CenterCrop((self.input_size, self.input_size)),
                transforms.ToTensor(),
            ]
        )
        imagenette = ImageFolder(train_root, transform=transform)
        imagenette_loader = DataLoader(
            imagenette,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(int(self.config.get("seed", 2026))),
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        training_started = time.perf_counter()
        self._set_teacher_statistics(fit_loader)
        optimizer = torch.optim.Adam(
            list(self.student.parameters()) + list(self.autoencoder.parameters()),
            lr=float(self.config.get("lr", 1e-4)),
            weight_decay=float(self.config.get("weight_decay", 1e-5)),
        )
        max_steps = int(self.config.get("max_steps", 70000))
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, int(0.95 * max_steps)), gamma=0.1
        )
        fit_iterator = iter(fit_loader)
        imagenette_iterator = iter(imagenette_loader)
        loss_value = math.nan
        self.student.train()
        self.autoencoder.train()
        for _ in range(max_steps):
            try:
                batch = next(fit_iterator)
            except StopIteration:
                fit_iterator = iter(fit_loader)
                batch = next(fit_iterator)
            try:
                imagenette_batch = next(imagenette_iterator)[0]
            except StopIteration:
                imagenette_iterator = iter(imagenette_loader)
                imagenette_batch = next(imagenette_iterator)[0]
            x = _images(batch, self.device)
            imagenette_x = imagenette_batch.to(self.device, non_blocking=True)
            with torch.no_grad():
                teacher = self._normalized_teacher(x)
            student = self.student(x)
            distance = (teacher - student[:, : self.channels]).square()
            threshold = torch.quantile(distance.detach().flatten(), 0.999)
            loss_hard = distance[distance >= threshold].mean()
            penalty = self.student(imagenette_x)[:, : self.channels].square().mean()
            augmented = self._augment(x)
            with torch.no_grad():
                teacher_augmented = self._normalized_teacher(augmented)
            ae_output = self.autoencoder(augmented)
            student_ae = self.student(augmented)[:, self.channels :]
            loss = (
                loss_hard
                + penalty
                + (teacher_augmented - ae_output).square().mean()
                + (ae_output - student_ae).square().mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_value = float(loss.detach())
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        training_seconds = time.perf_counter() - training_started
        calibration_started = time.perf_counter()
        self._set_map_quantiles(calibration_loader)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        map_calibration_seconds = time.perf_counter() - calibration_started
        return {
            "train_loss_last": loss_value,
            "max_steps": max_steps,
            "training_seconds": training_seconds,
            "map_calibration_seconds": map_calibration_seconds,
        }

    def model_eval(self) -> None:
        self.teacher.eval()
        self.student.eval()
        self.autoencoder.eval()

    def score_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if self.map_quantiles is None:
            raise RuntimeError("EfficientAD calibration quantiles are not set")
        map_st, map_ae = self._raw_maps(x)
        q = self.map_quantiles
        map_st = (
            0.1 * (map_st - q["qa_st"]) / (q["qb_st"] - q["qa_st"]).clamp_min(1e-12)
        )
        map_ae = (
            0.1 * (map_ae - q["qa_ae"]) / (q["qb_ae"] - q["qa_ae"]).clamp_min(1e-12)
        )
        return (0.5 * map_st + 0.5 * map_ae).amax(dim=(1, 2, 3))


def _position_encoding(height: int, width: int, device: torch.device) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing="ij",
    )
    return torch.stack(
        (
            x,
            y,
            torch.sin(math.pi * x),
            torch.cos(math.pi * x),
            torch.sin(math.pi * y),
            torch.cos(math.pi * y),
        ),
        dim=-1,
    ).reshape(-1, 6)


class ConditionalCoupling(nn.Module):
    def __init__(self, dimension: int, condition_dim: int, swap: bool) -> None:
        super().__init__()
        self.half = dimension // 2
        self.swap = swap
        self.net = nn.Sequential(
            nn.Linear(self.half + condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, dimension),
        )

    def forward(
        self, value: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.swap:
            value = torch.cat((value[:, self.half :], value[:, : self.half]), dim=1)
        first, second = value[:, : self.half], value[:, self.half :]
        log_scale, shift = self.net(torch.cat((first, condition), dim=1)).chunk(
            2, dim=1
        )
        log_scale = 1.5 * torch.tanh(log_scale)
        transformed = second * torch.exp(log_scale) + shift
        output = torch.cat((first, transformed), dim=1)
        if self.swap:
            output = torch.cat((output[:, self.half :], output[:, : self.half]), dim=1)
        return output, log_scale.sum(1)


class ConditionalFlow(nn.Module):
    def __init__(self, dimension: int, blocks: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ConditionalCoupling(dimension, 6, bool(i % 2)) for i in range(blocks)
        )

    def forward(
        self, value: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = value.new_zeros(value.shape[0])
        for block in self.blocks:
            value, update = block(value, condition)
            log_det += update
        return value, log_det


class CFLOWDetector(Detector):
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        super().__init__(config, device)
        self.input_size = int(config.get("input_size", 256))
        backbone = str(config.get("backbone", "wide_resnet50_2"))
        self.layer = str(config.get("layer", "layer3"))
        self.encoder = (
            ResNetFeatures(
                pretrained=True,
                layers=(self.layer,),
                backbone=backbone,
            )
            .to(device)
            .eval()
        )
        self.encoder_channels = self.encoder.channels[self.layer]
        self.implementation_variant = (
            f"cflow_conditional_flow_{backbone}_{self.layer}_replication"
        )
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        dimension = int(config.get("flow_dim", 64))
        generator = torch.Generator().manual_seed(int(config.get("seed", 2026)))
        projection = torch.randn(
            self.encoder_channels, dimension, generator=generator
        ) / math.sqrt(self.encoder_channels)
        self.projection = projection.to(device)
        self.flow = ConditionalFlow(dimension, int(config.get("flow_blocks", 4))).to(
            device
        )

    def _features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        feature = self.encoder(x)[self.layer]
        batch, _, height, width = feature.shape
        vectors = (
            feature.permute(0, 2, 3, 1).reshape(-1, self.encoder_channels)
            @ self.projection
        )
        condition = _position_encoding(height, width, x.device).repeat(batch, 1)
        return vectors, condition, height, width

    def fit(
        self, fit_loader: DataLoader, calibration_loader: DataLoader
    ) -> dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.flow.parameters(), lr=float(self.config.get("lr", 2e-4))
        )
        epochs = int(self.config.get("epochs", 20))
        max_patches = int(self.config.get("patches_per_batch", 4096))
        last = math.nan
        for _ in range(epochs):
            self.flow.train()
            total, count = 0.0, 0
            for batch in fit_loader:
                with torch.no_grad():
                    vectors, condition, _, _ = self._features(
                        _images(batch, self.device)
                    )
                if vectors.shape[0] > max_patches:
                    selection = torch.randperm(vectors.shape[0], device=self.device)[
                        :max_patches
                    ]
                    vectors, condition = vectors[selection], condition[selection]
                latent, log_det = self.flow(vectors, condition)
                loss = (0.5 * latent.square().sum(1) - log_det).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * vectors.shape[0]
                count += vectors.shape[0]
            last = total / max(count, 1)
        return {"train_nll_last": last, "epochs": epochs}

    def model_eval(self) -> None:
        self.encoder.eval()
        self.flow.eval()

    def score_tensor(self, x: torch.Tensor) -> torch.Tensor:
        vectors, condition, height, width = self._features(x)
        latent, log_det = self.flow(vectors, condition)
        nll = 0.5 * latent.square().sum(1) - log_det
        nll = nll.reshape(x.shape[0], height * width)
        return torch.quantile(nll, 0.99, dim=1)


class DINOv2KNNDetector(Detector):
    implementation_variant = "dinov2_vits14_patch_knn"
    input_size = 252  # exact multiple of the ViT-S/14 patch size

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        super().__init__(config, device)
        self.input_size = int(config.get("input_size", 252))
        hub_dir = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        if not hub_dir.exists():
            raise RuntimeError(
                "DINOv2 source is not cached. Clone facebookresearch/dinov2 into "
                f"{hub_dir} and cache dinov2_vits14_pretrain.pth before running."
            )
        self.model = (
            torch.hub.load(
                str(hub_dir), "dinov2_vits14", source="local", pretrained=True
            )
            .to(device)
            .eval()
        )
        self.memory: torch.Tensor | None = None

    def _patches(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model.forward_features(x)
        return F.normalize(output["x_norm_patchtokens"], dim=-1)

    def fit(
        self, fit_loader: DataLoader, calibration_loader: DataLoader
    ) -> dict[str, Any]:
        candidates: list[torch.Tensor] = []
        per_image = int(self.config.get("patches_per_image", 16))
        generator = torch.Generator().manual_seed(int(self.config.get("seed", 2026)))
        with torch.no_grad():
            for batch in fit_loader:
                patches = self._patches(_images(batch, self.device)).cpu()
                count = min(per_image, patches.shape[1])
                indices = torch.randperm(patches.shape[1], generator=generator)[:count]
                candidates.append(patches[:, indices].reshape(-1, patches.shape[-1]))
        memory = torch.cat(candidates)
        maximum = int(self.config.get("max_memory_bank_size", 20000))
        if memory.shape[0] > maximum:
            memory = memory[
                torch.randperm(memory.shape[0], generator=generator)[:maximum]
            ]
        self.memory = memory.to(self.device)
        return {"memory_bank_size": int(self.memory.shape[0])}

    def model_eval(self) -> None:
        self.model.eval()

    def score_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if self.memory is None:
            raise RuntimeError("fit must be called before score")
        queries = self._patches(x)
        flat = queries.reshape(-1, queries.shape[-1])
        minima: list[torch.Tensor] = []
        chunk = int(self.config.get("query_chunk_size", 2048))
        for start in range(0, flat.shape[0], chunk):
            similarity = flat[start : start + chunk] @ self.memory.T
            minima.append(1.0 - similarity.max(1).values)
        distances = torch.cat(minima).reshape(queries.shape[:2])
        return torch.quantile(distances, 0.99, dim=1)


def build_detector(name: str, config: dict[str, Any], device: torch.device) -> Detector:
    implementations = {
        "vae_per": VAEPERDetector,
        "patchcore": PatchCoreDetector,
        "stfpm": STFPMDetector,
        "efficientad": EfficientADDetector,
        "cflow": CFLOWDetector,
        "dinov2_knn": DINOv2KNNDetector,
    }
    try:
        detector_type = implementations[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown detector {name!r}; choose from {sorted(implementations)}"
        ) from exc
    return detector_type(config, device)
