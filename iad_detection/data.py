"""Data splitting and preprocessing for the IAD detection benchmark.

Only normal training samples are available to :meth:`fit`.  The test labels are
kept outside the dataset used for model construction and are consumed only by
the final metric function.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def content_hash(sample: torch.Tensor) -> str:
    array = sample.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


@dataclass(frozen=True)
class IADSplit:
    fit_indices: np.ndarray
    calibration_indices: np.ndarray
    fit_hashes: tuple[str, ...]
    calibration_hashes: tuple[str, ...]


def build_normal_split(
    data: torch.Tensor,
    *,
    calibration_fraction: float = 0.2,
    seed: int = 2026,
) -> IADSplit:
    """Hash-grouped deterministic fit/calibration split.

    Identical samples can never occur on both sides.  Groups are ordered by a
    seeded SHA-256 digest and whole groups are assigned to calibration until
    the requested count is reached.
    """

    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between zero and one")
    groups: dict[str, list[int]] = {}
    for index in range(int(data.shape[0])):
        groups.setdefault(content_hash(data[index]), []).append(index)
    ordered = sorted(
        groups,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    target = max(1, int(round(data.shape[0] * calibration_fraction)))
    calibration_groups: list[str] = []
    calibration_count = 0
    for value in ordered:
        if calibration_count >= target:
            break
        calibration_groups.append(value)
        calibration_count += len(groups[value])
    calibration_set = set(calibration_groups)
    fit_groups = [value for value in ordered if value not in calibration_set]
    fit_indices = np.asarray(
        sorted(index for value in fit_groups for index in groups[value]), dtype=np.int64
    )
    calibration_indices = np.asarray(
        sorted(index for value in calibration_groups for index in groups[value]),
        dtype=np.int64,
    )
    if fit_indices.size == 0 or calibration_indices.size == 0:
        raise ValueError("normal split produced an empty partition")
    return IADSplit(
        fit_indices=fit_indices,
        calibration_indices=calibration_indices,
        fit_hashes=tuple(fit_groups),
        calibration_hashes=tuple(calibration_groups),
    )


class IADTensorDataset(Dataset):
    """Tensor-backed dataset with a locked preprocessing profile."""

    def __init__(
        self,
        data: torch.Tensor,
        indices: np.ndarray | list[int],
        *,
        profile: str,
        input_size: int,
    ) -> None:
        if data.ndim != 4 or tuple(data.shape[1:]) != (1, 64, 64):
            raise ValueError(f"expected IAD shape [N,1,64,64], got {tuple(data.shape)}")
        if profile not in {"iad_native_x10", "imagenet_x10", "rgb_unit_x10"}:
            raise ValueError(f"unsupported preprocessing profile: {profile}")
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.profile = profile
        self.input_size = int(input_size)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int):
        original_index = int(self.indices[item])
        x = self.data[original_index].float() * 10.0
        if self.input_size and tuple(x.shape[-2:]) != (
            self.input_size,
            self.input_size,
        ):
            x = F.interpolate(
                x.unsqueeze(0),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        if self.profile in {"imagenet_x10", "rgb_unit_x10"}:
            x = x.clamp(0.0, 1.0).repeat(3, 1, 1)
        if self.profile == "imagenet_x10":
            mean = x.new_tensor((0.485, 0.456, 0.406))[:, None, None]
            std = x.new_tensor((0.229, 0.224, 0.225))[:, None, None]
            x = (x - mean) / std
        return x, original_index


class IndexedLabelDataset(Dataset):
    """Adapter for legacy models expecting ``(image, label, index)`` batches."""

    def __init__(self, base: IADTensorDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, item: int):
        image, index = self.base[item]
        return image, 0, index
