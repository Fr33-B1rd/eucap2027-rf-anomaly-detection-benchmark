from __future__ import annotations

import numpy as np
import pytest
import torch

from iad_detection.data import IADTensorDataset, build_normal_split
from iad_detection.metrics import evaluate_detection
from iad_detection.methods.models import (
    ConvVAE,
    EfficientADAutoencoder,
    MediumPatchDescriptionNetwork,
    per_score,
)


def test_hash_split_keeps_duplicate_samples_together() -> None:
    data = torch.arange(6 * 64 * 64, dtype=torch.float32).reshape(6, 1, 64, 64)
    data[5] = data[0]
    split = build_normal_split(data, calibration_fraction=0.34, seed=7)
    assert set(split.fit_indices).isdisjoint(split.calibration_indices)
    assert sorted(
        np.concatenate((split.fit_indices, split.calibration_indices)).tolist()
    ) == list(range(6))
    assert (0 in split.fit_indices) == (5 in split.fit_indices)
    assert set(split.fit_hashes).isdisjoint(split.calibration_hashes)


def test_transfer_preprocessing_is_fixed_x10_then_imagenet() -> None:
    data = torch.zeros(1, 1, 64, 64)
    data[0, 0, 0, 0] = 0.2
    dataset = IADTensorDataset(data, [0], profile="imagenet_x10", input_size=64)
    image, index = dataset[0]
    assert image.shape == (3, 64, 64)
    assert index == 0
    assert image[0, 0, 0].item() == pytest.approx((1.0 - 0.485) / 0.229)
    assert image[0, 1, 1].item() == pytest.approx((0.0 - 0.485) / 0.229)


def test_native_preprocessing_preserves_64_and_multiplies_by_ten() -> None:
    data = torch.full((1, 1, 64, 64), 0.02)
    dataset = IADTensorDataset(data, [0], profile="iad_native_x10", input_size=64)
    image, _ = dataset[0]
    assert image.shape == (1, 64, 64)
    assert image.mean().item() == pytest.approx(0.2)


def test_efficientad_preprocessing_does_not_apply_imagenet_norm_twice() -> None:
    data = torch.full((1, 1, 64, 64), 0.02)
    dataset = IADTensorDataset(data, [0], profile="rgb_unit_x10", input_size=256)
    image, _ = dataset[0]
    assert image.shape == (3, 256, 256)
    assert image.mean().item() == pytest.approx(0.2)


def test_efficientad_medium_pdn_and_autoencoder_shapes_match() -> None:
    pdn = MediumPatchDescriptionNetwork(384).eval()
    autoencoder = EfficientADAutoencoder(384).eval()
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        assert pdn(x).shape == (1, 384, 56, 56)
        assert autoencoder(x).shape == (1, 384, 56, 56)


def test_operating_threshold_uses_calibration_scores_only() -> None:
    calibration = np.arange(20, dtype=np.float64)
    labels = np.array([0, 0, 1, 1])
    scores_a = np.array([1.0, 2.0, 30.0, 40.0])
    scores_b = np.array([-1000.0, 1000.0, -500.0, 500.0])
    result_a = evaluate_detection(calibration, labels, scores_a, bootstrap_samples=0)
    result_b = evaluate_detection(calibration, labels, scores_b, bootstrap_samples=0)
    assert result_a["threshold"] == result_b["threshold"] == 19.0


def test_vae_and_per_shapes_and_positive_error_direction() -> None:
    model = ConvVAE()
    x = torch.rand(2, 1, 64, 64)
    reconstruction, mean, logvar = model(x)
    assert reconstruction.shape == x.shape
    assert mean.shape == logvar.shape == (2, 75)
    exact = per_score(x, x)
    perturbed = per_score(x, torch.zeros_like(x))
    assert torch.allclose(exact, torch.zeros_like(exact))
    assert torch.all(perturbed >= exact)
