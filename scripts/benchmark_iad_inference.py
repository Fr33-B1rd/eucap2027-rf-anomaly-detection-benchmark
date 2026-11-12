#!/usr/bin/env python
"""Benchmark the locked IAD methods with comparable latency/throughput rules.

This is a timing-only protocol.  It fits one representative normal-only cell,
then measures image-level anomaly scoring under two distinct conditions:

* batch-1 latency, with one CUDA synchronization per request;
* batch-16 throughput, synchronized around blocks of requests.

Both model-only and end-to-end (dataset preprocessing + host-to-device copy)
numbers are retained. Stored benchmark AUROCs are not recomputed here.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataio.iad import discover_iad_sources, safe_load_iad_pickle  # noqa: E402
from iad_detection.cflow_matrix import load_matrix_config  # noqa: E402
from iad_detection.data import (  # noqa: E402
    IADTensorDataset,
    IndexedLabelDataset,
    build_normal_split,
)
from iad_detection.final_matrix import (  # noqa: E402
    load_final_config,
    resolve_method_config,
)
from iad_detection.methods import build_detector  # noqa: E402


ScoreTensor = Callable[[torch.Tensor], torch.Tensor]
ReferenceScore = Callable[[torch.Tensor], np.ndarray]


@dataclass
class PreparedMethod:
    name: str
    input_size: int
    preprocessing_profile: str
    score_tensor: ScoreTensor
    reference_score: ReferenceScore
    model_eval: Callable[[], None]
    test_dataset: IADTensorDataset
    fit_seconds: float
    implementation_variant: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/iad_inference_benchmark.json",
    )
    parser.add_argument(
        "--final-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/iad_final_berzelius_matrix.json",
    )
    parser.add_argument(
        "--cflow-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/iad_cflow_official_matrix.json",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def git_state() -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=REPOSITORY_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": "unknown", "dirty": None}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def set_epoch_seed(seed: int, epoch: int) -> None:
    value = seed + epoch * 100_003
    random.seed(value)
    np.random.seed(value % (2**32 - 1))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_loader(
    dataset: IADTensorDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    drop_last: bool,
    seed: int,
    persistent_workers: bool = False,
) -> DataLoader:
    return DataLoader(
        IndexedLabelDataset(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(persistent_workers and workers > 0),
        generator=torch.Generator().manual_seed(seed),
    )


def infinite_batches(loader: DataLoader) -> Iterator[tuple[torch.Tensor, ...]]:
    while True:
        yield from loader


def numeric_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty timing vector")
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / 1024**2)


def model_latency(
    score: ScoreTensor,
    batch: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, Any]:
    with torch.inference_mode():
        for _ in range(warmup):
            score(batch)
        synchronize(device)
        reset_peak_memory(device)
        samples_ms: list[float] = []
        for _ in range(iterations):
            synchronize(device)
            started = time.perf_counter()
            score(batch)
            synchronize(device)
            samples_ms.append(
                1000.0 * (time.perf_counter() - started) / int(batch.shape[0])
            )
    return {
        "unit": "milliseconds_per_image",
        "summary": numeric_summary(samples_ms),
        "raw_samples": samples_ms,
        "peak_cuda_memory_mb": peak_memory_mb(device),
    }


def model_throughput(
    score: ScoreTensor,
    batch: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
    trials: int,
    device: torch.device,
) -> dict[str, Any]:
    if iterations < trials:
        raise ValueError("throughput iterations must be at least the trial count")
    per_trial = [iterations // trials] * trials
    for index in range(iterations % trials):
        per_trial[index] += 1
    with torch.inference_mode():
        for _ in range(warmup):
            score(batch)
        synchronize(device)
        reset_peak_memory(device)
        images_per_second: list[float] = []
        for trial_iterations in per_trial:
            synchronize(device)
            started = time.perf_counter()
            for _ in range(trial_iterations):
                score(batch)
            synchronize(device)
            elapsed = time.perf_counter() - started
            images_per_second.append(
                trial_iterations * int(batch.shape[0]) / elapsed
            )
    per_image_ms = [1000.0 / value for value in images_per_second]
    return {
        "unit": "images_per_second",
        "summary": numeric_summary(images_per_second),
        "raw_samples": images_per_second,
        "milliseconds_per_image_summary": numeric_summary(per_image_ms),
        "peak_cuda_memory_mb": peak_memory_mb(device),
    }


def end_to_end_latency(
    score: ScoreTensor,
    loader: DataLoader,
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, Any]:
    stream = infinite_batches(loader)
    with torch.inference_mode():
        for _ in range(warmup):
            images = next(stream)[0].to(device, non_blocking=True)
            score(images)
        synchronize(device)
        reset_peak_memory(device)
        samples_ms: list[float] = []
        for _ in range(iterations):
            synchronize(device)
            started = time.perf_counter()
            images = next(stream)[0].to(device, non_blocking=True)
            score(images)
            synchronize(device)
            samples_ms.append(
                1000.0 * (time.perf_counter() - started) / int(images.shape[0])
            )
    return {
        "unit": "milliseconds_per_image",
        "summary": numeric_summary(samples_ms),
        "raw_samples": samples_ms,
        "peak_cuda_memory_mb": peak_memory_mb(device),
    }


def end_to_end_throughput(
    score: ScoreTensor,
    loader: DataLoader,
    *,
    warmup: int,
    iterations: int,
    trials: int,
    device: torch.device,
) -> dict[str, Any]:
    if iterations < trials:
        raise ValueError("throughput iterations must be at least the trial count")
    stream = infinite_batches(loader)
    with torch.inference_mode():
        for _ in range(warmup):
            images = next(stream)[0].to(device, non_blocking=True)
            score(images)
        synchronize(device)
        reset_peak_memory(device)
        images_per_second: list[float] = []
        per_trial = [iterations // trials] * trials
        for index in range(iterations % trials):
            per_trial[index] += 1
        for trial_iterations in per_trial:
            count = 0
            synchronize(device)
            started = time.perf_counter()
            for _ in range(trial_iterations):
                images = next(stream)[0].to(device, non_blocking=True)
                count += int(images.shape[0])
                score(images)
            synchronize(device)
            elapsed = time.perf_counter() - started
            images_per_second.append(count / elapsed)
    per_image_ms = [1000.0 / value for value in images_per_second]
    return {
        "unit": "images_per_second",
        "summary": numeric_summary(images_per_second),
        "raw_samples": images_per_second,
        "milliseconds_per_image_summary": numeric_summary(per_image_ms),
        "peak_cuda_memory_mb": peak_memory_mb(device),
    }


def patchcore_image_scores(detector: Any, images: torch.Tensor) -> torch.Tensor:
    model = detector.model
    embeddings = model._compute_patch_embeddings(images)  # noqa: SLF001
    flat = embeddings.reshape(-1, model.embedding_dim)
    memory_t = model.memory_bank.t().contiguous()
    minima: list[torch.Tensor] = []
    for start in range(0, flat.shape[0], model.query_chunk_size):
        query = flat[start : start + model.query_chunk_size]
        query_norm = query.square().sum(1, keepdim=True)
        distances = (
            query_norm
            + model.memory_bank_norms.unsqueeze(0)
            - 2.0 * query @ memory_t
        ).clamp_min(0.0)
        minima.append(torch.sqrt(distances.min(1).values))
    patch_scores = torch.cat(minima).reshape(images.shape[0], *model.patch_grid_size)
    return patch_scores.amax(dim=(1, 2))


def native_score_function(method: str, detector: Any) -> ScoreTensor:
    if method == "patchcore":
        return lambda images: patchcore_image_scores(detector, images)
    return detector.score_tensor


def native_reference_function(detector: Any, device: torch.device) -> ReferenceScore:
    def reference(images: torch.Tensor) -> np.ndarray:
        count = int(images.shape[0])
        batch = (
            images.cpu(),
            torch.zeros(count, dtype=torch.int64),
            torch.arange(count, dtype=torch.int64),
        )
        synchronize(device)
        values = detector.score([batch])
        synchronize(device)
        return np.asarray(values, dtype=np.float64)

    return reference


def adapt_smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    if "epochs" in result:
        result["epochs"] = 1
    if "max_steps" in result:
        result["max_steps"] = 2
    for key, value in (
        ("candidate_pool_size", 256),
        ("max_memory_bank_size", 64),
        ("sample_patches_per_image", 4),
    ):
        if key in result:
            result[key] = value
    return result


def prepare_native_method(
    method: str,
    *,
    final_config: dict[str, Any],
    bundle: Any,
    split: Any,
    test_indices: np.ndarray,
    asset_root: Path,
    seed: int,
    workers: int,
    smoke: bool,
    device: torch.device,
) -> PreparedMethod:
    method_config = resolve_method_config(final_config, method, asset_root)
    method_config["seed"] = seed
    if smoke:
        method_config = adapt_smoke_config(method_config)
    detector = build_detector(method, method_config, device)
    profile = detector.preprocessing_profile
    input_size = int(detector.input_size)
    fit_indices = split.fit_indices
    calibration_indices = split.calibration_indices
    train_batch_size = int(method_config.get("batch_size", 32))
    if smoke:
        fit_indices = fit_indices[: max(64, 2 * train_batch_size)]
        calibration_indices = calibration_indices[:32]
        test_indices = test_indices[:256]
    fit_dataset = IADTensorDataset(
        bundle.train_data, fit_indices, profile=profile, input_size=input_size
    )
    calibration_dataset = IADTensorDataset(
        bundle.train_data,
        calibration_indices,
        profile=profile,
        input_size=input_size,
    )
    test_dataset = IADTensorDataset(
        bundle.test_data, test_indices, profile=profile, input_size=input_size
    )
    fit_loader = make_loader(
        fit_dataset,
        batch_size=train_batch_size,
        workers=workers,
        shuffle=True,
        drop_last=False,
        seed=seed,
    )
    calibration_loader = make_loader(
        calibration_dataset,
        batch_size=train_batch_size,
        workers=workers,
        shuffle=False,
        drop_last=False,
        seed=seed,
    )
    synchronize(device)
    started = time.perf_counter()
    detector.fit(fit_loader, calibration_loader)
    synchronize(device)
    fit_seconds = time.perf_counter() - started
    detector.model_eval()
    return PreparedMethod(
        name=method,
        input_size=input_size,
        preprocessing_profile=profile,
        score_tensor=native_score_function(method, detector),
        reference_score=native_reference_function(detector, device),
        model_eval=detector.model_eval,
        test_dataset=test_dataset,
        fit_seconds=fit_seconds,
        implementation_variant=str(detector.implementation_variant),
    )


def cflow_score_tensor(model: Any, references: list[float], images: torch.Tensor) -> torch.Tensor:
    from iad_detection.methods.cflow_official import (  # noqa: PLC0415
        GAUSSIAN_LOG_CONSTANT,
        positional_encoding_2d,
    )

    features = model.encoder(images)
    normality: torch.Tensor | None = None
    for decoder, feature, reference in zip(
        model.decoders, features, references, strict=True
    ):
        batch, channels, height, width = feature.shape
        patch_count = batch * height * width
        values = feature.permute(0, 2, 3, 1).reshape(patch_count, channels)
        position = positional_encoding_2d(
            model.condition_dim, height, width, model.device
        )
        condition = (
            position.permute(1, 2, 0)
            .reshape(-1, model.condition_dim)
            .repeat(batch, 1)
        )
        parts: list[torch.Tensor] = []
        for start in range(0, patch_count, model.fiber_size):
            stop = min(start + model.fiber_size, patch_count)
            latent, log_jacobian = decoder(
                values[start:stop], [condition[start:stop]]
            )
            log_probability = (
                channels * GAUSSIAN_LOG_CONSTANT
                - 0.5 * latent.square().sum(1)
                + log_jacobian
            ) / channels
            parts.append(log_probability.float())
        layer_map = torch.cat(parts).reshape(batch, height, width)
        probability = torch.exp(
            torch.clamp(layer_map - float(reference), min=-60.0, max=0.0)
        )
        upsampled = F.interpolate(
            probability.unsqueeze(1),
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=True,
        ).squeeze(1)
        normality = upsampled if normality is None else normality + upsampled
    if normality is None:
        raise RuntimeError("CFLOW produced no feature maps")
    return -normality.amin(dim=(1, 2))


def prepare_cflow_method(
    *,
    cflow_config: dict[str, Any],
    bundle: Any,
    split: Any,
    test_indices: np.ndarray,
    input_size: int,
    seed: int,
    workers: int,
    smoke: bool,
    device: torch.device,
) -> PreparedMethod:
    from iad_detection.methods.cflow_official import OfficialCFLOW  # noqa: PLC0415

    matrix = cflow_config["matrix"]
    key = str(input_size)
    if input_size not in [int(value) for value in matrix["input_sizes"]]:
        raise ValueError(f"CFLOW input {input_size} is not present in its matrix")
    train_batch_size = int(matrix["train_batch_size_by_input"][key])
    score_batch_size = int(matrix["score_batch_size_by_input"][key])
    training = dict(cflow_config["training"])
    fit_indices = split.fit_indices
    calibration_indices = split.calibration_indices
    if smoke:
        training["meta_epochs"] = 1
        training["sub_epochs"] = 1
        fit_indices = fit_indices[: max(64, train_batch_size)]
        calibration_indices = calibration_indices[:32]
        test_indices = test_indices[:256]
    profile = "imagenet_x10"
    fit_dataset = IADTensorDataset(
        bundle.train_data, fit_indices, profile=profile, input_size=input_size
    )
    calibration_dataset = IADTensorDataset(
        bundle.train_data,
        calibration_indices,
        profile=profile,
        input_size=input_size,
    )
    test_dataset = IADTensorDataset(
        bundle.test_data, test_indices, profile=profile, input_size=input_size
    )
    model = OfficialCFLOW(training, device)
    synchronize(device)
    started = time.perf_counter()
    for epoch in range(int(training["meta_epochs"])):
        set_epoch_seed(seed, epoch)
        fit_loader = make_loader(
            fit_dataset,
            batch_size=train_batch_size,
            workers=workers,
            shuffle=True,
            drop_last=True,
            seed=seed + epoch * 100_003,
        )
        model.train_meta_epoch(epoch, fit_loader)
    synchronize(device)
    fit_seconds = time.perf_counter() - started
    calibration_loader = make_loader(
        calibration_dataset,
        batch_size=score_batch_size,
        workers=workers,
        shuffle=False,
        drop_last=False,
        seed=seed,
    )
    calibration_maps, _ = model.collect_log_probability_maps(calibration_loader)
    references = model.calibration_references(calibration_maps)
    model.encoder.eval()
    model.decoders.eval()

    def score(images: torch.Tensor) -> torch.Tensor:
        return cflow_score_tensor(model, references, images)

    def reference(images: torch.Tensor) -> np.ndarray:
        count = int(images.shape[0])
        batch = (
            images.cpu(),
            torch.zeros(count, dtype=torch.int64),
            torch.arange(count, dtype=torch.int64),
        )
        maps, _ = model.collect_log_probability_maps([batch])
        return model.image_scores(maps, references, input_size=input_size)

    def model_eval() -> None:
        model.encoder.eval()
        model.decoders.eval()

    return PreparedMethod(
        name="cflow",
        input_size=input_size,
        preprocessing_profile=profile,
        score_tensor=score,
        reference_score=reference,
        model_eval=model_eval,
        test_dataset=test_dataset,
        fit_seconds=fit_seconds,
        implementation_variant=(
            "official_cflow_wrn50_2_l2_l3_l4_"
            "iad_calibration_reference_scoring"
        ),
    )


def first_images(dataset: IADTensorDataset, count: int) -> torch.Tensor:
    loader = make_loader(
        dataset,
        batch_size=count,
        workers=0,
        shuffle=False,
        drop_last=False,
        seed=0,
    )
    return next(iter(loader))[0]


def score_difference_statistics(
    left: np.ndarray, right: np.ndarray
) -> dict[str, float]:
    if left.shape != right.shape:
        raise ValueError(f"score shape mismatch: {left.shape} != {right.shape}")
    absolute = np.abs(left - right)
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1.0e-12)
    relative = absolute / scale
    return {
        "max_abs": float(np.max(absolute)),
        "max_rel": float(np.max(relative)),
    }


def scores_are_close(
    left: np.ndarray,
    right: np.ndarray,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> bool:
    return bool(
        np.allclose(
            left,
            right,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )
    )


def validate_score_semantics(
    prepared: PreparedMethod,
    *,
    device: torch.device,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, float]:
    images = first_images(prepared.test_dataset, min(4, len(prepared.test_dataset)))
    with torch.inference_mode():
        direct = prepared.score_tensor(images.to(device)).detach().cpu().numpy()
        reference = prepared.reference_score(images)
        individual = np.concatenate(
            [
                prepared.score_tensor(image[None].to(device)).detach().cpu().numpy()
                for image in images
            ]
        )
    reference_error = score_difference_statistics(direct, reference)
    batch_error = score_difference_statistics(direct, individual)
    if not scores_are_close(
        direct,
        reference,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    ):
        raise RuntimeError(
            f"{prepared.name} optimized/reference score mismatch: "
            f"max_abs={reference_error['max_abs']}, "
            f"max_rel={reference_error['max_rel']}; "
            f"atol={absolute_tolerance}, rtol={relative_tolerance}"
        )
    if not scores_are_close(
        direct,
        individual,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    ):
        raise RuntimeError(
            f"{prepared.name} batch-1/batched score mismatch: "
            f"max_abs={batch_error['max_abs']}, "
            f"max_rel={batch_error['max_rel']}; "
            f"atol={absolute_tolerance}, rtol={relative_tolerance}"
        )
    return {
        "optimized_vs_reference_max_abs": reference_error["max_abs"],
        "optimized_vs_reference_max_rel": reference_error["max_rel"],
        "batch1_vs_batched_max_abs": batch_error["max_abs"],
        "batch1_vs_batched_max_rel": batch_error["max_rel"],
        "tolerance_abs": absolute_tolerance,
        "tolerance_rel": relative_tolerance,
    }


def benchmark_method(
    prepared: PreparedMethod,
    *,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    prepared.model_eval()
    latency = config["latency"]
    throughput = config["throughput"]
    latency_batch = first_images(
        prepared.test_dataset, int(latency["batch_size"])
    ).to(device)
    throughput_batch = first_images(
        prepared.test_dataset, int(throughput["batch_size"])
    ).to(device)
    latency_loader = make_loader(
        prepared.test_dataset,
        batch_size=int(latency["batch_size"]),
        workers=int(latency["end_to_end_workers"]),
        shuffle=False,
        drop_last=False,
        seed=seed,
    )
    throughput_loader = make_loader(
        prepared.test_dataset,
        batch_size=int(throughput["batch_size"]),
        workers=int(throughput["end_to_end_workers"]),
        shuffle=False,
        drop_last=True,
        seed=seed,
        persistent_workers=True,
    )
    checks = validate_score_semantics(
        prepared,
        device=device,
        absolute_tolerance=float(config["score_tolerance_abs"]),
        relative_tolerance=float(config["score_tolerance_rel"]),
    )
    return {
        "method": prepared.name,
        "implementation_variant": prepared.implementation_variant,
        "input_size": prepared.input_size,
        "preprocessing_profile": prepared.preprocessing_profile,
        "representative_fit_seconds": prepared.fit_seconds,
        "score_semantics_check": checks,
        "model_only": {
            "batch1_latency": model_latency(
                prepared.score_tensor,
                latency_batch,
                warmup=int(latency["warmup_iterations"]),
                iterations=int(latency["measurement_iterations"]),
                device=device,
            ),
            "batch16_throughput": model_throughput(
                prepared.score_tensor,
                throughput_batch,
                warmup=int(throughput["warmup_iterations"]),
                iterations=int(throughput["measurement_iterations"]),
                trials=int(throughput["trials"]),
                device=device,
            ),
        },
        "end_to_end": {
            "batch1_latency": end_to_end_latency(
                prepared.score_tensor,
                latency_loader,
                warmup=int(latency["warmup_iterations"]),
                iterations=int(latency["measurement_iterations"]),
                device=device,
            ),
            "batch16_throughput": end_to_end_throughput(
                prepared.score_tensor,
                throughput_loader,
                warmup=int(throughput["warmup_iterations"]),
                iterations=int(throughput["measurement_iterations"]),
                trials=int(throughput["trials"]),
                device=device,
            ),
        },
    }


def validate_benchmark_config(config: dict[str, Any]) -> None:
    required = {
        "protocol_version",
        "representative",
        "methods",
        "cflow_input_size",
        "precision",
        "output_semantics",
        "score_tolerance_abs",
        "score_tolerance_rel",
        "latency",
        "throughput",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"benchmark config is missing {sorted(missing)}")
    if int(config["cflow_input_size"]) != 64:
        raise ValueError("the benchmark locks CFLOW to native 64x64 input")
    if config["precision"] != "float32":
        raise ValueError("the primary benchmark is locked to float32")
    if float(config["score_tolerance_abs"]) < 0:
        raise ValueError("score_tolerance_abs must be non-negative")
    if float(config["score_tolerance_rel"]) < 0:
        raise ValueError("score_tolerance_rel must be non-negative")
    if int(config["latency"]["batch_size"]) != 1:
        raise ValueError("latency benchmark must use batch size 1")
    if int(config["throughput"]["batch_size"]) != 16:
        raise ValueError("throughput benchmark must use batch size 16")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    benchmark_config = json.loads(args.benchmark_config.read_text(encoding="utf-8"))
    validate_benchmark_config(benchmark_config)
    if args.smoke:
        benchmark_config = json.loads(json.dumps(benchmark_config))
        benchmark_config["latency"]["warmup_iterations"] = 2
        benchmark_config["latency"]["measurement_iterations"] = 5
        benchmark_config["throughput"]["warmup_iterations"] = 2
        benchmark_config["throughput"]["measurement_iterations"] = 4
        benchmark_config["throughput"]["trials"] = 2
    final_config = load_final_config(args.final_config)
    cflow_config = load_matrix_config(args.cflow_config)
    methods = args.methods or list(benchmark_config["methods"])
    unknown = set(methods).difference(benchmark_config["methods"])
    if unknown:
        raise ValueError(f"unknown benchmark methods: {sorted(unknown)}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("GPU timing requires an available CUDA device")
    modulation = str(benchmark_config["representative"]["modulation"]).upper()
    seed = int(benchmark_config["representative"]["seed"])
    set_seed(seed)
    sources = {
        item.modulation: item for item in discover_iad_sources(args.dataset_dir)
    }
    if modulation not in sources:
        raise FileNotFoundError(
            f"missing {modulation}; found {sorted(sources)} in {args.dataset_dir}"
        )
    source = sources[modulation]
    bundle = safe_load_iad_pickle(source.path)
    split = build_normal_split(
        bundle.train_data,
        calibration_fraction=float(final_config["evaluation"]["calibration_fraction"]),
        seed=seed,
    )
    test_indices = np.arange(bundle.test_data.shape[0], dtype=np.int64)
    started_utc = time.time()
    results: list[dict[str, Any]] = []
    for method in methods:
        print(f"[benchmark] preparing {method}", flush=True)
        set_seed(seed)
        if method == "cflow":
            prepared = prepare_cflow_method(
                cflow_config=cflow_config,
                bundle=bundle,
                split=split,
                test_indices=test_indices,
                input_size=int(benchmark_config["cflow_input_size"]),
                seed=seed,
                workers=int(benchmark_config["throughput"]["end_to_end_workers"]),
                smoke=bool(args.smoke),
                device=device,
            )
        else:
            prepared = prepare_native_method(
                method,
                final_config=final_config,
                bundle=bundle,
                split=split,
                test_indices=test_indices,
                asset_root=args.asset_root,
                seed=seed,
                workers=int(benchmark_config["throughput"]["end_to_end_workers"]),
                smoke=bool(args.smoke),
                device=device,
            )
        print(f"[benchmark] timing {method}", flush=True)
        result = benchmark_method(
            prepared, config=benchmark_config, seed=seed, device=device
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "method": method,
                    "model_latency_ms": result["model_only"]["batch1_latency"][
                        "summary"
                    ]["median"],
                    "e2e_latency_ms": result["end_to_end"]["batch1_latency"][
                        "summary"
                    ]["median"],
                    "model_throughput_images_s": result["model_only"][
                        "batch16_throughput"
                    ]["summary"]["median"],
                    "e2e_throughput_images_s": result["end_to_end"][
                        "batch16_throughput"
                    ]["summary"]["median"],
                }
            ),
            flush=True,
        )
        del prepared
        gc.collect()
        torch.cuda.empty_cache()
        synchronize(device)
    state = git_state()
    output = {
        "protocol_version": benchmark_config["protocol_version"],
        "publication_eligible": not args.smoke,
        "timing_only": True,
        "stored_auroc_results_unchanged": True,
        "benchmark_config": benchmark_config,
        "benchmark_config_sha256": canonical_hash(benchmark_config),
        "final_config_sha256": canonical_hash(final_config),
        "cflow_config_sha256": canonical_hash(cflow_config),
        "source_sha256": file_sha256(source.path),
        "definitions": {
            "model_only": (
                "preprocessed tensor already resident on GPU; image-level anomaly "
                "score only"
            ),
            "end_to_end": (
                "IADTensorDataset preprocessing, DataLoader retrieval, host-to-device "
                "copy, and image-level anomaly scoring"
            ),
            "latency": (
                "batch size 1; wall clock with CUDA synchronization for every request"
            ),
            "throughput": (
                "batch size 16; wall clock synchronized around multi-batch trials"
            ),
        },
        "representative": {"modulation": modulation, "seed": seed},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "git_revision": state["revision"],
            "git_dirty": state["dirty"],
        },
        "wall_seconds": time.time() - started_utc,
        "results": results,
    }
    atomic_json(args.output, output)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
