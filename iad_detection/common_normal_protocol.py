"""Task indexing and deterministic splits for the common-normal IAD protocol."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CommonNormalTask:
    task_id: int
    task_kind: str
    method: str
    seed: int

    @property
    def relative_output_dir(self) -> Path:
        return Path(self.method) / f"seed{self.seed}"

    def to_dict(self) -> dict[str, int | str]:
        return {
            "task_id": self.task_id,
            "task_kind": self.task_kind,
            "method": self.method,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class IndexSplit:
    fit_indices: np.ndarray
    calibration_indices: np.ndarray


def load_common_normal_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "protocol_version",
        "dataset_dir",
        "protocol",
        "matrix",
        "evaluation",
        "methods",
        "assets",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"common-normal config is missing {sorted(missing)}")

    protocol = config["protocol"]
    for key in (
        "normal_reference_source",
        "normal_test_source",
        "anomaly_conditions",
        "expected_counts",
    ):
        if key not in protocol:
            raise ValueError(f"protocol.{key} is required")
    if str(protocol["normal_reference_source"]).upper() != "CHIRP":
        raise ValueError("the locked common normal reference source must be CHIRP")
    if str(protocol["normal_test_source"]).upper() != "CHIRP":
        raise ValueError("the locked common normal test source must be CHIRP")
    conditions = [str(value).upper() for value in protocol["anomaly_conditions"]]
    if conditions != ["16QAM", "CHIRP", "GMSK", "QPSK"]:
        raise ValueError("anomaly_conditions must be 16QAM, CHIRP, GMSK, QPSK")

    matrix = config["matrix"]
    for key in ("gpu_methods", "cpu_methods", "seeds"):
        if key not in matrix:
            raise ValueError(f"matrix.{key} is required")
    if not matrix["gpu_methods"]:
        raise ValueError("matrix.gpu_methods must be non-empty")
    if not matrix["seeds"]:
        raise ValueError("matrix.seeds must be non-empty")
    all_methods = list(matrix["gpu_methods"]) + list(matrix["cpu_methods"])
    if len(set(all_methods)) != len(all_methods):
        raise ValueError("a method cannot occur in both GPU and CPU task lists")
    absent = set(all_methods).difference(config["methods"])
    if absent:
        raise ValueError(f"methods have no configuration: {sorted(absent)}")
    if "dinov2_knn" in all_methods:
        raise ValueError("DINOv2-kNN is excluded from this benchmark matrix")
    if not 0.0 < float(config["evaluation"]["calibration_fraction"]) < 1.0:
        raise ValueError("evaluation.calibration_fraction must be between 0 and 1")
    if not 0.0 < float(config["evaluation"]["threshold_quantile"]) < 1.0:
        raise ValueError("evaluation.threshold_quantile must be between 0 and 1")
    return config


def expand_common_normal_tasks(
    config: dict[str, Any], task_kind: str
) -> list[CommonNormalTask]:
    if task_kind not in {"gpu", "cpu"}:
        raise ValueError("task_kind must be 'gpu' or 'cpu'")
    methods = config["matrix"][f"{task_kind}_methods"]
    seeds = config["matrix"]["seeds"]
    tasks: list[CommonNormalTask] = []
    for method in methods:
        for seed in seeds:
            tasks.append(
                CommonNormalTask(
                    task_id=len(tasks),
                    task_kind=task_kind,
                    method=str(method),
                    seed=int(seed),
                )
            )
    return tasks


def seeded_index_split(
    sample_count: int,
    *,
    calibration_fraction: float,
    seed: int,
) -> IndexSplit:
    """Split normal-reference indices without inspecting sample contents."""

    if sample_count < 2:
        raise ValueError("at least two samples are required")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")
    calibration_count = max(1, int(round(sample_count * calibration_fraction)))
    if calibration_count >= sample_count:
        raise ValueError("calibration split would leave no fitting samples")
    permutation = np.random.default_rng(seed).permutation(sample_count)
    calibration = np.sort(permutation[:calibration_count]).astype(np.int64)
    fitting = np.sort(permutation[calibration_count:]).astype(np.int64)
    if np.intersect1d(fitting, calibration).size:
        raise RuntimeError("fitting and calibration indices overlap")
    if fitting.size + calibration.size != sample_count:
        raise RuntimeError("normal split does not cover the reference pool")
    return IndexSplit(fit_indices=fitting, calibration_indices=calibration)


def canonical_config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
