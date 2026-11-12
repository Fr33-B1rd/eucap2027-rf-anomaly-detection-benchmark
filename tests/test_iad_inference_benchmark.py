from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import numpy as np

from scripts.benchmark_iad_inference import (
    numeric_summary,
    score_difference_statistics,
    scores_are_close,
    validate_benchmark_config,
)
from scripts.analyze_iad_final_results import LOCKED_INPUT_SIZES


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "iad_inference_benchmark.json"


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_benchmark_locks_comparable_batches_and_cflow64() -> None:
    config = load_config()
    validate_benchmark_config(config)
    assert config["cflow_input_size"] == 64
    assert LOCKED_INPUT_SIZES["cflow"] == 64
    assert config["latency"]["batch_size"] == 1
    assert config["throughput"]["batch_size"] == 16
    assert config["output_semantics"] == "image_level_anomaly_score_only"


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("cflow_input_size",), 128, "native 64x64"),
        (("latency", "batch_size"), 2, "batch size 1"),
        (("throughput", "batch_size"), 8, "batch size 16"),
    ],
)
def test_benchmark_rejects_incomparable_settings(
    path: tuple[str, ...], value: int, message: str
) -> None:
    config = copy.deepcopy(load_config())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        validate_benchmark_config(config)


def test_numeric_summary_uses_sample_sd_and_retains_quantiles() -> None:
    summary = numeric_summary([1.0, 2.0, 3.0])
    assert summary["n"] == 3
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["sample_sd"] == pytest.approx(1.0)
    assert summary["median"] == pytest.approx(2.0)
    assert summary["q25"] == pytest.approx(1.5)
    assert summary["q75"] == pytest.approx(2.5)


def test_score_equivalence_accounts_for_method_score_scale() -> None:
    batched = np.asarray([25.0, 35.0], dtype=np.float32)
    individual = batched + np.asarray([0.0027, -0.0027], dtype=np.float32)
    statistics = score_difference_statistics(batched, individual)

    assert statistics["max_abs"] == pytest.approx(0.0027, abs=2.0e-6)
    assert statistics["max_rel"] < 2.0e-4
    assert scores_are_close(
        batched,
        individual,
        absolute_tolerance=1.0e-4,
        relative_tolerance=2.0e-4,
    )
    assert not scores_are_close(
        batched,
        individual,
        absolute_tolerance=1.0e-4,
        relative_tolerance=1.0e-5,
    )
