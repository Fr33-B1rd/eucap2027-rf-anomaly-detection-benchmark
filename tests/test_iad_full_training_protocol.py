from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from iad_detection.common_normal_protocol import expand_common_normal_tasks
from scripts.run_iad_full_training_berzelius import (
    evaluate_without_calibration,
    load_full_training_config,
    select_fit_indices,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/iad_benchmark.json"
SAMPLE_CONFIGS = {
    16: ROOT / "configs/iad_training_samples_n16_berzelius.json",
    32: ROOT / "configs/iad_training_samples_n32_berzelius.json",
    64: ROOT / "configs/iad_training_samples_n64_berzelius.json",
    128: ROOT / "configs/iad_training_samples_n128_berzelius.json",
    256: ROOT / "configs/iad_training_samples_n256_berzelius.json",
}


def test_full_training_config_has_no_calibration_protocol() -> None:
    config = load_full_training_config(CONFIG)
    assert config["protocol"]["normal_usage"] == "all_reference_normals_for_fitting"
    assert config["protocol"]["held_out_calibration"] is False
    assert "calibration_fraction" not in config["evaluation"]
    assert "threshold_quantile" not in config["evaluation"]
    tasks = expand_common_normal_tasks(config, "gpu")
    assert len(tasks) == 20
    assert {(task.method, task.seed) for task in tasks} == {
        (method, seed)
        for method in ("patchcore", "efficientad", "cflow", "vae_per")
        for seed in (2026, 2027, 2028, 2029, 2030)
    }


def test_full_training_config_rejects_calibration_fields() -> None:
    config = load_full_training_config(CONFIG)
    invalid = copy.deepcopy(config)
    invalid["evaluation"]["threshold_quantile"] = 0.99
    temporary = ROOT / "tests" / "_invalid_full_training_config.json"
    try:
        temporary.write_text(__import__("json").dumps(invalid), encoding="utf-8")
        with pytest.raises(ValueError, match="calibration settings"):
            load_full_training_config(temporary)
    finally:
        temporary.unlink(missing_ok=True)


def test_training_sample_configs_are_paired_and_nested() -> None:
    expected_methods = ("patchcore", "efficientad", "cflow", "vae_per")
    expected_seeds = (2026, 2027, 2028, 2029, 2030)
    for budget, path in SAMPLE_CONFIGS.items():
        config = load_full_training_config(path)
        assert config["protocol"]["normal_train_budget"] == budget
        assert config["protocol"]["held_out_calibration"] is False
        tasks = expand_common_normal_tasks(config, "gpu")
        assert len(tasks) == 20
        assert [(task.method, task.seed) for task in tasks] == [
            (method, seed) for method in expected_methods for seed in expected_seeds
        ]

    for seed in expected_seeds:
        selections = {
            budget: select_fit_indices(4388, seed=seed, budget=budget)
            for budget in SAMPLE_CONFIGS
        }
        for smaller, larger in zip(sorted(selections), sorted(selections)[1:]):
            assert selections[smaller].size == smaller
            assert selections[larger].size == larger
            assert set(selections[smaller]).issubset(set(selections[larger]))
        assert np.array_equal(
            selections[16], select_fit_indices(4388, seed=seed, budget=16)
        )


def test_threshold_free_metrics_do_not_emit_decision_threshold() -> None:
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    scores = np.asarray([0.0, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.0])
    result = evaluate_without_calibration(
        labels,
        scores,
        max_fpr=0.01,
        bootstrap_samples=10,
        bootstrap_seed=2026,
    )
    assert result["auroc"] == pytest.approx(1.0)
    assert result["tpr_at_fpr_limit"] == pytest.approx(1.0)
    assert result["decision_threshold"] is None
    assert not any("calibration" in key for key in result)
