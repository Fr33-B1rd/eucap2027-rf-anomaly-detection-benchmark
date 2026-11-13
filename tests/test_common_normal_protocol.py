from pathlib import Path

import numpy as np

from iad_detection.common_normal_protocol import (
    expand_common_normal_tasks,
    load_common_normal_config,
    seeded_index_split,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/iad_common_normal.json"


def test_locked_common_normal_config_and_task_counts() -> None:
    config = load_common_normal_config(CONFIG)
    assert config["protocol"]["normal_reference_source"] == "CHIRP"
    assert config["protocol"]["normal_test_source"] == "CHIRP"
    assert config["evaluation"]["threshold_quantile"] == 0.99
    gpu = expand_common_normal_tasks(config, "gpu")
    cpu = expand_common_normal_tasks(config, "cpu")
    assert len(gpu) == 25
    assert len(cpu) == 0
    assert [gpu[index].method for index in (0, 5, 10, 15, 20)] == [
        "patchcore",
        "efficientad",
        "cflow",
        "stfpm",
        "vae_per",
    ]


def test_seeded_index_split_is_reproducible_and_disjoint() -> None:
    first = seeded_index_split(4388, calibration_fraction=0.2, seed=2026)
    repeated = seeded_index_split(4388, calibration_fraction=0.2, seed=2026)
    other = seeded_index_split(4388, calibration_fraction=0.2, seed=2027)
    assert first.fit_indices.size == 3510
    assert first.calibration_indices.size == 878
    assert np.array_equal(first.fit_indices, repeated.fit_indices)
    assert np.array_equal(first.calibration_indices, repeated.calibration_indices)
    assert not np.array_equal(first.calibration_indices, other.calibration_indices)
    assert np.intersect1d(first.fit_indices, first.calibration_indices).size == 0
    assert np.array_equal(
        np.sort(np.concatenate((first.fit_indices, first.calibration_indices))),
        np.arange(4388),
    )
