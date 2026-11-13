from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from iad_detection.final_matrix import (
    canonical_config_hash,
    expand_final_tasks,
    load_final_config,
    resolve_method_config,
)
from scripts.analyze_iad_final_results import (
    exact_sign_flip_pvalue,
    holm_adjust,
    t_interval,
)
from scripts.run_iad_final_matrix import code_snapshot_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "iad_method_configs.json"


def test_final_matrix_has_stable_80_cell_task_map() -> None:
    config = load_final_config(CONFIG_PATH)
    tasks = expand_final_tasks(config)
    assert len(tasks) == 80
    assert (tasks[0].method, tasks[0].modulation, tasks[0].seed) == (
        "patchcore",
        "16QAM",
        2026,
    )
    assert (tasks[20].method, tasks[20].modulation, tasks[20].seed) == (
        "efficientad",
        "16QAM",
        2026,
    )
    assert (tasks[40].method, tasks[40].modulation, tasks[40].seed) == (
        "stfpm",
        "16QAM",
        2026,
    )
    assert (tasks[60].method, tasks[60].modulation, tasks[60].seed) == (
        "vae_per",
        "16QAM",
        2026,
    )
    assert tasks[-1].relative_output_dir == Path("vae_per/QPSK/seed2030")
    assert config["matrix"]["methods"] == [
        "patchcore",
        "efficientad",
        "stfpm",
        "vae_per",
    ]
    assert {
        method: values["input_size"]
        for method, values in config["methods"].items()
    } == {
        "patchcore": 64,
        "efficientad": 256,
        "stfpm": 64,
        "vae_per": 64,
    }


def test_dinov2_is_rejected_from_final_matrix(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["matrix"]["methods"].append("dinov2_knn")
    config["methods"]["dinov2_knn"] = {}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="intentionally excluded"):
        load_final_config(path)


def test_efficientad_assets_resolve_below_explicit_root(tmp_path: Path) -> None:
    config = load_final_config(CONFIG_PATH)
    resolved = resolve_method_config(config, "efficientad", tmp_path)
    assert Path(resolved["teacher_weights"]).is_relative_to(tmp_path.resolve())
    assert Path(resolved["imagenette_dir"]).is_relative_to(tmp_path.resolve())
    assert "teacher_weights_relative" not in resolved
    assert "imagenette_dir_relative" not in resolved


def test_config_and_code_hashes_are_stable_and_sensitive() -> None:
    config = load_final_config(CONFIG_PATH)
    assert canonical_config_hash(config) == canonical_config_hash(copy.deepcopy(config))
    changed = copy.deepcopy(config)
    changed["methods"]["stfpm"]["epochs"] += 1
    assert canonical_config_hash(config) != canonical_config_hash(changed)
    assert len(code_snapshot_sha256()) == 64


def test_small_n_exact_test_and_holm_adjustment() -> None:
    assert exact_sign_flip_pvalue([1, 1, 1, 1, 1]) == pytest.approx(0.0625)
    assert exact_sign_flip_pvalue([0, 0, 0, 0, 0]) == pytest.approx(1.0)
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert t_interval([2.0] * 5) == (2.0, 2.0)
