from pathlib import Path

import pytest

from scripts.prepare_iad_assets import safe_member_path
from scripts.run_iad_benchmark import load_full_training_config
from scripts.run_iad_matrix import selected_task_ids


ROOT = Path(__file__).resolve().parents[1]


def test_public_benchmark_config_is_loadable() -> None:
    config = load_full_training_config(ROOT / "configs" / "iad_benchmark.json")
    assert config["matrix"]["gpu_methods"] == [
        "patchcore",
        "efficientad",
        "cflow",
        "vae_per",
    ]
    assert len(config["matrix"]["seeds"]) == 5


def test_matrix_task_selection_defaults_to_all_tasks() -> None:
    assert selected_task_ids(4, None) == [0, 1, 2, 3]
    assert selected_task_ids(4, [3, 1]) == [3, 1]


@pytest.mark.parametrize("requested", [[-1], [4], [1, 1]])
def test_matrix_task_selection_rejects_invalid_requests(requested: list[int]) -> None:
    with pytest.raises(ValueError):
        selected_task_ids(4, requested)


def test_asset_extraction_rejects_parent_traversal(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        safe_member_path(tmp_path, "../escape.txt")
