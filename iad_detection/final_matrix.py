"""Stable task indexing and validation for the Berzelius final rerun matrix."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FinalTask:
    task_id: int
    method: str
    modulation: str
    seed: int

    @property
    def relative_output_dir(self) -> Path:
        return Path(self.method) / self.modulation / f"seed{self.seed}"

    def to_dict(self) -> dict[str, int | str]:
        return {
            "task_id": self.task_id,
            "method": self.method,
            "modulation": self.modulation,
            "seed": self.seed,
        }


def load_final_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"protocol_version", "matrix", "evaluation", "methods", "assets"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"final matrix config is missing {sorted(missing)}")
    matrix = config["matrix"]
    for key in ("methods", "modulations", "seeds"):
        if key not in matrix or not matrix[key]:
            raise ValueError(f"matrix.{key} must be non-empty")
    absent = set(matrix["methods"]).difference(config["methods"])
    if absent:
        raise ValueError(f"matrix methods have no configuration: {sorted(absent)}")
    if "dinov2_knn" in matrix["methods"]:
        raise ValueError("DINOv2-kNN is intentionally excluded from the final matrix")
    return config


def expand_final_tasks(config: dict[str, Any]) -> list[FinalTask]:
    matrix = config["matrix"]
    tasks: list[FinalTask] = []
    for method in matrix["methods"]:
        for modulation in matrix["modulations"]:
            for seed in matrix["seeds"]:
                tasks.append(
                    FinalTask(
                        task_id=len(tasks),
                        method=str(method),
                        modulation=str(modulation).strip().upper(),
                        seed=int(seed),
                    )
                )
    identities = {(task.method, task.modulation, task.seed) for task in tasks}
    if len(identities) != len(tasks):
        raise ValueError("final matrix contains duplicate method/modulation/seed cells")
    return tasks


def canonical_config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def resolve_method_config(
    config: dict[str, Any], method: str, asset_root: str | Path
) -> dict[str, Any]:
    method_config = dict(config["methods"][method])
    if method == "efficientad":
        root = Path(asset_root).expanduser().resolve()
        method_config["teacher_weights"] = str(
            root / method_config.pop("teacher_weights_relative")
        )
        method_config["imagenette_dir"] = str(
            root / method_config.pop("imagenette_dir_relative")
        )
    return method_config
