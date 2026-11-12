"""Configuration validation and stable task indexing for the CFLOW matrix."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CFLOWTask:
    task_id: int
    input_size: int
    modulation: str
    seed: int
    train_batch_size: int
    score_batch_size: int

    @property
    def relative_output_dir(self) -> Path:
        return (
            Path(f"input{self.input_size}")
            / self.modulation
            / f"seed{self.seed}"
        )

    def to_dict(self) -> dict[str, int | str]:
        return {
            "task_id": self.task_id,
            "input_size": self.input_size,
            "modulation": self.modulation,
            "seed": self.seed,
            "train_batch_size": self.train_batch_size,
            "score_batch_size": self.score_batch_size,
        }


def load_matrix_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "protocol_version",
        "matrix",
        "training",
        "evaluation",
        "checkpoint",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"CFLOW matrix config is missing {sorted(missing)}")
    matrix = config["matrix"]
    for key in (
        "input_sizes",
        "modulations",
        "seeds",
        "train_batch_size_by_input",
        "score_batch_size_by_input",
    ):
        if key not in matrix:
            raise ValueError(f"CFLOW matrix section is missing {key!r}")
    return config


def expand_tasks(config: dict[str, Any]) -> list[CFLOWTask]:
    matrix = config["matrix"]
    tasks: list[CFLOWTask] = []
    for input_size in matrix["input_sizes"]:
        size = int(input_size)
        key = str(size)
        if key not in matrix["train_batch_size_by_input"]:
            raise ValueError(f"missing train batch size for input {size}")
        if key not in matrix["score_batch_size_by_input"]:
            raise ValueError(f"missing score batch size for input {size}")
        for modulation in matrix["modulations"]:
            canonical_modulation = str(modulation).strip().upper()
            for seed in matrix["seeds"]:
                tasks.append(
                    CFLOWTask(
                        task_id=len(tasks),
                        input_size=size,
                        modulation=canonical_modulation,
                        seed=int(seed),
                        train_batch_size=int(
                            matrix["train_batch_size_by_input"][key]
                        ),
                        score_batch_size=int(
                            matrix["score_batch_size_by_input"][key]
                        ),
                    )
                )
    if not tasks:
        raise ValueError("CFLOW matrix must contain at least one task")
    identities = {
        (task.input_size, task.modulation, task.seed) for task in tasks
    }
    if len(identities) != len(tasks):
        raise ValueError("CFLOW matrix contains duplicate input/modulation/seed cells")
    return tasks


def canonical_config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
