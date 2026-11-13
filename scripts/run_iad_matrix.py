#!/usr/bin/env python
"""Run selected IAD benchmark cells in a platform-independent Python process."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "iad_benchmark.json"
DEFAULT_ASSET_ROOT = REPOSITORY_ROOT / ".cache" / "iad_assets"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs" / "iad_benchmark"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="*",
        help="Task IDs to run; omit to run the complete configured matrix.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--list-tasks", action="store_true")
    return parser.parse_args()


def task_count(config: Path) -> int:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "run_iad_benchmark.py"),
        "--config",
        str(config),
        "--task-kind",
        "gpu",
        "--task-count",
    ]
    return int(subprocess.check_output(command, text=True).strip())


def selected_task_ids(total: int, requested: list[int] | None) -> list[int]:
    values = list(range(total)) if requested is None else list(requested)
    invalid = [value for value in values if value < 0 or value >= total]
    if invalid:
        raise ValueError(f"task IDs outside [0, {total - 1}]: {invalid}")
    if len(values) != len(set(values)):
        raise ValueError("task IDs must be unique")
    return values


def main() -> None:
    args = parse_args()
    runner = REPOSITORY_ROOT / "scripts" / "run_iad_benchmark.py"
    if args.list_tasks:
        subprocess.run(
            [
                sys.executable,
                str(runner),
                "--config",
                str(args.config),
                "--task-kind",
                "gpu",
                "--list-tasks",
            ],
            check=True,
        )
        return

    total = task_count(args.config)
    task_ids = selected_task_ids(total, args.task_ids)
    for position, task_id in enumerate(task_ids, start=1):
        print(f"[matrix {position}/{len(task_ids)}] task {task_id}", flush=True)
        command = [
            sys.executable,
            str(runner),
            "--config",
            str(args.config),
            "--task-kind",
            "gpu",
            "--task-id",
            str(task_id),
            "--dataset-dir",
            str(args.dataset_dir),
            "--asset-root",
            str(args.asset_root),
            "--output-root",
            str(args.output_root),
            "--device",
            args.device,
            "--workers",
            str(args.workers),
        ]
        if args.resume:
            command.append("--resume")
        if args.smoke:
            command.append("--smoke")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
