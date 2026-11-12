#!/usr/bin/env python
"""Run the pre-specified 1000-epoch IAD VAE-PER ablation.

Training completes before any test score is computed. Model states at the
pre-specified epochs are retained only to evaluate the complete learning curve;
no checkpoint is selected using test labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataio.iad import discover_iad_sources, safe_load_iad_pickle  # noqa: E402
from iad_detection.data import (  # noqa: E402
    IADTensorDataset,
    IndexedLabelDataset,
    build_normal_split,
)
from iad_detection.final_matrix import (  # noqa: E402
    canonical_config_hash,
    expand_final_tasks,
    load_final_config,
    resolve_method_config,
)
from iad_detection.metrics import evaluate_detection  # noqa: E402
from iad_detection.methods import build_detector  # noqa: E402
from scripts.run_iad_final_berzelius import (  # noqa: E402
    atomic_json,
    atomic_scores,
    file_sha256,
    git_state,
    make_loader,
    print_tasks,
    smoke_test_indices,
    synchronize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/iad_vae_per1000_berzelius.json",
    )
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--task-count", action="store_true")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "outputs/iad_vae_per1000_berzelius",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def code_snapshot_sha256() -> str:
    roots = (
        REPOSITORY_ROOT / "dataio" / "iad",
        REPOSITORY_ROOT / "iad_detection",
    )
    paths = [
        REPOSITORY_ROOT / "scripts/run_iad_vae_epoch_ablation.py",
        REPOSITORY_ROOT / "scripts/run_iad_final_berzelius.py",
    ]
    for root in roots:
        paths.extend(root.rglob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.as_posix()):
        relative = path.relative_to(REPOSITORY_ROOT).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def checkpoint_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def train_to_checkpoints(
    detector: Any,
    fit_loader: DataLoader,
    *,
    epochs: int,
    checkpoints: list[int],
    lr: float,
    kl_weight: float,
    device: torch.device,
) -> tuple[list[float], dict[int, dict[str, torch.Tensor]], float]:
    optimizer = torch.optim.Adam(detector.model.parameters(), lr=lr)
    history: list[float] = []
    states: dict[int, dict[str, torch.Tensor]] = {}
    checkpoint_set = set(checkpoints)
    synchronize(device)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        detector.model.train()
        total = 0.0
        count = 0
        for batch in fit_loader:
            x = batch[0].to(device, non_blocking=True)
            reconstruction, mean, logvar = detector.model(x)
            mse = F.mse_loss(reconstruction, x, reduction="sum")
            kl = -0.5 * torch.sum(1 + logvar - mean.square() - logvar.exp())
            loss = (mse + kl_weight * kl) / x.shape[0]
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite VAE loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * x.shape[0]
            count += x.shape[0]
        history.append(total / max(count, 1))
        if epoch in checkpoint_set:
            states[epoch] = checkpoint_state(detector.model)
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(
                f"epoch={epoch}/{epochs} train_loss={history[-1]:.8f}",
                flush=True,
            )
    synchronize(device)
    elapsed = time.perf_counter() - started
    if sorted(states) != checkpoints:
        raise RuntimeError(f"missing checkpoint states: {sorted(states)}")
    return history, states, elapsed


def main() -> None:
    args = parse_args()
    config = load_final_config(args.config)
    tasks = expand_final_tasks(config)
    if set(config["matrix"]["methods"]) != {"vae_per"}:
        raise ValueError("the VAE epoch ablation config must contain only vae_per")
    if args.task_count:
        print(len(tasks))
        return
    if args.list_tasks:
        print_tasks(tasks)
        return
    if args.task_id is None or not 0 <= args.task_id < len(tasks):
        raise SystemExit(f"--task-id must be in [0, {len(tasks) - 1}]")
    if args.workers < 0:
        raise SystemExit("--workers cannot be negative")

    task = tasks[args.task_id]
    method_config = resolve_method_config(config, "vae_per", REPOSITORY_ROOT)
    formal_epochs = int(method_config["epochs"])
    formal_checkpoints = sorted(
        {int(value) for value in config["evaluation"]["checkpoint_epochs"]}
    )
    if not formal_checkpoints or formal_checkpoints[-1] != formal_epochs:
        raise ValueError("checkpoint_epochs must end at the configured epoch count")
    if formal_checkpoints[0] <= 0:
        raise ValueError("checkpoint epochs must be positive")

    epochs = 2 if args.smoke else formal_epochs
    checkpoints = [1, 2] if args.smoke else formal_checkpoints
    dataset_dir = (args.dataset_dir or Path(config["dataset_dir"])).expanduser()
    output_dir = args.output_root.expanduser() / task.relative_output_dir
    if args.smoke:
        output_dir = args.output_root.expanduser() / "smoke" / task.relative_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "run.json"
    scores_path = output_dir / "scores.npz"

    sources = {item.modulation: item for item in discover_iad_sources(dataset_dir)}
    if task.modulation not in sources:
        raise FileNotFoundError(
            f"missing {task.modulation}; found {sorted(sources)} in {dataset_dir}"
        )
    source = sources[task.modulation]
    identity = {
        "protocol_version": config["protocol_version"],
        "config_sha256": canonical_config_hash(config),
        "code_snapshot_sha256": code_snapshot_sha256(),
        "task": task.to_dict(),
        "source_sha256": file_sha256(source.path),
        "smoke": bool(args.smoke),
    }
    if run_path.exists():
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        if previous.get("identity") != identity:
            raise RuntimeError(f"existing result identity mismatch: {run_path}")
        if args.resume:
            print(f"[resume] completed task {task.task_id}: {run_path}", flush=True)
            return
        raise FileExistsError(f"result exists; pass --resume: {run_path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    set_seed(task.seed)
    started_utc = datetime.now(timezone.utc).isoformat()
    end_to_end_started = time.perf_counter()
    bundle = safe_load_iad_pickle(source.path)
    split = build_normal_split(
        bundle.train_data,
        calibration_fraction=float(config["evaluation"]["calibration_fraction"]),
        seed=task.seed,
    )
    fit_indices = split.fit_indices
    calibration_indices = split.calibration_indices
    test_indices = np.arange(bundle.test_data.shape[0], dtype=np.int64)
    if args.smoke:
        fit_indices = fit_indices[:128]
        calibration_indices = calibration_indices[:32]
        test_indices = smoke_test_indices(bundle.test_labels)

    method_config["seed"] = task.seed
    detector = build_detector("vae_per", method_config, device)
    fit_dataset = IADTensorDataset(
        bundle.train_data,
        fit_indices,
        profile=detector.preprocessing_profile,
        input_size=detector.input_size,
    )
    calibration_dataset = IADTensorDataset(
        bundle.train_data,
        calibration_indices,
        profile=detector.preprocessing_profile,
        input_size=detector.input_size,
    )
    test_dataset = IADTensorDataset(
        bundle.test_data,
        test_indices,
        profile=detector.preprocessing_profile,
        input_size=detector.input_size,
    )
    batch_size = int(method_config["batch_size"])
    fit_loader = make_loader(
        fit_dataset,
        batch_size=batch_size,
        workers=args.workers,
        shuffle=True,
        seed=task.seed,
    )
    calibration_loader = make_loader(
        calibration_dataset,
        batch_size=batch_size,
        workers=args.workers,
        shuffle=False,
        seed=task.seed,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=batch_size,
        workers=args.workers,
        shuffle=False,
        seed=task.seed,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history, states, training_seconds = train_to_checkpoints(
        detector,
        fit_loader,
        epochs=epochs,
        checkpoints=checkpoints,
        lr=float(method_config["lr"]),
        kl_weight=float(method_config["kl_weight"]),
        device=device,
    )

    # Training is now complete. Test labels and scores cannot influence model
    # fitting or the pre-specified checkpoint set from this point onward.
    test_labels = np.asarray(bundle.test_labels[test_indices], dtype=np.int64)
    checkpoint_results: dict[str, Any] = {}
    score_arrays: dict[str, np.ndarray] = {
        "calibration_indices": np.asarray(calibration_indices, dtype=np.int64),
        "test_indices": np.asarray(test_indices, dtype=np.int64),
        "test_labels": test_labels,
        "train_loss_history": np.asarray(history, dtype=np.float64),
        "checkpoint_epochs": np.asarray(checkpoints, dtype=np.int64),
    }
    for epoch in checkpoints:
        detector.model.load_state_dict(states[epoch])
        detector.model_eval()
        synchronize(device)
        calibration_started = time.perf_counter()
        calibration_scores = detector.score(calibration_loader)
        synchronize(device)
        calibration_seconds = time.perf_counter() - calibration_started
        test_started = time.perf_counter()
        test_scores = detector.score(test_loader)
        synchronize(device)
        test_seconds = time.perf_counter() - test_started
        metrics = evaluate_detection(
            calibration_scores,
            test_labels,
            test_scores,
            threshold_quantile=float(config["evaluation"]["threshold_quantile"]),
            bootstrap_samples=(
                20
                if args.smoke and epoch == checkpoints[-1]
                else int(config["evaluation"]["bootstrap_samples"])
                if epoch == checkpoints[-1]
                else 0
            ),
            bootstrap_seed=task.seed,
        )
        key = f"epoch{epoch:04d}"
        checkpoint_results[key] = {
            "epoch": epoch,
            "train_loss": history[epoch - 1],
            "metrics": metrics,
            "calibration_scoring_wall_seconds": calibration_seconds,
            "test_scoring_wall_seconds": test_seconds,
            "milliseconds_per_test_image": 1000.0 * test_seconds / len(test_dataset),
        }
        score_arrays[f"calibration_scores_{key}"] = np.asarray(
            calibration_scores, dtype=np.float64
        )
        score_arrays[f"test_scores_{key}"] = np.asarray(test_scores, dtype=np.float64)
        score_arrays[f"threshold_{key}"] = np.asarray(
            [metrics["threshold"]], dtype=np.float64
        )
        print(
            f"checkpoint={epoch} AUROC={metrics['auroc']:.6f} "
            f"AUPRC={metrics['auprc']:.6f}",
            flush=True,
        )

    if bool(config.get("save_scores", True)):
        atomic_scores(scores_path, **score_arrays)
    final_key = f"epoch{checkpoints[-1]:04d}"
    final_result = checkpoint_results[final_key]
    environment_git = git_state()
    result = {
        "identity": identity,
        "publication_eligible": not args.smoke,
        "method": "vae_per",
        "implementation_variant": detector.implementation_variant,
        "experiment": "pre_specified_epoch_ablation_no_test_checkpoint_selection",
        "modulation": bundle.modulation,
        "seed": task.seed,
        "preprocessing_profile": detector.preprocessing_profile,
        "input_size": detector.input_size,
        "fit_samples": len(fit_dataset),
        "calibration_samples": len(calibration_dataset),
        "test_samples": len(test_dataset),
        "normal_hash_overlap": len(set(split.fit_hashes) & set(split.calibration_hashes)),
        "config": method_config,
        "checkpoint_policy": {
            "epochs": checkpoints,
            "selection": "none; all pre-specified checkpoints are reported",
            "formal_comparison_epoch": checkpoints[-1],
            "test_evaluation_started_after_all_training_completed": True,
        },
        "fit": {
            "epochs": epochs,
            "train_loss_last": history[-1],
            "train_loss_history": history,
        },
        "checkpoints": checkpoint_results,
        "metrics": final_result["metrics"],
        "timing": {
            "training_wall_seconds": training_seconds,
            "end_to_end_wall_seconds": time.perf_counter() - end_to_end_started,
            "definition": (
                "CUDA-synchronized training wall time; all test scoring occurs "
                "only after training reaches the final pre-specified epoch"
            ),
        },
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else None
        ),
        "scores_file": scores_path.name if bool(config.get("save_scores", True)) else None,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "git_revision": environment_git["revision"],
            "git_dirty": environment_git["dirty"],
        },
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(run_path, result)
    print(
        f"final_AUROC={result['metrics']['auroc']:.6f} "
        f"training={training_seconds:.3f}s wrote={run_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
