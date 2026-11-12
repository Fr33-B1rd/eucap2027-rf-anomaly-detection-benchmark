"""Metrics whose operating point is fixed using normal calibration data."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _stratified_bootstrap(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    aurocs: list[float] = []
    auprcs: list[float] = []
    for _ in range(samples):
        draw = np.concatenate(
            [
                rng.choice(negative, negative.size, replace=True),
                rng.choice(positive, positive.size, replace=True),
            ]
        )
        aurocs.append(float(roc_auc_score(labels[draw], scores[draw])))
        auprcs.append(float(average_precision_score(labels[draw], scores[draw])))
    return {"auroc": aurocs, "auprc": auprcs}


def evaluate_detection(
    calibration_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    *,
    threshold_quantile: float = 0.95,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 2026,
) -> dict[str, Any]:
    """Evaluate scores with the convention that larger means more anomalous."""

    calibration_scores = np.asarray(calibration_scores, dtype=np.float64).reshape(-1)
    test_scores = np.asarray(test_scores, dtype=np.float64).reshape(-1)
    test_labels = np.asarray(test_labels, dtype=np.int64).reshape(-1)
    if calibration_scores.size == 0 or not np.isfinite(calibration_scores).all():
        raise ValueError("calibration scores must be finite and non-empty")
    if test_scores.shape != test_labels.shape or not np.isfinite(test_scores).all():
        raise ValueError("test labels/scores must be finite and have equal shapes")
    if set(np.unique(test_labels).tolist()) != {0, 1}:
        raise ValueError("final test labels must contain both normal=0 and anomaly=1")

    threshold = float(
        np.quantile(calibration_scores, threshold_quantile, method="higher")
    )
    predicted = test_scores >= threshold
    normal = test_labels == 0
    anomaly = test_labels == 1
    result: dict[str, Any] = {
        "auroc": float(roc_auc_score(test_labels, test_scores)),
        "auprc": float(average_precision_score(test_labels, test_scores)),
        "threshold": threshold,
        "threshold_quantile": float(threshold_quantile),
        "calibration_fpr": float(np.mean(calibration_scores >= threshold)),
        "test_fpr": float(np.mean(predicted[normal])),
        "test_tpr": float(np.mean(predicted[anomaly])),
        "n_calibration": int(calibration_scores.size),
        "n_test_normal": int(normal.sum()),
        "n_test_anomaly": int(anomaly.sum()),
        "score_direction": "higher_is_more_anomalous",
    }
    if bootstrap_samples > 0:
        values = _stratified_bootstrap(
            test_labels, test_scores, samples=bootstrap_samples, seed=bootstrap_seed
        )
        for metric, draws in values.items():
            low, high = np.quantile(draws, [0.025, 0.975])
            result[f"{metric}_ci95"] = [float(low), float(high)]
    return result
