"""Leakage-resistant IAD image-level anomaly-detection benchmark."""

from .data import IADSplit, build_normal_split, content_hash
from .metrics import evaluate_detection

__all__ = ["IADSplit", "build_normal_split", "content_hash", "evaluate_detection"]
