"""Inter-annotator agreement metrics for IAD-Loc masks.

These functions deliberately operate on binary masks rather than proposal
scores.  The proposal generator is an annotation aid and must never be treated
as an annotator when agreement is reported.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


BBox = tuple[int, int, int, int]


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("mask contains NaN or infinite values")
    unique = np.unique(array)
    if not np.isin(unique, (0, 1, False, True)).all():
        raise ValueError(f"mask must be binary, got values {unique.tolist()}")
    return array.astype(bool, copy=False)


def tight_bbox(mask: np.ndarray) -> BBox | None:
    """Return an exclusive-upper-bound ``(x0, y0, x1, y1)`` bbox."""
    binary = _binary_mask(mask)
    rows, cols = np.nonzero(binary)
    if rows.size == 0:
        return None
    return (
        int(cols.min()),
        int(rows.min()),
        int(cols.max()) + 1,
        int(rows.max()) + 1,
    )


def mask_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Sørensen-Dice score; two empty masks agree perfectly."""
    a = _binary_mask(mask_a)
    b = _binary_mask(mask_b)
    if a.shape != b.shape:
        raise ValueError(f"mask shape mismatch: {a.shape} vs {b.shape}")
    denominator = int(a.sum()) + int(b.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(a, b).sum() / denominator)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Binary-mask intersection over union; two empty masks score 1."""
    a = _binary_mask(mask_a)
    b = _binary_mask(mask_b)
    if a.shape != b.shape:
        raise ValueError(f"mask shape mismatch: {a.shape} vs {b.shape}")
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def bbox_iou(box_a: Sequence[int] | None, box_b: Sequence[int] | None) -> float:
    """IoU for exclusive-upper-bound bboxes.

    Two missing boxes score 1; a single missing box scores 0.
    """
    if box_a is None and box_b is None:
        return 1.0
    if box_a is None or box_b is None:
        return 0.0
    if len(box_a) != 4 or len(box_b) != 4:
        raise ValueError("bbox must contain exactly four coordinates")
    ax0, ay0, ax1, ay1 = (int(value) for value in box_a)
    bx0, by0, bx1, by1 = (int(value) for value in box_b)
    if ax1 <= ax0 or ay1 <= ay0 or bx1 <= bx0 or by1 <= by0:
        raise ValueError("bbox must have positive width and height")
    intersection_width = max(0, min(ax1, bx1) - max(ax0, bx0))
    intersection_height = max(0, min(ay1, by1) - max(ay0, by0))
    intersection = intersection_width * intersection_height
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - intersection
    return float(intersection / union)


@dataclass(frozen=True)
class PairAgreement:
    dice: float
    mask_iou: float
    bbox_iou: float
    foreground_pixels_a: int
    foreground_pixels_b: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compare_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> PairAgreement:
    """Compute all pairwise agreement measures used by the pilot QC report."""
    a = _binary_mask(mask_a)
    b = _binary_mask(mask_b)
    if a.shape != b.shape:
        raise ValueError(f"mask shape mismatch: {a.shape} vs {b.shape}")
    return PairAgreement(
        dice=mask_dice(a, b),
        mask_iou=mask_iou(a, b),
        bbox_iou=bbox_iou(tight_bbox(a), tight_bbox(b)),
        foreground_pixels_a=int(a.sum()),
        foreground_pixels_b=int(b.sum()),
    )
