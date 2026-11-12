import numpy as np
import pytest

from iad_labeling.agreement import bbox_iou, compare_masks, mask_dice, mask_iou, tight_bbox


def test_identical_empty_masks_agree() -> None:
    empty = np.zeros((64, 64), dtype=np.uint8)
    assert mask_dice(empty, empty) == 1.0
    assert mask_iou(empty, empty) == 1.0
    assert tight_bbox(empty) is None
    assert bbox_iou(None, None) == 1.0


def test_tight_bbox_uses_exclusive_upper_bounds() -> None:
    mask = np.zeros((8, 9), dtype=np.uint8)
    mask[2:5, 3:7] = 1
    assert tight_bbox(mask) == (3, 2, 7, 5)


def test_pair_metrics_for_shifted_rectangles() -> None:
    left = np.zeros((8, 8), dtype=np.uint8)
    right = np.zeros((8, 8), dtype=np.uint8)
    left[1:5, 1:5] = 1
    right[1:5, 3:7] = 1
    result = compare_masks(left, right)
    assert result.dice == pytest.approx(0.5)
    assert result.mask_iou == pytest.approx(1 / 3)
    assert result.bbox_iou == pytest.approx(1 / 3)
    assert result.foreground_pixels_a == 16
    assert result.foreground_pixels_b == 16


def test_invalid_nonbinary_mask_is_rejected() -> None:
    with pytest.raises(ValueError, match="binary"):
        mask_dice(np.full((2, 2), 0.5), np.zeros((2, 2)))
