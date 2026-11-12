"""Canonical schema and validation for IAD manual localization annotations."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import re
from typing import Any, Mapping

import numpy as np


SCHEMA_NAME = "iad.manual_localization"
SCHEMA_VERSION = "1.0.0"
MASK_SHAPE = (64, 64)
POOLED_MASK_SHAPE = (16, 16)
BBOX_FORMAT = "xyxy_exclusive"
ALLOWED_STATUSES = frozenset({"draft", "final", "uncertain", "unlocalizable"})
SUGGESTED_AMBIGUITY_REASONS = (
    "low_snr",
    "diffuse_boundary",
    "multiple_regions",
    "edge_truncation",
    "normal_or_artifact",
    "proposal_disagreement",
    "other",
)

_SAMPLE_ID_RE = re.compile(r"^iad-sha256-[0-9a-f]{64}$")
_SAMPLE_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ANNOTATOR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class AnnotationValidationError(ValueError):
    """Raised when an annotation or one of its identifiers is invalid."""


def utc_now() -> str:
    """Return a stable, timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def validate_sample_id(sample_id: Any) -> str:
    if not isinstance(sample_id, str) or not _SAMPLE_ID_RE.fullmatch(sample_id):
        raise AnnotationValidationError(
            "sample_id must match 'iad-sha256-' followed by 64 lowercase hex digits"
        )
    return sample_id


def validate_sample_hash(sample_hash: Any, sample_id: str | None = None) -> str:
    if not isinstance(sample_hash, str) or not _SAMPLE_HASH_RE.fullmatch(sample_hash):
        raise AnnotationValidationError(
            "sample_hash must match 'sha256:' followed by 64 lowercase hex digits"
        )
    if sample_id is not None:
        validate_sample_id(sample_id)
        if sample_hash.removeprefix("sha256:") != sample_id.removeprefix("iad-sha256-"):
            raise AnnotationValidationError("sample_hash does not match sample_id")
    return sample_hash


def validate_sample_label(sample_label: Any) -> int:
    if (
        isinstance(sample_label, bool)
        or not isinstance(sample_label, int)
        or sample_label not in (0, 1)
    ):
        raise AnnotationValidationError(
            "sample_label must be a non-boolean integer equal to 0 or 1"
        )
    return sample_label


def validate_annotator(annotator: Any) -> str:
    if not isinstance(annotator, str) or not _ANNOTATOR_RE.fullmatch(annotator):
        raise AnnotationValidationError(
            "annotator must be 1-64 characters using letters, digits, '.', '_' or '-'"
        )
    if annotator.upper().split(".", 1)[0] in _WINDOWS_RESERVED_NAMES:
        raise AnnotationValidationError("annotator is a reserved filesystem name")
    return annotator


def binary_mask(mask: Any, *, shape: tuple[int, int] = MASK_SHAPE) -> np.ndarray:
    """Validate a binary two-dimensional mask and return uint8."""
    array = np.asarray(mask)
    if array.shape != shape:
        raise AnnotationValidationError(
            f"mask must have shape {list(shape)}, got {list(array.shape)}"
        )
    if not (
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise AnnotationValidationError("mask must contain numeric binary values")
    try:
        finite = np.isfinite(array)
    except TypeError as exc:
        raise AnnotationValidationError("mask must contain numeric binary values") from exc
    if not bool(finite.all()):
        raise AnnotationValidationError("mask contains NaN or infinite values")
    if not bool(np.isin(array, (0, 1, False, True)).all()):
        raise AnnotationValidationError("mask values must be exactly 0 or 1")
    return array.astype(np.uint8, copy=False)


def tight_bbox(mask: Any) -> list[int] | None:
    """Return a tight [x0, y0, x1, y1] bbox with exclusive upper bounds."""
    array = binary_mask(mask)
    rows, columns = np.nonzero(array)
    if rows.size == 0:
        return None
    return [
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    ]


def max_pool_mask(mask: Any) -> np.ndarray:
    """Downsample a 64x64 mask to 16x16 using non-overlapping 4x4 max-pooling."""
    array = binary_mask(mask)
    return array.reshape(16, 4, 16, 4).max(axis=(1, 3)).astype(np.uint8)


def _confidence(value: Any) -> float:
    if isinstance(value, bool):
        raise AnnotationValidationError("confidence must be a number from 0 to 1")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AnnotationValidationError("confidence must be a number from 0 to 1") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise AnnotationValidationError("confidence must be a finite number from 0 to 1")
    return number


def _ambiguity_reasons(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise AnnotationValidationError("ambiguity_reasons must be a list of strings")
    if len(value) > 16:
        raise AnnotationValidationError("ambiguity_reasons may contain at most 16 entries")
    output: list[str] = []
    for reason in value:
        if not isinstance(reason, str):
            raise AnnotationValidationError("each ambiguity reason must be a string")
        reason = reason.strip()
        if not reason or len(reason) > 80 or any(ord(char) < 32 for char in reason):
            raise AnnotationValidationError(
                "ambiguity reasons must contain 1-80 printable characters"
            )
        if reason not in output:
            output.append(reason)
    return output


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    """Copy small JSON-compatible provenance values and reject exotic objects."""
    if depth > 8:
        raise AnnotationValidationError("proposal provenance is nested too deeply")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AnnotationValidationError("proposal provenance contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        if len(value) > 128:
            raise AnnotationValidationError("proposal provenance contains too many fields")
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 128:
                raise AnnotationValidationError(
                    "proposal provenance keys must be short strings"
                )
            output[key] = _json_safe(item, depth=depth + 1)
        return output
    if isinstance(value, (list, tuple)):
        if len(value) > 256:
            raise AnnotationValidationError("proposal provenance list is too long")
        return [_json_safe(item, depth=depth + 1) for item in value]
    raise AnnotationValidationError("proposal provenance must be JSON-compatible")


def build_annotation(
    payload: Mapping[str, Any],
    *,
    sample_id: str,
    sample_hash: str,
    sample_label: int,
    annotator: str,
    proposal_provenance: Mapping[str, Any] | None,
    created_at: str | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    """Build a canonical annotation, deriving every geometry field server-side."""
    if not isinstance(payload, Mapping):
        raise AnnotationValidationError("annotation payload must be a JSON object")
    sample_id = validate_sample_id(sample_id)
    sample_hash = validate_sample_hash(sample_hash, sample_id)
    sample_label = validate_sample_label(sample_label)
    annotator = validate_annotator(annotator)

    mask_value = payload.get("mask_64", payload.get("mask"))
    if mask_value is None:
        raise AnnotationValidationError("mask_64 is required")
    mask = binary_mask(mask_value)

    status = payload.get("status", "draft")
    if status not in ALLOWED_STATUSES:
        raise AnnotationValidationError(
            f"status must be one of {sorted(ALLOWED_STATUSES)}"
        )
    confidence = _confidence(payload.get("confidence", 0.5))
    reasons = _ambiguity_reasons(payload.get("ambiguity_reasons", []))
    proposal_visible = payload.get("proposal_visible_at_save", False)
    if not isinstance(proposal_visible, bool):
        raise AnnotationValidationError("proposal_visible_at_save must be boolean")
    foreground_pixels = int(mask.sum())
    if status == "final" and sample_label == 1 and foreground_pixels == 0:
        raise AnnotationValidationError(
            "a final anomaly annotation must contain foreground; use "
            "'unlocalizable' when no reliable region can be marked"
        )
    if status == "final" and sample_label == 0 and foreground_pixels != 0:
        raise AnnotationValidationError("a final normal annotation must have an empty mask")
    if status == "unlocalizable" and foreground_pixels != 0:
        raise AnnotationValidationError("an unlocalizable annotation must have an empty mask")

    timestamp = now or utc_now()
    created = created_at or timestamp
    provenance = (
        _json_safe(proposal_provenance) if proposal_provenance is not None else None
    )
    pooled = max_pool_mask(mask)
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "sample_hash": sample_hash,
        "sample_label": sample_label,
        "annotator": annotator,
        "created_at": created,
        "updated_at": timestamp,
        "status": status,
        "confidence": confidence,
        "ambiguity_reasons": reasons,
        "mask_shape": list(MASK_SHAPE),
        "mask_64": mask.tolist(),
        "bbox_format": BBOX_FORMAT,
        "bbox_xyxy": tight_bbox(mask),
        "pooled_mask_shape": list(POOLED_MASK_SHAPE),
        "pooling": "max_4x4",
        "mask_16": pooled.tolist(),
        "proposal_provenance": provenance,
        "proposal_visible_at_save": proposal_visible,
    }


def validate_stored_annotation(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a saved record, including all server-derived geometry fields."""
    if not isinstance(record, Mapping):
        raise AnnotationValidationError("stored annotation must be a JSON object")
    if record.get("schema_name") != SCHEMA_NAME:
        raise AnnotationValidationError("unsupported annotation schema_name")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise AnnotationValidationError("unsupported annotation schema_version")

    canonical = build_annotation(
        record,
        sample_id=record.get("sample_id"),
        sample_hash=record.get("sample_hash"),
        sample_label=record.get("sample_label"),
        annotator=record.get("annotator"),
        proposal_provenance=record.get("proposal_provenance"),
        created_at=record.get("created_at"),
        now=record.get("updated_at"),
    )
    for field in (
        "mask_shape",
        "bbox_format",
        "bbox_xyxy",
        "pooled_mask_shape",
        "pooling",
        "mask_16",
    ):
        if record.get(field) != canonical[field]:
            raise AnnotationValidationError(f"stored derived field is invalid: {field}")
    return canonical
