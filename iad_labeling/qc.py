"""Pilot annotation coverage, agreement, and adjudication reporting."""

from __future__ import annotations

import csv
import itertools
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

import numpy as np

from .agreement import compare_masks
from .schema import (
    AnnotationValidationError,
    validate_sample_hash,
    validate_sample_id,
    validate_sample_label,
    validate_stored_annotation,
)


FINAL_STATUSES = frozenset({"final", "uncertain", "unlocalizable"})
AGREEMENT_STATUS = "final"


@dataclass(frozen=True)
class AnnotationView:
    path: Path
    sample_id: str
    sample_hash: str
    sample_label: int
    annotator: str
    status: str
    confidence: float
    ambiguity_reasons: tuple[str, ...]
    mask: np.ndarray


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def load_pilot_index(path: str | Path) -> dict[str, dict]:
    pilot_path = Path(path).resolve()
    rows = _read_jsonl(pilot_path)
    index: dict[str, dict] = {}
    for row_number, row in enumerate(rows, start=1):
        try:
            sample_id = validate_sample_id(row.get("sample_id"))
            sample_hash = validate_sample_hash(row.get("hash"), sample_id)
            sample_label = validate_sample_label(row.get("label"))
        except AnnotationValidationError as error:
            raise ValueError(f"{pilot_path}: row {row_number}: {error}") from error
        if sample_id in index:
            raise ValueError(f"{pilot_path}: duplicate sample_id {sample_id}")
        canonical = dict(row)
        canonical.update(
            {
                "sample_id": sample_id,
                "hash": sample_hash,
                "label": sample_label,
            }
        )
        index[sample_id] = canonical
    if not index:
        raise ValueError(f"{pilot_path}: pilot manifest is empty")
    return index


def load_annotations(
    root: str | Path,
    *,
    pilot_ids: set[str] | None = None,
) -> list[AnnotationView]:
    annotation_root = Path(root).resolve()
    if not annotation_root.is_dir():
        raise FileNotFoundError(f"annotation directory does not exist: {annotation_root}")
    records: list[AnnotationView] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(annotation_root.glob("*/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: invalid JSON") from error
        try:
            payload = validate_stored_annotation(payload)
        except AnnotationValidationError as error:
            raise ValueError(f"{path}: {error}") from error
        sample_id = str(payload["sample_id"])
        annotator = str(payload["annotator"])
        status = str(payload["status"])
        if not sample_id or not annotator:
            raise ValueError(f"{path}: sample_id and annotator are required")
        if path.parent.name != annotator or path.stem != sample_id:
            raise ValueError(f"{path}: path does not match annotator/sample_id fields")
        if pilot_ids is not None and sample_id not in pilot_ids:
            raise ValueError(f"{path}: sample is not in the pilot manifest")
        key = (sample_id, annotator)
        if key in seen:
            raise ValueError(f"duplicate annotation for {annotator}/{sample_id}")
        seen.add(key)
        confidence = float(payload["confidence"])
        mask = np.asarray(payload["mask_64"])
        reasons = payload["ambiguity_reasons"]
        records.append(
            AnnotationView(
                path=path,
                sample_id=sample_id,
                sample_hash=payload["sample_hash"],
                sample_label=payload["sample_label"],
                annotator=annotator,
                status=status,
                confidence=confidence,
                ambiguity_reasons=tuple(reasons),
                mask=mask.astype(np.uint8, copy=False),
            )
        )
    return records


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return {"count": 0, "mean": None, "median": None, "minimum": None}
    return {
        "count": len(finite),
        "mean": float(np.mean(finite)),
        "median": float(median(finite)),
        "minimum": float(min(finite)),
    }


def build_qc_report(
    pilot_index: dict[str, dict],
    records: list[AnnotationView],
    *,
    expected_annotators: Iterable[str] = (),
    minimum_dice: float = 0.80,
    minimum_bbox_iou: float = 0.75,
) -> tuple[dict, list[dict]]:
    """Return a machine-readable summary and an adjudication queue.

    Agreement is computed only for ``final``/``final`` pairs.  Uncertain and
    unlocalizable decisions are retained in coverage and routed to
    adjudication rather than silently excluded.
    """
    if not 0.0 <= minimum_dice <= 1.0 or not 0.0 <= minimum_bbox_iou <= 1.0:
        raise ValueError("agreement thresholds must be in [0, 1]")
    if not pilot_index:
        raise ValueError("pilot index is empty")
    validated_pilot: dict[str, dict] = {}
    for index_sample_id, row in pilot_index.items():
        if not isinstance(row, dict):
            raise ValueError(f"pilot index entry {index_sample_id!r} must be an object")
        try:
            sample_id = validate_sample_id(row.get("sample_id"))
            sample_hash = validate_sample_hash(row.get("hash"), sample_id)
            sample_label = validate_sample_label(row.get("label"))
        except AnnotationValidationError as error:
            raise ValueError(
                f"pilot index entry {index_sample_id!r}: {error}"
            ) from error
        if index_sample_id != sample_id:
            raise ValueError(
                f"pilot index key {index_sample_id!r} does not match row sample_id "
                f"{sample_id!r}"
            )
        canonical = dict(row)
        canonical.update(
            {
                "sample_id": sample_id,
                "hash": sample_hash,
                "label": sample_label,
            }
        )
        validated_pilot[sample_id] = canonical
    pilot_index = validated_pilot

    grouped: dict[str, list[AnnotationView]] = defaultdict(list)
    for record in records:
        meta = pilot_index.get(record.sample_id)
        if meta is None:
            raise ValueError(f"{record.path}: sample is not in the pilot manifest")
        try:
            sample_hash = validate_sample_hash(record.sample_hash)
            sample_label = validate_sample_label(record.sample_label)
        except AnnotationValidationError as error:
            raise ValueError(f"{record.path}: {error}") from error
        if sample_hash != meta["hash"]:
            raise ValueError(
                f"{record.path}: annotation sample_hash does not match pilot hash"
            )
        if sample_label != meta["label"]:
            raise ValueError(
                f"{record.path}: annotation sample_label does not match pilot label"
            )
        grouped[record.sample_id].append(record)

    annotators = sorted(set(expected_annotators) | {record.annotator for record in records})
    coverage: dict[str, dict] = {}
    for annotator in annotators:
        own = [record for record in records if record.annotator == annotator]
        status_counts = Counter(record.status for record in own)
        completed = sum(status_counts[status] for status in FINAL_STATUSES)
        missing = sorted(set(pilot_index) - {record.sample_id for record in own})
        coverage[annotator] = {
            "records": len(own),
            "completed": completed,
            "completion_rate": completed / len(pilot_index),
            "status_counts": dict(sorted(status_counts.items())),
            "missing_count": len(missing),
            "missing_sample_ids": missing,
        }

    pair_rows: list[dict] = []
    queue: list[dict] = []
    for sample_id in sorted(pilot_index):
        sample_records = sorted(grouped.get(sample_id, []), key=lambda item: item.annotator)
        meta = pilot_index[sample_id]
        statuses = {record.annotator: record.status for record in sample_records}
        reasons = sorted({reason for record in sample_records for reason in record.ambiguity_reasons})
        if any(record.status in {"uncertain", "unlocalizable"} for record in sample_records):
            queue.append(
                {
                    "sample_id": sample_id,
                    "modulation": meta.get("modulation"),
                    "reason": "non_finalizable_status",
                    "annotators": ",".join(record.annotator for record in sample_records),
                    "statuses": json.dumps(statuses, sort_keys=True),
                    "dice": None,
                    "bbox_iou": None,
                    "ambiguity_reasons": "|".join(reasons),
                }
            )
        for left, right in itertools.combinations(sample_records, 2):
            if left.status != AGREEMENT_STATUS or right.status != AGREEMENT_STATUS:
                continue
            agreement = compare_masks(left.mask, right.mask)
            pair = {
                "sample_id": sample_id,
                "modulation": meta.get("modulation"),
                "annotator_a": left.annotator,
                "annotator_b": right.annotator,
                **agreement.to_dict(),
            }
            pair_rows.append(pair)
            if agreement.dice < minimum_dice or agreement.bbox_iou < minimum_bbox_iou:
                queue.append(
                    {
                        "sample_id": sample_id,
                        "modulation": meta.get("modulation"),
                        "reason": "low_agreement",
                        "annotators": f"{left.annotator},{right.annotator}",
                        "statuses": json.dumps(statuses, sort_keys=True),
                        "dice": agreement.dice,
                        "bbox_iou": agreement.bbox_iou,
                        "ambiguity_reasons": "|".join(reasons),
                    }
                )

    per_modulation: dict[str, dict] = {}
    modulations = sorted({str(row.get("modulation", "")) for row in pilot_index.values()})
    for modulation in modulations:
        selected = [row for row in pair_rows if str(row["modulation"]) == modulation]
        per_modulation[modulation] = {
            "dice": _summary(row["dice"] for row in selected),
            "mask_iou": _summary(row["mask_iou"] for row in selected),
            "bbox_iou": _summary(row["bbox_iou"] for row in selected),
        }

    report = {
        "protocol": {
            "pilot_samples": len(pilot_index),
            "agreement_status": AGREEMENT_STATUS,
            "minimum_dice": minimum_dice,
            "minimum_bbox_iou": minimum_bbox_iou,
            "note": (
                "Thresholds are preregistered protocol choices, not established "
                "physical constants."
            ),
        },
        "coverage": coverage,
        "agreement": {
            "pair_count": len(pair_rows),
            "overall": {
                "dice": _summary(row["dice"] for row in pair_rows),
                "mask_iou": _summary(row["mask_iou"] for row in pair_rows),
                "bbox_iou": _summary(row["bbox_iou"] for row in pair_rows),
            },
            "per_modulation": per_modulation,
            "pairs": pair_rows,
        },
        "adjudication": {
            "queue_count": len(queue),
            "sample_count": len({row["sample_id"] for row in queue}),
        },
    }
    return report, queue


def write_qc_outputs(
    report: dict,
    queue: list[dict],
    *,
    report_path: str | Path,
    queue_path: str | Path,
) -> None:
    json_path = Path(report_path)
    csv_path = Path(queue_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fieldnames = [
        "sample_id",
        "modulation",
        "reason",
        "annotators",
        "statuses",
        "dice",
        "bbox_iou",
        "ambiguity_reasons",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(queue)
