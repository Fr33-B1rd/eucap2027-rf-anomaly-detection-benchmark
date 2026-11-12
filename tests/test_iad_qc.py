import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from iad_labeling.qc import build_qc_report, load_annotations, load_pilot_index
from iad_labeling.schema import (
    AnnotationValidationError,
    build_annotation,
    validate_stored_annotation,
)


def _payload(
    sample_id: str,
    annotator: str,
    mask: np.ndarray,
    status: str = "final",
    sample_label: int = 1,
) -> dict:
    digest = "a" * 64 if sample_id.endswith("a") else "b" * 64
    canonical_id = f"iad-sha256-{digest}"
    assert sample_id == canonical_id
    return build_annotation(
        {
            "status": status,
            "confidence": 0.9,
            "ambiguity_reasons": [],
            "mask_64": mask.astype(int).tolist(),
        },
        sample_id=sample_id,
        sample_hash=f"sha256:{digest}",
        sample_label=sample_label,
        annotator=annotator,
        proposal_provenance=None,
        now="2026-07-29T00:00:00Z",
    )


def test_qc_routes_low_agreement_and_uncertain_samples(tmp_path: Path) -> None:
    sample_a = f"iad-sha256-{'a' * 64}"
    sample_b = f"iad-sha256-{'b' * 64}"
    pilot_path = tmp_path / "pilot.jsonl"
    pilot_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_id": sample_a,
                        "hash": f"sha256:{'a' * 64}",
                        "label": 1,
                        "modulation": "QPSK",
                    }
                ),
                json.dumps(
                    {
                        "sample_id": sample_b,
                        "hash": f"sha256:{'b' * 64}",
                        "label": 1,
                        "modulation": "CHIRP",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    root = tmp_path / "annotations"
    for annotator in ("alice", "bob"):
        (root / annotator).mkdir(parents=True)

    mask_a = np.zeros((64, 64), dtype=np.uint8)
    mask_b = np.zeros((64, 64), dtype=np.uint8)
    mask_a[10:20, 10:20] = 1
    mask_b[10:20, 30:40] = 1
    (root / "alice" / f"{sample_a}.json").write_text(
        json.dumps(_payload(sample_a, "alice", mask_a)), encoding="utf-8"
    )
    (root / "bob" / f"{sample_a}.json").write_text(
        json.dumps(_payload(sample_a, "bob", mask_b)), encoding="utf-8"
    )
    uncertain = _payload(sample_b, "alice", np.zeros((64, 64)), status="uncertain")
    uncertain["ambiguity_reasons"] = ["overlap_with_background"]
    # Rebuild after changing a user-provided field so all derived fields remain valid.
    uncertain = build_annotation(
        uncertain,
        sample_id=sample_b,
        sample_hash=f"sha256:{'b' * 64}",
        sample_label=1,
        annotator="alice",
        proposal_provenance=None,
        now="2026-07-29T00:00:00Z",
    )
    (root / "alice" / f"{sample_b}.json").write_text(
        json.dumps(uncertain), encoding="utf-8"
    )

    pilot = load_pilot_index(pilot_path)
    records = load_annotations(root, pilot_ids=set(pilot))
    assert {
        (record.sample_id, record.sample_hash, record.sample_label) for record in records
    } == {
        (sample_a, f"sha256:{'a' * 64}", 1),
        (sample_b, f"sha256:{'b' * 64}", 1),
    }
    report, queue = build_qc_report(
        pilot,
        records,
        expected_annotators=["alice", "bob"],
        minimum_dice=0.8,
        minimum_bbox_iou=0.75,
    )

    assert report["coverage"]["alice"]["completed"] == 2
    assert report["coverage"]["bob"]["missing_count"] == 1
    assert report["agreement"]["pair_count"] == 1
    assert {row["reason"] for row in queue} == {"low_agreement", "non_finalizable_status"}


@pytest.mark.parametrize("invalid_label", [None, True, False])
def test_stored_annotation_rejects_null_or_boolean_label(invalid_label: object) -> None:
    sample_id = f"iad-sha256-{'a' * 64}"
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    record = _payload(sample_id, "alice", mask)
    record["sample_label"] = invalid_label

    with pytest.raises(AnnotationValidationError, match="sample_label"):
        validate_stored_annotation(record)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sample_id", "invalid-id", "sample_id"),
        ("hash", f"sha256:{'b' * 64}", "sample_hash does not match sample_id"),
        ("label", None, "sample_label"),
        ("label", True, "sample_label"),
    ],
)
def test_load_pilot_index_rejects_invalid_identity_fields(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    sample_id = f"iad-sha256-{'a' * 64}"
    row = {
        "sample_id": sample_id,
        "hash": f"sha256:{'a' * 64}",
        "label": 1,
    }
    row[field] = value
    pilot_path = tmp_path / "pilot.jsonl"
    pilot_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_pilot_index(pilot_path)


def test_build_qc_report_rejects_annotation_label_mismatch(tmp_path: Path) -> None:
    sample_id = f"iad-sha256-{'a' * 64}"
    sample_hash = f"sha256:{'a' * 64}"
    mask = np.zeros((64, 64), dtype=np.uint8)
    record = _payload(sample_id, "alice", mask, sample_label=0)
    annotation_root = tmp_path / "annotations" / "alice"
    annotation_root.mkdir(parents=True)
    annotation_path = annotation_root / f"{sample_id}.json"
    annotation_path.write_text(json.dumps(record), encoding="utf-8")
    records = load_annotations(annotation_root.parent)
    pilot = {
        sample_id: {
            "sample_id": sample_id,
            "hash": sample_hash,
            "label": 1,
        }
    }

    with pytest.raises(ValueError, match="sample_label does not match pilot label"):
        build_qc_report(pilot, records)


def test_build_qc_report_rejects_annotation_hash_mismatch(tmp_path: Path) -> None:
    sample_id = f"iad-sha256-{'a' * 64}"
    sample_hash = f"sha256:{'a' * 64}"
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    record = _payload(sample_id, "alice", mask)
    annotation_root = tmp_path / "annotations" / "alice"
    annotation_root.mkdir(parents=True)
    annotation_path = annotation_root / f"{sample_id}.json"
    annotation_path.write_text(json.dumps(record), encoding="utf-8")
    loaded = load_annotations(annotation_root.parent)
    tampered = replace(loaded[0], sample_hash=f"sha256:{'b' * 64}")
    pilot = {
        sample_id: {
            "sample_id": sample_id,
            "hash": sample_hash,
            "label": 1,
        }
    }

    with pytest.raises(ValueError, match="sample_hash does not match pilot hash"):
        build_qc_report(pilot, [tampered])
