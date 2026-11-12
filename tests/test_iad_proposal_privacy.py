from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from iad_labeling.webapp import create_app


SAMPLE_DIGEST = "a" * 64
SAMPLE_ID = f"iad-sha256-{SAMPLE_DIGEST}"
SAMPLE_HASH = f"sha256:{SAMPLE_DIGEST}"
OTHER_SAMPLE_ID = f"iad-sha256-{'b' * 64}"


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _make_app(tmp_path: Path):
    source = tmp_path / "QPSK_Train_Test.pkl"
    # Proposal requests do not deserialize the source. The file only satisfies
    # manifest path validation in this isolated API privacy test.
    source.write_bytes(b"unused by proposal endpoint")
    manifest = tmp_path / "pilot.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": SAMPLE_ID,
                "hash": SAMPLE_HASH,
                "modulation": "QPSK",
                "split": "test",
                "source_index": 0,
                "label": 1,
                "shape": [1, 64, 64],
                "source_path": str(source),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    proposals = tmp_path / "proposals"
    proposals.mkdir()
    app = create_app(
        manifest_path=manifest,
        proposals_dir=proposals,
        annotations_dir=tmp_path / "annotations",
        annotator_id="privacy-reviewer",
        source_root=tmp_path,
        testing=True,
    )
    return app, proposals


def _write_artifact(
    proposals: Path,
    *,
    metadata_sample_id: str = SAMPLE_ID,
) -> Path:
    proposal_mask = np.zeros((64, 64), dtype=np.uint8)
    proposal_mask[5:9, 11:14] = 1
    embedded_metadata = {
        "sample_id": metadata_sample_id,
        "diagnostics": {
            "score_map": [["EMBEDDED_SCORE_SENTINEL_73E1"]],
            "nested": {"heatmap": "EMBEDDED_HEATMAP_SENTINEL_8C4A"},
        },
    }
    artifact = proposals / f"{SAMPLE_ID}.npz"
    np.savez_compressed(
        artifact,
        proposal_mask=proposal_mask,
        score_map=np.full((64, 64), 0.25, dtype=np.float32),
        bbox_xyxy=np.asarray([11, 5, 14, 9], dtype=np.int32),
        metadata_json=np.asarray(json.dumps(embedded_metadata)),
    )
    return artifact


def _write_sidecar(
    proposals: Path,
    *,
    sidecar_sample_id: str = SAMPLE_ID,
) -> Path:
    sidecar = proposals / f"{SAMPLE_ID}.json"
    sidecar.write_text(
        json.dumps(
            {
                "sample_id": sidecar_sample_id,
                "private": {
                    "score_values": ["SIDECAR_SCORE_SENTINEL_D59B"],
                    "nested": {"heatmap": "SIDECAR_HEATMAP_SENTINEL_F204"},
                },
            }
        ),
        encoding="utf-8",
    )
    return sidecar


def test_proposal_api_returns_only_allowlisted_audit_provenance(
    tmp_path: Path,
) -> None:
    app, proposals = _make_app(tmp_path)
    artifact = _write_artifact(proposals)
    sidecar = _write_sidecar(proposals)

    response = app.test_client().get(f"/api/samples/{SAMPLE_ID}/proposal")

    assert response.status_code == 200
    document = response.get_json()
    provenance = document["proposal"]["provenance"]
    assert provenance == {
        "artifact_file": artifact.name,
        "artifact_sha256": _sha256(artifact),
        "displayed_component": "proposal_mask",
        "heatmap_values_exposed_to_ui": False,
        "sidecar_file": sidecar.name,
        "sidecar_sha256": _sha256(sidecar),
    }
    serialized = response.get_data(as_text=True)
    for forbidden_marker in (
        "EMBEDDED_SCORE_SENTINEL_73E1",
        "EMBEDDED_HEATMAP_SENTINEL_8C4A",
        "SIDECAR_SCORE_SENTINEL_D59B",
        "SIDECAR_HEATMAP_SENTINEL_F204",
        "embedded_metadata",
        '"sidecar":',
    ):
        assert forbidden_marker not in serialized


@pytest.mark.parametrize("mismatch_location", ("metadata", "sidecar"))
def test_proposal_api_still_rejects_mismatched_provenance_sample_id(
    tmp_path: Path,
    mismatch_location: str,
) -> None:
    app, proposals = _make_app(tmp_path)
    _write_artifact(
        proposals,
        metadata_sample_id=(
            OTHER_SAMPLE_ID if mismatch_location == "metadata" else SAMPLE_ID
        ),
    )
    _write_sidecar(
        proposals,
        sidecar_sample_id=(
            OTHER_SAMPLE_ID if mismatch_location == "sidecar" else SAMPLE_ID
        ),
    )

    response = app.test_client().get(f"/api/samples/{SAMPLE_ID}/proposal")

    assert response.status_code == 422
    assert "sample_id mismatch" in response.get_json()["error"]
