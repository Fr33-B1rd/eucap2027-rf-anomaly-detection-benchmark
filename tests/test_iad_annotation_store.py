from __future__ import annotations

import io
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from dataio.iad import hash_iad_sample
from iad_labeling.schema import AnnotationValidationError
from iad_labeling.store import (
    AnnotationStore,
    ManifestValidationError,
    PilotManifest,
)
from iad_labeling.webapp import create_app


def _write_source(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    train = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    normal = torch.zeros((1, 64, 64), dtype=torch.float32)
    anomaly_a = torch.linspace(0.0, 1.0, 64 * 64, dtype=torch.float32).reshape(
        1, 64, 64
    )
    anomaly_b = torch.linspace(100.0, 101.0, 64 * 64, dtype=torch.float32).reshape(
        1, 64, 64
    )
    payload = {
        "train_data": train,
        "train_label": np.asarray([0], dtype=np.int64),
        "test_data": torch.stack((normal, anomaly_a, anomaly_b)),
        "test_label": np.asarray([0, 1, 1], dtype=np.int64),
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return anomaly_a, anomaly_b


def _manifest_row(source: Path, sample: torch.Tensor, index: int) -> dict:
    sample_id, sample_hash = hash_iad_sample(sample)
    return {
        "sample_id": sample_id,
        "hash": sample_hash,
        "modulation": "QPSK",
        "split": "test",
        "source_index": index,
        "label": 1,
        "shape": [1, 64, 64],
        "source_path": str(source.resolve()),
    }


@pytest.fixture()
def pilot_artifacts(tmp_path: Path) -> dict:
    source = tmp_path / "QPSK_Train_Test.pkl"
    anomaly_a, anomaly_b = _write_source(source)
    rows = (
        _manifest_row(source, anomaly_a, 1),
        _manifest_row(source, anomaly_b, 2),
    )
    manifest = tmp_path / "pilot.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )

    proposals = tmp_path / "proposals"
    proposals.mkdir()
    proposal_mask = np.zeros((64, 64), dtype=np.uint8)
    proposal_mask[8:12, 20:27] = 1
    metadata = {
        "sample_id": rows[0]["sample_id"],
        "method": "robust_cfar",
        "deep_model_used": False,
    }
    np.savez_compressed(
        proposals / f"{rows[0]['sample_id']}.npz",
        score_map=np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64),
        binary_proposal=proposal_mask,
        proposal_mask=proposal_mask,
        bbox_xyxy=np.asarray([20, 8, 27, 12], dtype=np.int32),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    return {
        "source": source,
        "rows": rows,
        "manifest": manifest,
        "proposals": proposals,
        "annotations": tmp_path / "annotations",
    }


def _mask(*, empty: bool = False) -> list[list[int]]:
    array = np.zeros((64, 64), dtype=np.uint8)
    if not empty:
        array[10:14, 11:19] = 1
    return array.tolist()


def _payload(annotator: str, *, status: str = "final", empty: bool = False) -> dict:
    return {
        "annotator": annotator,
        "mask_64": _mask(empty=empty),
        "status": status,
        "confidence": 0.8,
        "ambiguity_reasons": ["diffuse_boundary"],
        "proposal_visible_at_save": True,
    }


def test_store_derives_geometry_and_isolates_annotators(
    pilot_artifacts: dict,
) -> None:
    row = pilot_artifacts["rows"][0]
    store = AnnotationStore(pilot_artifacts["annotations"])
    provenance = {"method": "robust_cfar", "artifact_sha256": "sha256:" + "c" * 64}

    alice = store.save(
        sample_id=row["sample_id"],
        sample_hash=row["hash"],
        sample_label=1,
        annotator="alice",
        payload=_payload("alice"),
        proposal_provenance=provenance,
    )
    bob = store.save(
        sample_id=row["sample_id"],
        sample_hash=row["hash"],
        sample_label=1,
        annotator="bob",
        payload=_payload("bob", status="draft"),
        proposal_provenance=None,
    )

    assert alice["bbox_xyxy"] == [11, 10, 19, 14]
    assert alice["bbox_format"] == "xyxy_exclusive"
    assert np.asarray(alice["mask_64"]).shape == (64, 64)
    pooled = np.asarray(alice["mask_16"])
    assert pooled.shape == (16, 16)
    expected_pool = (
        np.asarray(alice["mask_64"], dtype=np.uint8)
        .reshape(16, 4, 16, 4)
        .max(axis=(1, 3))
    )
    np.testing.assert_array_equal(pooled, expected_pool)
    assert alice["sample_hash"] == row["hash"]
    assert alice["sample_label"] == 1
    assert alice["schema_name"] == "iad.manual_localization"
    assert alice["schema_version"] == "1.0.0"
    assert bob["annotator"] == "bob"
    assert store.path_for(row["sample_id"], "alice") != store.path_for(
        row["sample_id"], "bob"
    )
    assert store.load(row["sample_id"], "alice")["annotator"] == "alice"
    assert store.load(row["sample_id"], "bob")["annotator"] == "bob"
    assert not list(pilot_artifacts["annotations"].rglob("*.tmp"))

    first_created = alice["created_at"]
    updated = store.save(
        sample_id=row["sample_id"],
        sample_hash=row["hash"],
        sample_label=1,
        annotator="alice",
        payload=_payload("alice", status="uncertain"),
        proposal_provenance=provenance,
    )
    assert updated["created_at"] == first_created
    assert store.load(row["sample_id"], "bob")["status"] == "draft"


def test_store_rejects_bad_shapes_paths_and_semantically_invalid_masks(
    pilot_artifacts: dict,
) -> None:
    row = pilot_artifacts["rows"][0]
    store = AnnotationStore(pilot_artifacts["annotations"])
    bad_shape = _payload("alice")
    bad_shape["mask_64"] = [[0] * 64 for _ in range(63)]
    with pytest.raises(AnnotationValidationError, match="shape"):
        store.save(
            sample_id=row["sample_id"],
            sample_hash=row["hash"],
            sample_label=1,
            annotator="alice",
            payload=bad_shape,
            proposal_provenance=None,
        )
    with pytest.raises(AnnotationValidationError, match="final anomaly"):
        store.save(
            sample_id=row["sample_id"],
            sample_hash=row["hash"],
            sample_label=1,
            annotator="alice",
            payload=_payload("alice", empty=True),
            proposal_provenance=None,
        )
    with pytest.raises(AnnotationValidationError, match="unlocalizable"):
        store.save(
            sample_id=row["sample_id"],
            sample_hash=row["hash"],
            sample_label=1,
            annotator="alice",
            payload=_payload("alice", status="unlocalizable"),
            proposal_provenance=None,
        )
    with pytest.raises(AnnotationValidationError, match="annotator"):
        store.path_for(row["sample_id"], "../bob")
    with pytest.raises(AnnotationValidationError, match="sample_id"):
        store.path_for("../../outside", "alice")
    assert not (pilot_artifacts["annotations"].parent / "bob").exists()


def test_manifest_rejects_source_path_outside_configured_root(
    pilot_artifacts: dict, tmp_path: Path
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    with pytest.raises(ManifestValidationError, match="escapes source_root"):
        PilotManifest(pilot_artifacts["manifest"], source_root=allowed)


def test_api_renders_original_image_hides_score_map_and_saves_atomically(
    pilot_artifacts: dict,
) -> None:
    app = create_app(
        manifest_path=pilot_artifacts["manifest"],
        proposals_dir=pilot_artifacts["proposals"],
        annotations_dir=pilot_artifacts["annotations"],
        annotator_id="alice",
        source_root=pilot_artifacts["source"].parent,
        testing=True,
    )
    client = app.test_client()
    row = pilot_artifacts["rows"][0]
    sample_id = row["sample_id"]

    response = client.get("/api/samples?annotator=alice")
    assert response.status_code == 200
    assert response.get_json()["count"] == 2

    for contrast in ("global", "percentile"):
        response = client.get(
            f"/api/samples/{sample_id}/image?contrast={contrast}"
        )
        assert response.status_code == 200
        assert response.headers["X-IAD-Image-Shape"] == "64x64"
        assert response.headers["X-IAD-Contrast-Mode"] == contrast
        with Image.open(io.BytesIO(response.data)) as image:
            assert image.size == (64, 64)

    response = client.get(f"/api/samples/{sample_id}/proposal")
    assert response.status_code == 200
    proposal = response.get_json()["proposal"]
    assert np.asarray(proposal["mask_64"]).shape == (64, 64)
    assert proposal["bbox_xyxy"] == [20, 8, 27, 12]
    assert "score_map" not in proposal
    assert "score_map" not in response.get_data(as_text=True)
    assert proposal["provenance"]["heatmap_values_exposed_to_ui"] is False
    assert "embedded_metadata" not in proposal["provenance"]

    response = client.put(
        f"/api/samples/{sample_id}/annotation",
        json={
            **_payload("alice"),
            "sample_hash": "sha256:" + "f" * 64,
            "bbox_xyxy": [0, 0, 64, 64],
        },
    )
    assert response.status_code == 201
    annotation = response.get_json()["annotation"]
    assert annotation["sample_hash"] == row["hash"]
    assert annotation["bbox_xyxy"] == [11, 10, 19, 14]
    assert annotation["proposal_provenance"]["artifact_file"] == f"{sample_id}.npz"
    annotation_path = (
        pilot_artifacts["annotations"] / "alice" / f"{sample_id}.json"
    )
    assert annotation_path.is_file()
    assert not list(pilot_artifacts["annotations"].rglob("*.tmp"))

    response = client.get(
        f"/api/samples/{sample_id}/annotation?annotator=alice"
    )
    assert response.status_code == 200
    assert response.get_json()["annotation"]["annotator"] == "alice"


def test_api_rejects_traversal_bad_shape_and_invalid_final_status(
    pilot_artifacts: dict,
) -> None:
    app = create_app(
        manifest_path=pilot_artifacts["manifest"],
        proposals_dir=pilot_artifacts["proposals"],
        annotations_dir=pilot_artifacts["annotations"],
        annotator_id="alice",
        testing=True,
    )
    client = app.test_client()
    sample_id = pilot_artifacts["rows"][0]["sample_id"]

    bad_annotator = client.put(
        f"/api/samples/{sample_id}/annotation",
        json={**_payload("alice"), "annotator": "../bob"},
    )
    assert bad_annotator.status_code == 400

    bad_shape = _payload("alice")
    bad_shape["mask_64"] = [[0] * 64]
    response = client.put(
        f"/api/samples/{sample_id}/annotation", json=bad_shape
    )
    assert response.status_code == 400
    assert "shape" in response.get_json()["error"]

    response = client.put(
        f"/api/samples/{sample_id}/annotation",
        json=_payload("alice", empty=True),
    )
    assert response.status_code == 400
    assert "final anomaly" in response.get_json()["error"]

    response = client.get("/api/samples/%2E%2E%2Foutside/image")
    assert response.status_code in {400, 404}
    assert not (pilot_artifacts["annotations"].parent / "bob").exists()


def test_api_locks_reviewer_identity_and_hash_checks_source_on_put(
    pilot_artifacts: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(
        manifest_path=pilot_artifacts["manifest"],
        proposals_dir=pilot_artifacts["proposals"],
        annotations_dir=pilot_artifacts["annotations"],
        annotator_id="alice",
        source_root=pilot_artifacts["source"].parent,
        testing=True,
    )
    client = app.test_client()
    sample_id = pilot_artifacts["rows"][0]["sample_id"]

    response = client.get("/api/samples?annotator=bob")
    assert response.status_code == 403
    assert "locked" in response.get_json()["error"]
    response = client.put(
        f"/api/samples/{sample_id}/annotation",
        json=_payload("bob"),
    )
    assert response.status_code == 403
    assert not (pilot_artifacts["annotations"] / "bob").exists()

    manifest = app.extensions["iad_manifest"]

    def reject_source(_sample_id: str) -> np.ndarray:
        raise ManifestValidationError("source identity sentinel")

    monkeypatch.setattr(manifest, "load_array", reject_source)
    response = client.put(
        f"/api/samples/{sample_id}/annotation",
        json=_payload("alice"),
    )
    assert response.status_code == 400
    assert "source identity sentinel" in response.get_json()["error"]
    assert not (pilot_artifacts["annotations"] / "alice").exists()


def test_api_rejects_malformed_proposal_shape(pilot_artifacts: dict) -> None:
    row = pilot_artifacts["rows"][0]
    np.savez_compressed(
        pilot_artifacts["proposals"] / f"{row['sample_id']}.npz",
        proposal_mask=np.zeros((63, 64), dtype=np.uint8),
        score_map=np.zeros((64, 64), dtype=np.float32),
        bbox_xyxy=np.empty((0,), dtype=np.int32),
    )
    app = create_app(
        manifest_path=pilot_artifacts["manifest"],
        proposals_dir=pilot_artifacts["proposals"],
        annotations_dir=pilot_artifacts["annotations"],
        annotator_id="alice",
        testing=True,
    )
    response = app.test_client().get(
        f"/api/samples/{row['sample_id']}/proposal"
    )
    assert response.status_code == 422
    assert "shape" in response.get_json()["error"]
