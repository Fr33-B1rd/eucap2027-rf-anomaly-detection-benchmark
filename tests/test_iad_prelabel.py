from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch

from dataio.iad.manifest import hash_iad_sample
from iad_labeling.prelabel import PrelabelConfig, RobustCFARPrelabeler
from scripts.generate_iad_prelabels import main as generate_prelabels


def _normal_stack(
    seed: int = 7,
    count: int = 48,
    shape: tuple[int, int] = (64, 64),
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=1.0, scale=0.025, size=(count, *shape))
    return np.clip(data, 1e-4, None).astype(np.float32)


def test_constant_background_has_finite_calibration_and_empty_proposal():
    normal = np.full((12, 1, 32, 32), 2.0, dtype=np.float32)
    model = RobustCFARPrelabeler().fit(normal)

    result = model.prelabel(np.full((1, 32, 32), 2.0, dtype=np.float32))

    assert np.isfinite(result.score_map).all()
    assert np.all(model.background_scale >= model.config.mad_floor)
    assert result.binary_proposal.sum() == 0
    assert result.bbox is None


def test_rectangular_burst_produces_high_overlap_and_tight_bbox():
    normal = _normal_stack()
    config = PrelabelConfig(
        gaussian_sigma=0.4,
        global_score_threshold=5.0,
        strong_score_threshold=8.0,
        closing_iterations=1,
        min_component_area=8,
    )
    model = RobustCFARPrelabeler(config).fit(normal)
    image = normal[0].copy()
    expected = np.zeros((64, 64), dtype=bool)
    expected[18:34, 27:45] = True
    image[expected] += 2.5

    result = model.prelabel(image)
    proposal = result.binary_proposal.astype(bool)
    intersection = int((proposal & expected).sum())
    recall = intersection / int(expected.sum())
    precision = intersection / max(int(proposal.sum()), 1)

    assert recall > 0.9
    assert precision > 0.85
    assert result.bbox is not None
    assert abs(result.bbox["x0"] - 27) <= 1
    assert abs(result.bbox["x1"] - 45) <= 1
    assert abs(result.bbox["y0"] - 18) <= 1
    assert abs(result.bbox["y1"] - 34) <= 1


def test_diagonal_chirp_is_retained_by_ridge_rule():
    normal = _normal_stack(seed=11)
    config = PrelabelConfig(
        gaussian_sigma=0.25,
        global_score_threshold=5.0,
        strong_score_threshold=8.0,
        closing_iterations=0,
        fill_holes=False,
        min_component_area=100,
        min_ridge_length=20.0,
        min_ridge_linearity=0.95,
    )
    model = RobustCFARPrelabeler(config).fit(normal)
    image = normal[1].copy()
    expected = np.zeros((64, 64), dtype=bool)
    for col in range(10, 52):
        row = 8 + (col - 10)
        image[row, col] += 3.0
        expected[row, col] = True

    result = model.prelabel(image)
    proposal = result.binary_proposal.astype(bool)
    recall = int((proposal & expected).sum()) / int(expected.sum())
    retained = [
        component
        for component in result.diagnostics["components"]
        if component["retained"]
    ]

    assert recall > 0.9
    assert any(component["ridge_rule"] for component in retained)
    assert result.bbox is not None
    assert result.bbox["x0"] <= 10
    assert result.bbox["x1"] >= 52
    assert result.bbox["y0"] <= 8
    assert result.bbox["y1"] >= 50


def test_heldout_normal_has_no_anomaly_proposal():
    normal = _normal_stack(seed=19, count=96)
    config = PrelabelConfig(
        global_score_threshold=8.0,
        strong_score_threshold=12.0,
    )
    model = RobustCFARPrelabeler(config).fit(normal[:-1])

    result = model.prelabel(normal[-1])

    assert result.binary_proposal.sum() == 0
    assert result.bbox is None
    assert result.diagnostics["retained_component_count"] == 0


def test_cli_writes_independent_pilot_artifacts_without_modifying_pickle(
    tmp_path: Path,
):
    normal = _normal_stack(seed=31, count=10, shape=(24, 24))
    anomaly = normal[:2].copy()
    anomaly[:, 7:13, 9:16] += 2.0
    test_data = np.concatenate([anomaly, normal[2:5]], axis=0)[:, None]
    payload = {
        "train_data": torch.from_numpy(normal[:, None]),
        "train_label": np.zeros(10, dtype=np.float32),
        "test_data": torch.from_numpy(test_data),
        "test_label": np.asarray([1, 1, 0, 0, 0], dtype=np.float32),
    }
    pkl_path = tmp_path / "PILOT_Train_Test.pkl"
    with pkl_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    before = pkl_path.read_bytes()
    output_dir = tmp_path / "prelabels"

    exit_code = generate_prelabels(
        [
            "--dataset-pkl",
            str(pkl_path),
            "--output-dir",
            str(output_dir),
            "--pilot-count",
            "2",
            "--pilot-label",
            "anomaly",
            "--gaussian-sigma",
            "0.4",
            "--global-score-threshold",
            "5",
            "--strong-score-threshold",
            "8",
        ]
    )

    assert exit_code == 0
    assert pkl_path.read_bytes() == before
    run_manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["deep_model_used"] is False
    assert run_manifest["input_files_modified"] is False
    dataset_dir = output_dir / "PILOT"
    assert (dataset_dir / "normal_calibration.npz").is_file()
    assert (dataset_dir / "prelabels.jsonl").is_file()
    artifacts = sorted((dataset_dir / "samples").glob("*.npz"))
    assert len(artifacts) == 2
    with np.load(artifacts[0], allow_pickle=False) as artifact:
        assert artifact["score_map"].shape == (24, 24)
        assert artifact["binary_proposal"].shape == (24, 24)
        np.testing.assert_array_equal(
            artifact["proposal_mask"],
            artifact["binary_proposal"],
        )
        assert artifact["bbox_xyxy"].shape == (4,)


def test_pilot_manifest_uses_sample_id_and_normal_hash_disjoint_qc(
    tmp_path: Path,
):
    normal = _normal_stack(seed=41, count=12, shape=(24, 24))
    anomaly = normal[0].copy()
    anomaly[6:15, 8:18] += 2.5
    source_path = tmp_path / "QPSK_Train_Test.pkl"
    payload = {
        "train_data": torch.from_numpy(normal[:, None]),
        "train_label": np.zeros(12, dtype=np.int64),
        "test_data": torch.from_numpy(
            np.stack([anomaly, normal[1]], axis=0)[:, None]
        ),
        "test_label": np.asarray([1, 0], dtype=np.int64),
    }
    with source_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    source_bytes_before = source_path.read_bytes()
    sample_id, sample_hash = hash_iad_sample(payload["test_data"][0])
    pilot_path = tmp_path / "iad_pilot.jsonl"
    pilot_row = {
        "sample_id": sample_id,
        "hash": sample_hash,
        "modulation": "QPSK",
        "split": "test",
        "source_index": 0,
        "label": 1,
        "shape": [1, 24, 24],
        "source_path": str(source_path.resolve()),
    }
    pilot_path.write_text(
        json.dumps(pilot_row, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    pilot_bytes_before = pilot_path.read_bytes()
    output_dir = tmp_path / "pilot_prelabels"

    exit_code = generate_prelabels(
        [
            "--pilot-manifest",
            str(pilot_path),
            "--output-dir",
            str(output_dir),
            "--normal-calibration-fraction",
            "0.25",
            "--normal-split-seed",
            "73",
            "--gaussian-sigma",
            "0.4",
            "--global-score-threshold",
            "5",
            "--strong-score-threshold",
            "8",
        ]
    )

    assert exit_code == 0
    assert source_path.read_bytes() == source_bytes_before
    assert pilot_path.read_bytes() == pilot_bytes_before
    artifact_path = output_dir / "samples" / f"{sample_id}.npz"
    assert artifact_path.is_file()
    with np.load(artifact_path, allow_pickle=False) as artifact:
        np.testing.assert_array_equal(
            artifact["proposal_mask"],
            artifact["binary_proposal"],
        )
        metadata = json.loads(str(artifact["metadata_json"].item()))
        assert metadata["sample_id"] == sample_id
        assert metadata["hash"] == sample_hash
        assert metadata["source_index"] == 0
        assert metadata["modulation"] == "QPSK"
        assert metadata["source_content_verified"] is True
        assert metadata["config"]["score_mode"] == "positive"
        provenance = metadata["calibration_provenance"]
        assert provenance["normal_split"]["hash_group_overlap_count"] == 0
        assert (
            provenance["threshold_selection"]["anomaly_samples_used"]
            == 0
        )

    run_manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["mode"] == "canonical_pilot_manifest"
    assert run_manifest["pilot_sample_count"] == 1
    assert run_manifest["artifact_naming"] == "samples/<sample_id>.npz"
    source_summary = run_manifest["sources"][0]
    assert source_summary["normal_split"]["hash_group_overlap_count"] == 0
    qc = source_summary["heldout_normal_qc"]
    assert qc["evaluation_population"].endswith(
        "normal_train_hash_groups_only"
    )
    assert qc["heldout_normal_sample_count"] > 0
    assert 0.0 <= qc["proposal_nonempty_rate"] <= 1.0
    assert 0.0 <= qc["proposal_pixel_rate"] <= 1.0
    assert qc["threshold_tuning"]["anomaly_samples_used"] == 0


def test_normal_only_threshold_search_meets_target_without_anomaly_influence(
    tmp_path: Path,
):
    normal = _normal_stack(seed=53, count=20, shape=(24, 24))

    def run_variant(name: str, anomaly_amplitude: float) -> tuple[dict, dict]:
        variant_dir = tmp_path / name
        variant_dir.mkdir()
        anomaly = normal[0].copy()
        anomaly[5:18, 7:19] += anomaly_amplitude
        source_path = variant_dir / "QPSK_Train_Test.pkl"
        payload = {
            "train_data": torch.from_numpy(normal[:, None]),
            "train_label": np.zeros(20, dtype=np.int64),
            "test_data": torch.from_numpy(
                np.stack([anomaly, normal[1]], axis=0)[:, None]
            ),
            "test_label": np.asarray([1, 0], dtype=np.int64),
        }
        with source_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        sample_id, sample_hash = hash_iad_sample(payload["test_data"][0])
        pilot_path = variant_dir / "iad_pilot.jsonl"
        pilot_path.write_text(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "hash": sample_hash,
                    "modulation": "QPSK",
                    "split": "test",
                    "source_index": 0,
                    "label": 1,
                    "shape": [1, 24, 24],
                    "source_path": str(source_path.resolve()),
                },
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        output_dir = variant_dir / "prelabels"
        assert (
            generate_prelabels(
                [
                    "--pilot-manifest",
                    str(pilot_path),
                    "--output-dir",
                    str(output_dir),
                    "--normal-calibration-fraction",
                    "0.3",
                    "--normal-split-seed",
                    "91",
                    "--fit-max-samples",
                    "8",
                    "--target-normal-proposal-rate",
                    "0.2",
                    "--threshold-search-min",
                    "1",
                    "--threshold-search-step",
                    "1",
                    "--strong-threshold-margin",
                    "3",
                    "--gaussian-sigma",
                    "0.4",
                ]
            )
            == 0
        )
        run_manifest = json.loads(
            (output_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        with np.load(
            output_dir / "samples" / f"{sample_id}.npz",
            allow_pickle=False,
        ) as artifact:
            metadata = json.loads(str(artifact["metadata_json"].item()))
        return run_manifest["sources"][0], metadata

    weak_summary, weak_metadata = run_variant("weak", 0.5)
    strong_summary, strong_metadata = run_variant("strong", 1000.0)
    for source_summary, metadata in (
        (weak_summary, weak_metadata),
        (strong_summary, strong_metadata),
    ):
        selection = source_summary["threshold_selection"]
        assert selection["mode"] == "heldout_normal_target_rate_grid_search"
        assert selection["anomaly_samples_used"] == 0
        assert selection["achieved_normal_proposal_rate"] <= 0.2
        assert selection["trials"][-1]["meets_target"] is True
        assert all(
            trial["proposal_nonempty_rate"] > 0.2
            for trial in selection["trials"][:-1]
        )
        assert source_summary["normal_split"]["hash_group_overlap_count"] == 0
        assert (
            source_summary["heldout_normal_qc"]["proposal_nonempty_rate"]
            == selection["achieved_normal_proposal_rate"]
        )
        assert (
            metadata["config"]["global_score_threshold"]
            == selection["selected_global_score_threshold"]
        )
        assert (
            metadata["config"]["strong_score_threshold"]
            == selection["selected_strong_score_threshold"]
        )
        assert (
            metadata["calibration_provenance"]["threshold_selection"][
                "anomaly_samples_used"
            ]
            == 0
        )

    # The pilot anomaly changes by three orders of magnitude, while the
    # normal-only training/QC population is identical.
    assert (
        weak_summary["selected_config"]["global_score_threshold"]
        == strong_summary["selected_config"]["global_score_threshold"]
    )
    assert (
        weak_summary["selected_config"]["strong_score_threshold"]
        == strong_summary["selected_config"]["strong_score_threshold"]
    )
