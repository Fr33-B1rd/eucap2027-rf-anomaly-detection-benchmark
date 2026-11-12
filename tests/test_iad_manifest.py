from __future__ import annotations

import json
import hashlib
import os
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from dataio.iad import (
    MANIFEST_FIELDS,
    IADDataError,
    UnsafeIADPickleError,
    audit_iad_manifest,
    build_iad_manifest,
    discover_iad_sources,
    hash_iad_sample,
    load_manifest,
    manifest_jsonl_sha256,
    safe_load_iad_pickle,
    select_anomaly_pilot,
    write_manifest_csv,
    write_manifest_jsonl,
)
from scripts.build_iad_manifest import main as build_manifest_main
from scripts.export_iad_pilot import main as export_pilot_main


def _sample(start: float) -> torch.Tensor:
    return torch.arange(start, start + 4, dtype=torch.float64).reshape(1, 2, 2)


def _write_iad_pickle(
    path: Path,
    *,
    train: list[torch.Tensor],
    test: list[torch.Tensor],
    test_labels: list[int],
    protocol: int = pickle.HIGHEST_PROTOCOL,
) -> None:
    payload = {
        "train_data": torch.stack(train),
        "train_label": np.zeros(len(train), dtype=np.float32),
        "test_data": torch.stack(test),
        "test_label": np.asarray(test_labels, dtype=np.float32),
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=protocol)


@pytest.fixture()
def synthetic_iad(tmp_path: Path) -> Path:
    alpha_train = [_sample(0), _sample(10)]
    alpha_test = [_sample(100), _sample(110), _sample(120), _sample(130)]
    _write_iad_pickle(
        tmp_path / "ALPHA_Train_Test.pkl",
        train=alpha_train,
        test=alpha_test,
        test_labels=[1, 1, 1, 0],
    )

    # Exact groups:
    # - beta train[0] == alpha train[1] (cross-file duplicate)
    # - beta test[0] == alpha test[0] (cross-file anomaly duplicate)
    # - beta test[3] == alpha train[0] (exact train-test leakage)
    beta_train = [_sample(10), _sample(20)]
    beta_test = [_sample(100), _sample(210), _sample(220), _sample(0)]
    _write_iad_pickle(
        tmp_path / "BETA_Train_Test.pkl",
        train=beta_train,
        test=beta_test,
        test_labels=[1, 1, 1, 0],
    )

    # These must never be discovered or deserialized.
    (tmp_path / "ALPHA_Train_Test.pkl.bak").write_bytes(b"not a pickle")
    (tmp_path / "unrelated.pkl").write_bytes(b"not a pickle")
    return tmp_path


def test_discovery_excludes_backup_and_nonmatching_files(
    synthetic_iad: Path,
) -> None:
    sources = discover_iad_sources(synthetic_iad)
    assert [source.modulation for source in sources] == ["ALPHA", "BETA"]
    assert [source.path.name for source in sources] == [
        "ALPHA_Train_Test.pkl",
        "BETA_Train_Test.pkl",
    ]
    assert all(not source.path.name.endswith(".bak") for source in sources)


def test_safe_loader_supports_current_and_legacy_pickle_protocols(
    tmp_path: Path,
) -> None:
    for protocol, modulation in (
        (pickle.HIGHEST_PROTOCOL, "CURRENT"),
        (3, "LEGACY"),
    ):
        path = tmp_path / f"{modulation}_Train_Test.pkl"
        _write_iad_pickle(
            path,
            train=[_sample(0), _sample(10)],
            test=[_sample(20), _sample(30)],
            test_labels=[1, 0],
            protocol=protocol,
        )
        bundle = safe_load_iad_pickle(path)
        assert bundle.modulation == modulation
        assert bundle.train_data.shape == (2, 1, 2, 2)
        assert bundle.test_data.dtype == torch.float64
        np.testing.assert_array_equal(bundle.train_labels, [0, 0])
        np.testing.assert_array_equal(bundle.test_labels, [1, 0])


def test_restricted_loader_denies_arbitrary_code(tmp_path: Path) -> None:
    marker = tmp_path / "pickle_executed.txt"

    class Exploit:
        def __reduce__(self):
            return (os.system, (f'echo unsafe > "{marker}"',))

    malicious = tmp_path / "EVIL_Train_Test.pkl"
    with malicious.open("wb") as handle:
        pickle.dump(Exploit(), handle)

    with pytest.raises(UnsafeIADPickleError, match="global denied"):
        safe_load_iad_pickle(malicious)
    assert not marker.exists()


def test_content_hash_is_stable_and_source_independent() -> None:
    sample = _sample(42)
    sample_id_1, hash_1 = hash_iad_sample(sample)
    sample_id_2, hash_2 = hash_iad_sample(sample.clone())
    assert sample_id_1 == sample_id_2
    assert hash_1 == hash_2
    assert sample_id_1 == f"iad-sha256-{hash_1.removeprefix('sha256:')}"

    changed = sample.clone()
    changed[0, 0, 0] += 1
    assert hash_iad_sample(changed)[1] != hash_1


def test_manifest_and_audit_detect_duplicates_and_leakage(
    synthetic_iad: Path,
) -> None:
    records = build_iad_manifest(synthetic_iad)
    assert len(records) == 12
    assert all(tuple(record.shape) == (1, 2, 2) for record in records)
    assert all(Path(record.source_path).is_absolute() for record in records)

    audit = audit_iad_manifest(records)
    assert audit["record_count"] == 12
    assert audit["manifest_schema_version"] == 1
    assert audit["sample_hash_schema"]["version"] == 1
    assert audit["manifest_jsonl_sha256"].startswith("sha256:")
    assert audit["dataset_fingerprint"].startswith("sha256:")
    assert len(audit["source_files"]) == 2
    assert all(
        source["file_sha256"].startswith("sha256:") for source in audit["source_files"]
    )
    assert audit["unique_hash_count"] == 9
    assert audit["duplicate_hash_group_count"] == 3
    assert audit["cross_file_duplicate_group_count"] == 3
    assert audit["train_test_leakage_group_count"] == 1

    leakage = audit["train_test_leakage_groups"][0]
    assert leakage["splits"] == ["test", "train"]
    assert {row["modulation"] for row in leakage["records"]} == {
        "ALPHA",
        "BETA",
    }


def test_jsonl_csv_round_trip_and_fixed_contract(
    synthetic_iad: Path,
    tmp_path: Path,
) -> None:
    records = build_iad_manifest(synthetic_iad)
    jsonl_path = write_manifest_jsonl(records, tmp_path / "manifest.jsonl")
    csv_path = write_manifest_csv(records, tmp_path / "manifest.csv")

    assert load_manifest(jsonl_path) == records
    assert load_manifest(csv_path) == records
    assert manifest_jsonl_sha256(records) == (
        "sha256:" + hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
    )

    first_json = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    assert tuple(first_json) == MANIFEST_FIELDS
    assert isinstance(first_json["shape"], list)

    csv_header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert csv_header.split(",") == list(MANIFEST_FIELDS)


def test_pilot_is_deterministic_balanced_and_hash_unique(
    synthetic_iad: Path,
) -> None:
    records = build_iad_manifest(synthetic_iad)
    first = select_anomaly_pilot(records, per_modulation=2, seed=2026)
    second = select_anomaly_pilot(
        reversed(records),
        per_modulation=2,
        seed=2026,
    )

    assert first == second
    assert Counter(row.modulation for row in first) == {
        "ALPHA": 2,
        "BETA": 2,
    }
    assert len({row.hash for row in first}) == 4
    assert all(row.split == "test" and row.label == 1 for row in first)

    with pytest.raises(IADDataError, match="cannot select"):
        select_anomaly_pilot(records, per_modulation=4, seed=2026)


def test_cli_writes_manifests_audit_and_pilot(
    synthetic_iad: Path,
    tmp_path: Path,
) -> None:
    manifest_dir = tmp_path / "outputs" / "manifest"
    result = build_manifest_main(
        [
            "--dataset-dir",
            str(synthetic_iad),
            "--output-dir",
            str(manifest_dir),
        ]
    )
    assert result == 0
    jsonl_path = manifest_dir / "iad_manifest.jsonl"
    assert jsonl_path.exists()
    assert (manifest_dir / "iad_manifest.csv").exists()
    audit = json.loads(
        (manifest_dir / "iad_manifest_audit.json").read_text(encoding="utf-8")
    )
    assert audit["train_test_leakage_group_count"] == 1
    assert audit["manifest_jsonl_sha256"] == (
        "sha256:" + hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
    )

    # Existing output is an explicit error unless --overwrite is supplied.
    assert (
        build_manifest_main(
            [
                "--dataset-dir",
                str(synthetic_iad),
                "--output-dir",
                str(manifest_dir),
            ]
        )
        == 2
    )

    pilot_dir = tmp_path / "outputs" / "pilot"
    result = export_pilot_main(
        [
            "--manifest",
            str(jsonl_path),
            "--output-dir",
            str(pilot_dir),
            "--per-modulation",
            "2",
            "--seed",
            "2026",
        ]
    )
    assert result == 0
    pilot = load_manifest(pilot_dir / "iad_pilot.jsonl")
    assert len(pilot) == 4
    assert len({row.sample_id for row in pilot}) == 4
    assert (pilot_dir / "iad_pilot.csv").exists()


def test_clear_error_for_inverse_original_labels(tmp_path: Path) -> None:
    path = tmp_path / "INVERSE_Train_Test.pkl"
    payload = {
        "train_data": torch.stack([_sample(0), _sample(10)]),
        "train_label": np.ones(2, dtype=np.float32),
        "test_data": torch.stack([_sample(20), _sample(30)]),
        "test_label": np.asarray([0, 1], dtype=np.float32),
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with pytest.raises(IADDataError, match="inverse-label pickle"):
        safe_load_iad_pickle(path)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_loader_rejects_nonfinite_spectrogram_values(
    tmp_path: Path,
    bad_value: float,
) -> None:
    path = tmp_path / "NONFINITE_Train_Test.pkl"
    train = [_sample(0), _sample(10)]
    train[0][0, 0, 0] = bad_value
    _write_iad_pickle(
        path,
        train=train,
        test=[_sample(20), _sample(30)],
        test_labels=[1, 0],
    )

    with pytest.raises(IADDataError, match="NaN or infinity"):
        safe_load_iad_pickle(path)
