#!/usr/bin/env python3
"""Build deterministic IAD JSONL/CSV manifests and a duplicate audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataio.iad import (  # noqa: E402
    IADDataError,
    audit_iad_manifest,
    build_iad_manifest,
    discover_iad_sources,
    write_manifest_audit,
    write_manifest_csv,
    write_manifest_jsonl,
)


OUTPUT_NAMES = (
    "iad_manifest.jsonl",
    "iad_manifest.csv",
    "iad_manifest_audit.json",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build stable, content-hashed manifests for exact "
            "'*_Train_Test.pkl' IAD files. '.bak' files are excluded."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="Directory containing the original read-only IAD pickle files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "Directory for iad_manifest.jsonl, iad_manifest.csv, and "
            "iad_manifest_audit.json."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing manifest outputs.",
    )
    parser.add_argument(
        "--fail-on-leakage",
        action="store_true",
        help=(
            "Return exit code 3 after writing outputs when exact content "
            "appears in both train and test splits."
        ),
    )
    return parser


def _output_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    return tuple(output_dir / name for name in OUTPUT_NAMES)  # type: ignore[return-value]


def _check_outputs(paths: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"output already exists: {joined}. Pass --overwrite to replace it."
        )
    for path in paths:
        if path.exists() and not path.is_file():
            raise IADDataError(f"output path exists but is not a file: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    jsonl_path, csv_path, audit_path = _output_paths(output_dir)

    try:
        sources = discover_iad_sources(args.dataset_dir)
        _check_outputs(
            (jsonl_path, csv_path, audit_path),
            overwrite=args.overwrite,
        )
        print(
            "IAD sources: " + ", ".join(source.path.name for source in sources),
            flush=True,
        )
        records = build_iad_manifest(
            args.dataset_dir,
            progress=lambda message: print(f"[iad-manifest] {message}", flush=True),
        )
        audit = audit_iad_manifest(records)
        write_manifest_jsonl(records, jsonl_path)
        write_manifest_csv(records, csv_path)
        write_manifest_audit(audit, audit_path)
    except (IADDataError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote {len(records)} records:")
    print(f"  JSONL: {jsonl_path}")
    print(f"  CSV:   {csv_path}")
    print(f"  audit: {audit_path}")
    print(
        "Audit: "
        f"{audit['unique_hash_count']} unique hashes, "
        f"{audit['duplicate_hash_group_count']} duplicate groups, "
        f"{audit['cross_file_duplicate_group_count']} cross-file groups, "
        f"{audit['train_test_leakage_group_count']} train-test leakage groups."
    )
    print(f"Dataset fingerprint: {audit['dataset_fingerprint']}")
    print(f"Manifest JSONL SHA-256: {audit['manifest_jsonl_sha256']}")

    leakage_count = int(audit["train_test_leakage_group_count"])
    if leakage_count:
        print(
            "WARNING: exact train-test content leakage was detected; "
            "see iad_manifest_audit.json.",
            file=sys.stderr,
        )
        if args.fail_on_leakage:
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
