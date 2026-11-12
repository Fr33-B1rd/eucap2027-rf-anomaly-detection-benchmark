#!/usr/bin/env python3
"""Export a deterministic, hash-grouped IAD anomaly annotation pilot."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataio.iad import (  # noqa: E402
    IADDataError,
    load_manifest,
    select_anomaly_pilot,
    write_manifest_csv,
    write_manifest_jsonl,
)


OUTPUT_NAMES = ("iad_pilot.jsonl", "iad_pilot.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select test/label=1 IAD samples for a pilot annotation set. "
            "Selection is deterministic and globally unique by content hash."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Input iad_manifest.jsonl or iad_manifest.csv.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for iad_pilot.jsonl and iad_pilot.csv.",
    )
    parser.add_argument(
        "--per-modulation",
        type=int,
        default=50,
        help="Unique anomalous test samples per modulation (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Fixed deterministic selection seed (default: 2026).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing pilot outputs.",
    )
    return parser


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
    jsonl_path = output_dir / OUTPUT_NAMES[0]
    csv_path = output_dir / OUTPUT_NAMES[1]

    try:
        _check_outputs(
            (jsonl_path, csv_path),
            overwrite=args.overwrite,
        )
        records = load_manifest(args.manifest)
        pilot = select_anomaly_pilot(
            records,
            per_modulation=args.per_modulation,
            seed=args.seed,
        )
        write_manifest_jsonl(pilot, jsonl_path)
        write_manifest_csv(pilot, csv_path)
    except (IADDataError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    counts = Counter(record.modulation for record in pilot)
    print(
        f"Wrote {len(pilot)} globally hash-unique anomaly records "
        f"with seed {args.seed}:"
    )
    for modulation, count in sorted(counts.items()):
        print(f"  {modulation}: {count}")
    print(f"  JSONL: {jsonl_path}")
    print(f"  CSV:   {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
