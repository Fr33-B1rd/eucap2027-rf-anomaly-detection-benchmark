#!/usr/bin/env python
"""Create IAD-Loc pilot coverage, agreement, and adjudication reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from iad_labeling.qc import (  # noqa: E402
    build_qc_report,
    load_annotations,
    load_pilot_index,
    write_qc_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-annotator",
        action="append",
        default=[],
        help="Expected annotator ID; repeat for independent reviewers.",
    )
    parser.add_argument("--minimum-dice", type=float, default=0.80)
    parser.add_argument("--minimum-bbox-iou", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pilot = load_pilot_index(args.pilot_manifest)
    records = load_annotations(args.annotations_dir, pilot_ids=set(pilot))
    report, queue = build_qc_report(
        pilot,
        records,
        expected_annotators=args.expected_annotator,
        minimum_dice=args.minimum_dice,
        minimum_bbox_iou=args.minimum_bbox_iou,
    )
    write_qc_outputs(
        report,
        queue,
        report_path=args.output_dir / "pilot_qc_report.json",
        queue_path=args.output_dir / "adjudication_queue.csv",
    )
    print(json.dumps(report["adjudication"], sort_keys=True))


if __name__ == "__main__":
    main()
