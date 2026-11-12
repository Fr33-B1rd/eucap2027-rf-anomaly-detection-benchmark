#!/usr/bin/env python
"""Run the IAD manual-localization UI on a local loopback interface."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from iad_labeling.webapp import create_app  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Serve a local 64x64 IAD spectrogram annotation UI. The browser "
            "receives binary proposals only, never proposal score maps."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Pilot manifest (.jsonl preferred; .csv/.json also supported).",
    )
    parser.add_argument(
        "--proposals-dir",
        type=Path,
        required=True,
        help="Directory containing independent <sample_id>.npz proposal files.",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        required=True,
        help="Output root; annotations are isolated under one directory per annotator.",
    )
    parser.add_argument(
        "--annotator",
        required=True,
        help=(
            "Reviewer identity locked for this server process. Start a separate "
            "process for each independent reviewer."
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Optional root that every manifest source_path must remain within.",
    )
    parser.add_argument(
        "--global-min",
        type=float,
        help="Optional fixed global contrast lower bound (requires --global-max).",
    )
    parser.add_argument(
        "--global-max",
        type=float,
        help="Optional fixed global contrast upper bound (requires --global-min).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1, local machine only).",
    )
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode; intended only for local development.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if (args.global_min is None) != (args.global_max is None):
        raise SystemExit("--global-min and --global-max must be supplied together")
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            "WARNING: a non-loopback --host exposes the annotation service to "
            "other machines; authentication is not provided.",
            file=sys.stderr,
        )

    app = create_app(
        manifest_path=args.manifest,
        proposals_dir=args.proposals_dir,
        annotations_dir=args.annotations_dir,
        annotator_id=args.annotator,
        source_root=args.source_root,
        global_min=args.global_min,
        global_max=args.global_max,
        testing=False,
    )
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=args.debug,
    )


if __name__ == "__main__":
    main()
