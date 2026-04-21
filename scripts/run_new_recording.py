"""End-to-end CLI: raw TIFF folder -> Suite2p -> full analysis pipeline.

Given a recording folder containing raw .tif files, this script:
    1. Runs Suite2p to produce suite2p/plane0/F.npy, stat.npy, ops.npy, ...
    2. Runs the full analysis chain:
         analyze -> heatmap -> image_all -> cluster -> correlate

The recording folder is expected to follow the YYYY-MM-DD_##### convention so
metadata lookups (AAV/FPS/zoom) resolve. Use --steps to run a subset.

Examples:
    python scripts/run_new_recording.py --recording D:/data/raw/2024-11-20_00003
    python scripts/run_new_recording.py --recording D:/data/raw/2024-11-20_00003 --steps analyze heatmap
    python scripts/run_new_recording.py --batch D:/data/raw/ --ops E:/suite2p_2p_ops_240621.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_package_to_path() -> None:
    """Make `calcium_core` importable when run from a checkout."""
    here = Path(__file__).resolve()
    src = here.parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> int:
    _add_package_to_path()

    from calcium_core.pipeline import (
        PrintReporter,
        list_steps,
        run_pipeline,
        run_pipeline_on_batch,
    )
    from calcium_core.pipeline.steps import FULL_STEPS

    parser = argparse.ArgumentParser(
        description="Raw TIFF -> Suite2p -> full calcium_core pipeline.",
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--recording", type=str, help="Path to a single recording folder (contains raw .tif files).")
    source.add_argument("--batch", type=str, help="Parent directory — run every subfolder.")
    parser.add_argument(
        "--steps",
        nargs="+",
        metavar="STEP",
        default=list(FULL_STEPS),
        help=f"Steps to run (default: {list(FULL_STEPS)}).",
    )
    parser.add_argument(
        "--ops",
        type=str,
        default=None,
        help="Override path to Suite2p ops.npy (applies to the 'suite2p' step).",
    )
    parser.add_argument("--list-steps", action="store_true", help="List available steps and exit.")
    args = parser.parse_args()

    if args.list_steps:
        for name in list_steps():
            print(name)
        return 0

    if not (args.recording or args.batch):
        parser.error("one of --recording or --batch is required")

    if args.ops is not None:
        import os
        os.environ["CALCIUM_CORE_SUITE2P_OPS"] = args.ops

    reporter = PrintReporter()
    if args.recording:
        run_pipeline(args.recording, steps=args.steps, progress=reporter)
    else:
        run_pipeline_on_batch(args.batch, steps=args.steps, progress=reporter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
