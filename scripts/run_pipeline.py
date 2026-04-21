"""CLI entry point — `python scripts/run_pipeline.py --recording PATH [--steps ...]`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_package_to_path() -> None:
    """Make `calcium_core` importable when run directly from source."""
    here = Path(__file__).resolve()
    src = here.parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> int:
    _add_package_to_path()
    from calcium_core.pipeline import (  # noqa: E402
        DEFAULT_STEPS,
        PrintReporter,
        list_steps,
        run_pipeline,
        run_pipeline_on_batch,
    )

    parser = argparse.ArgumentParser(description="Run the calcium_core pipeline.")
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--recording", type=str, help="Path to a single recording directory.")
    source.add_argument("--batch", type=str, help="Parent directory — run every subfolder.")
    parser.add_argument(
        "--steps", nargs="+", metavar="STEP", default=list(DEFAULT_STEPS),
        help=f"Steps to run (default: {list(DEFAULT_STEPS)}).",
    )
    parser.add_argument("--list-steps", action="store_true", help="List available steps and exit.")
    args = parser.parse_args()

    if args.list_steps:
        for name in list_steps():
            print(name)
        return 0

    if not (args.recording or args.batch):
        parser.error("one of --recording or --batch is required")

    reporter = PrintReporter()
    if args.recording:
        run_pipeline(args.recording, steps=args.steps, progress=reporter)
    else:
        run_pipeline_on_batch(args.batch, steps=args.steps, progress=reporter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
