"""Pipeline runner — the one entry point the app / CLI / scripts call."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

from .progress import PrintReporter, ProgressReporter
from .steps import DEFAULT_STEPS, run_step


def run_pipeline(
    recording_path: str | Path,
    steps: Optional[Sequence[str]] = None,
    progress: Optional[ProgressReporter] = None,
) -> None:
    """Run the named steps on a single recording.

    Parameters
    ----------
    recording_path : path to `YYYY-MM-DD_#####` directory
    steps : sequence of step names (see `pipeline.steps.list_steps()`).
            Defaults to `DEFAULT_STEPS`.
    progress : optional reporter. Defaults to a per-line print reporter.
    """
    recording_path = str(recording_path)
    steps = list(steps) if steps else list(DEFAULT_STEPS)
    reporter = progress or PrintReporter()

    for i, name in enumerate(steps):
        reporter.report("pipeline", i / len(steps), f"starting step {name!r}")
        run_step(name, recording_path, reporter)
    reporter.report("pipeline", 1.0, f"completed {len(steps)} step(s)")


def run_pipeline_on_batch(
    parent_folder: str | Path,
    steps: Optional[Sequence[str]] = None,
    progress: Optional[ProgressReporter] = None,
) -> None:
    """Run `run_pipeline` on every immediate subfolder of `parent_folder`."""
    parent = Path(parent_folder)
    subfolders: Iterable[Path] = (p for p in parent.iterdir() if p.is_dir())
    for p in subfolders:
        run_pipeline(p, steps=steps, progress=progress)
