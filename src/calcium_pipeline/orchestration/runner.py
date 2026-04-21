"""
``run_pipeline`` — apply a sequence of stages to one (or many) recordings.

Used by the CLI entrypoint and by the GUI's "Run" button. Honours:

* cancellation (via the shared :class:`ProgressBus`)
* per-stage error swallowing (``config.orchestration["skip_on_error"]``)
* per-stage elapsed-time accounting (returned in the manifest)
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

from ..config.schema import PipelineConfig
from ..io.recording import Recording
from ..stages import STAGE_ORDER, get_stage
from .progress import ProgressBus, ProgressEvent, emit, stage_scope


def run_pipeline(
    recording: Union[Recording, str, Path, Iterable],
    config: PipelineConfig,
    stages: Optional[Iterable[str]] = None,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Run ``stages`` (default: ``STAGE_ORDER``) on ``recording``.

    If ``recording`` is an iterable of recordings, runs the whole pipeline on
    each in turn, collecting per-recording manifests under
    ``result["per_recording"][rec_name]``.

    Parameters
    ----------
    recording : Recording | str | Path | iterable
        Single recording (path, id, or ``Recording``) OR an iterable of those.
    config : PipelineConfig
        Tunables. The orchestration sub-dict's ``skip_on_error`` toggles
        whether a failed stage aborts the whole pipeline.
    stages : iterable of str, optional
        Stage names (see :data:`stages.STAGE_ORDER`). Defaults to the ordered
        pipeline from :mod:`config.defaults`.
    progress_callback : callable | ProgressBus, optional
        Receives :class:`ProgressEvent` instances. Passed through to every
        stage so the GUI sees a single stream of events.

    Returns
    -------
    manifest : dict
        ``{"total_elapsed_s": float, "stages": {stage: manifest, ...},
           "per_recording": {...} if batched}``
    """
    if stages is None:
        stages = config.orchestration["default_pipeline"]
    stage_list = list(stages)

    skip_on_error = bool(config.orchestration.get("skip_on_error", False))

    # Batch mode: iterable of recordings.
    if _looks_like_batch(recording):
        manifest: dict = {"per_recording": {}, "total_elapsed_s": 0.0}
        t0 = time.monotonic()
        for rec in recording:
            name = getattr(rec, "name", str(rec))
            try:
                manifest["per_recording"][str(name)] = run_pipeline(
                    rec, config, stages=stage_list,
                    progress_callback=progress_callback,
                )
            except Exception as ex:  # noqa: BLE001
                emit(progress_callback, ProgressEvent(
                    stage="pipeline", kind="error",
                    message=f"{name}: {ex!r}",
                ))
                if not skip_on_error:
                    raise
        manifest["total_elapsed_s"] = time.monotonic() - t0
        return manifest

    # Single-recording mode.
    manifest = {"stages": {}, "total_elapsed_s": 0.0}
    t0 = time.monotonic()

    for stage_name in stage_list:
        run_fn = get_stage(stage_name)
        try:
            result = run_fn(recording, config, progress_callback=progress_callback)
            manifest["stages"][stage_name] = result
        except Exception as ex:  # noqa: BLE001
            tb = traceback.format_exc()
            emit(progress_callback, ProgressEvent(
                stage=stage_name, kind="error",
                message=f"{stage_name} failed: {ex!r}",
                payload={"traceback": tb},
            ))
            manifest["stages"][stage_name] = {"error": repr(ex), "traceback": tb}
            if not skip_on_error:
                break

    manifest["total_elapsed_s"] = time.monotonic() - t0
    return manifest


def _looks_like_batch(x) -> bool:
    """True iff ``x`` is iterable but not a ``str`` / ``Path`` / ``Recording``."""
    if isinstance(x, (str, Path, Recording)):
        return False
    try:
        iter(x)
        return True
    except TypeError:
        return False


__all__ = ["run_pipeline"]
