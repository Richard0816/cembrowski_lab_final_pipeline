"""
Stage 2 — cell filter inference.

Loads the trained dual-branch CNN checkpoint and scores every suite2p ROI,
writing:

* ``<plane0>/predicted_cell_prob.npy``   — float32 ``(N,)`` sigmoid scores
* ``<plane0>/predicted_cell_mask.npy``   — bool ``(N,)``  ``prob >= threshold``

Thin wrapper around :func:`cellfilter.predict.predict_recording`; we re-use
the existing GUI-friendly ``progress_cb`` parameter and adapt it into a
:class:`ProgressEvent` so the whole pipeline speaks one progress language.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional, Union

from ..cellfilter.predict import load_model, predict_recording
from ..config.schema import PipelineConfig
from ..io.recording import Recording
from ..orchestration.progress import (
    ProgressEvent,
    check_cancelled,
    emit,
    stage_scope,
)
from ._common import resolve_recording


STAGE_NAME = "cellfilter_infer"


def run(
    recording: Union[Recording, str, Path],
    config: PipelineConfig,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Score every ROI with the trained cell-filter CNN.

    The ``config.cellfilter`` dict is only consulted for the output names and
    threshold; model hyperparameters are baked into the checkpoint.
    """
    rec = resolve_recording(recording, config)
    ckpt_dir = Path(config.paths.cellfilter_ckpt_dir)
    ckpt_path = ckpt_dir / "best.pt"

    with stage_scope(progress_callback, STAGE_NAME, total=0,
                     message=f"cellfilter inference on {rec.name}"):
        t0 = time.monotonic()

        model, device = load_model(ckpt_path=ckpt_path)
        emit(progress_callback, ProgressEvent(
            stage=STAGE_NAME, kind="log",
            message=f"loaded checkpoint {ckpt_path.name} on {device}",
        ))

        # Adapt the cellfilter's (i, n, roi_prob) callback to a ProgressEvent.
        def _cb(i: int, n: int, p: float):
            check_cancelled(progress_callback)
            emit(progress_callback, ProgressEvent(
                stage=STAGE_NAME,
                message=f"ROI {i}/{n} p={p:.3f}",
                current=i, total=n, kind="tick",
                payload={"roi_prob": p},
            ))

        out_prob = predict_recording(
            rec_id=rec.name,
            model=model,
            device=device,
            progress_cb=_cb,
        )

        elapsed = time.monotonic() - t0

    return {
        "prob_path": out_prob,
        "mask_path": rec.predicted_mask_path,
        "elapsed_s": float(elapsed),
    }


__all__ = ["run", "STAGE_NAME"]
