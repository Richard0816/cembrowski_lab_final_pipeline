"""
Stage 7 — pairwise cross-correlations.

For every pair of ROIs active together within a coactivation bin (or within
any user-chosen window), compute the full lag cross-correlation (GPU via
cupy when available, NumPy fallback otherwise), a surrogate-shift null,
and Benjamini-Hochberg FDR-corrected significance.

Outputs to ``<plane0>/<prefix>xcorr/``:

* ``pairs.npy``             — ``(n_pairs, 2)`` int32 ROI index pairs
* ``lags_s.npy``            — ``(n_lags,)``    float32 lag grid in seconds
* ``xcorr.npy``             — ``(n_pairs, n_lags)`` float32
* ``zero_lag.npy``          — ``(n_pairs,)``   float32 zero-lag corr
* ``pval_shift.npy``        — ``(n_pairs,)``   float32 surrogate p-values
* ``qval_bh.npy``           — ``(n_pairs,)``   float32 BH-FDR q-values
* ``lag_argmax_s.npy``      — ``(n_pairs,)``   float32 peak-correlation lag

.. note::
   The numerical implementation (``compute_cross_correlation_gpu``,
   ``run_crosscorr_per_coactivation_bin_fast``, ``_bh_fdr``, etc.) will be
   ported from the legacy ``crosscorrelation.py`` in a follow-up commit. The
   run signature and output layout are locked here so downstream consumers
   and the GUI can integrate against them today.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

from ..config.schema import PipelineConfig
from ..io.recording import Recording
from ..orchestration.progress import (
    ProgressEvent,
    check_cancelled,
    emit,
    stage_scope,
)
from ._common import resolve_fps, resolve_recording


STAGE_NAME = "crosscorrelation"


def run(
    recording: Union[Recording, str, Path],
    config: PipelineConfig,
    progress_callback: Optional[Callable] = None,
) -> dict:
    rec = resolve_recording(recording, config)
    fps = resolve_fps(rec, config)
    p = config.crosscorrelation
    prefix = config.signal_extraction["filtered_prefix"]

    plane0 = rec.plane0
    out_dir = plane0 / f"{prefix}xcorr"
    out_dir.mkdir(exist_ok=True)

    with stage_scope(progress_callback, STAGE_NAME, total=0,
                     message=f"crosscorrelation for {rec.name}"):
        t0 = time.monotonic()

        emit(progress_callback, ProgressEvent(
            stage=STAGE_NAME, kind="log",
            message=(
                f"crosscorrelation port pending — legacy numerics live in "
                f"crosscorrelation.py (GPU/CPU, surrogate shift, BH-FDR). "
                f"Stub run() produced no artifacts."
            ),
        ))

        elapsed = time.monotonic() - t0

    return {
        "out_dir": out_dir,
        "elapsed_s": float(elapsed),
        "status": "stub",
    }


__all__ = ["run", "STAGE_NAME"]
