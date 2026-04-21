"""
Stage 1 — signal extraction.

Reads suite2p's ``F.npy`` / ``Fneu.npy`` and writes three float32 memmaps
shaped ``(T, N)`` under the configured prefix (default ``r0p7_``):

* ``<prefix>dff.memmap.float32``          — rolling-percentile ΔF/F
* ``<prefix>dff_lowpass.memmap.float32``  — causal Butterworth low-pass
* ``<prefix>dff_dt.memmap.float32``       — Savitzky-Golay first derivative

Batches over ROIs to cap RAM; emits one progress tick per batch.

Deliberately does *not* filter by cell-score. The dual-branch CNN in
:mod:`stages.cellfilter_infer` does that downstream.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

from ..config.schema import PipelineConfig
from ..core.signal import (
    robust_df_over_f_1d,
    lowpass_causal_1d,
    sg_first_derivative_1d,
)
from ..io.memmap import (
    change_batch_according_to_free_ram,
    s2p_load_raw,
)
from ..io.recording import Recording
from ..orchestration.progress import (
    ProgressEvent,
    check_cancelled,
    emit,
    stage_scope,
)
from ._common import resolve_fps, resolve_recording


STAGE_NAME = "signal_extraction"


def run(
    recording: Union[Recording, str, Path],
    config: PipelineConfig,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Compute ΔF/F, low-pass, and derivative memmaps for one recording.

    Returns
    -------
    dict
        ``{"dff": Path, "lowpass": Path, "dt": Path, "T": int, "N": int,
          "elapsed_s": float}``
    """
    rec = resolve_recording(recording, config)
    params = config.signal_extraction
    fps = resolve_fps(rec, config)

    prefix = params["prefix"]
    r = params["neuropil_factor"]
    win_sec = params["baseline_window_sec"]
    perc = params["baseline_percentile"]
    cutoff_hz = params["lowpass_cutoff_hz"]
    order = params["lowpass_order"]
    sg_win_ms = params["sg_window_ms"]
    sg_poly = params["sg_poly"]

    plane0 = rec.plane0

    with stage_scope(progress_callback, STAGE_NAME, total=0,
                     message=f"signal extraction on {rec.name}"):
        t0 = time.monotonic()

        F_cell, F_neu, num_frames, num_rois, time_major = s2p_load_raw(plane0)
        # Standardise to (T, N).
        if not time_major:
            F_cell = F_cell.T.astype(np.float32, copy=False)
            F_neu = F_neu.T.astype(np.float32, copy=False)

        emit(progress_callback, ProgressEvent(
            stage=STAGE_NAME,
            message=f"loaded F/Fneu: T={num_frames}, N={num_rois}, fps={fps:.3f}",
            current=0, total=num_rois, kind="log",
        ))

        # Allocate output memmaps.
        dff_path = plane0 / f"{prefix}dff.memmap.float32"
        low_path = plane0 / f"{prefix}dff_lowpass.memmap.float32"
        dt_path = plane0 / f"{prefix}dff_dt.memmap.float32"

        dff_mm = np.memmap(dff_path, mode="w+", dtype="float32",
                           shape=(num_frames, num_rois))
        low_mm = np.memmap(low_path, mode="w+", dtype="float32",
                           shape=(num_frames, num_rois))
        dt_mm = np.memmap(dt_path, mode="w+", dtype="float32",
                          shape=(num_frames, num_rois))

        batch_size = max(1, change_batch_according_to_free_ram() * 20)
        sos = None

        for batch_start in range(0, num_rois, batch_size):
            check_cancelled(progress_callback)
            batch_end = min(num_rois, batch_start + batch_size)

            F_cell_b = F_cell[:, batch_start:batch_end]
            F_neu_b = F_neu[:, batch_start:batch_end]
            corrected = (F_cell_b - r * F_neu_b).astype(np.float32)

            for j in range(corrected.shape[1]):
                trace = corrected[:, j]
                global_j = batch_start + j

                dff = robust_df_over_f_1d(
                    trace, win_sec=win_sec, perc=perc, fps=fps,
                )
                low, _, sos = lowpass_causal_1d(
                    dff, fps=fps, cutoff_hz=cutoff_hz, order=order,
                    zi=None, sos=sos,
                )
                deriv = sg_first_derivative_1d(
                    low, fps=fps, win_ms=sg_win_ms, poly=sg_poly,
                )

                dff_mm[:, global_j] = dff
                low_mm[:, global_j] = low
                dt_mm[:, global_j] = deriv

            dff_mm.flush()
            low_mm.flush()
            dt_mm.flush()

            emit(progress_callback, ProgressEvent(
                stage=STAGE_NAME,
                message=f"ROIs {batch_start}-{batch_end-1}/{num_rois}",
                current=batch_end, total=num_rois, kind="tick",
            ))

        # Release memmaps so they're readable by downstream stages.
        del dff_mm, low_mm, dt_mm

        elapsed = time.monotonic() - t0

    return {
        "dff": dff_path,
        "lowpass": low_path,
        "dt": dt_path,
        "T": int(num_frames),
        "N": int(num_rois),
        "fps": float(fps),
        "prefix": prefix,
        "elapsed_s": float(elapsed),
    }


__all__ = ["run", "STAGE_NAME"]
