"""
Stage 4 — event detection (per-ROI onsets + population density events).

For every ROI:
  * robust-z the SG-derivative trace
  * Schmitt-trigger hysteresis onsets at ``(z_enter, z_exit)`` with a minimum
    separation

Then, given the per-ROI onset lists, build a smoothed onset-density signal
and run the peak/baseline-walk/Gaussian-fit machinery in
:func:`core.events.detect_event_windows` to produce population event windows.

Outputs to ``<plane0>/<prefix>events/``:
  * ``onsets_by_roi.npz``   — per-ROI onset arrays (ragged, stored in an npz)
  * ``event_windows.npy``   — ``(E, 2)`` float32 ``[start_s, end_s]``
  * ``activation_matrix.npy`` — ``(N, E)`` bool; any onset in window
  * ``first_time.npy``      — ``(N, E)`` float32; earliest onset per window
  * ``event_diagnostics.npz`` — everything plotted by ``plot_event_detection``
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

from ..config.schema import PipelineConfig
from ..core.events import (
    EventDetectionParams,
    detect_event_windows,
    hysteresis_onsets,
    mad_z,
)
from ..io.memmap import open_s2p_memmaps
from ..io.recording import Recording
from ..orchestration.progress import (
    ProgressEvent,
    check_cancelled,
    emit,
    stage_scope,
)
from ._common import resolve_fps, resolve_recording


STAGE_NAME = "events"


def _event_params_from_config(params: dict) -> EventDetectionParams:
    """Copy relevant keys from the config dict into the dataclass."""
    fields = {
        "bin_sec", "smooth_sigma_bins", "normalize_by_num_rois",
        "min_prominence", "min_width_bins", "min_distance_bins",
        "baseline_mode", "baseline_percentile", "baseline_window_s",
        "noise_quiet_percentile", "noise_mad_factor",
        "end_threshold_k", "max_event_duration_s", "merge_gap_s",
        "use_gaussian_boundary", "gaussian_quantile",
        "gaussian_fit_pad_s", "gaussian_min_sigma_s",
    }
    return EventDetectionParams(**{k: params[k] for k in fields if k in params})


def run(
    recording: Union[Recording, str, Path],
    config: PipelineConfig,
    progress_callback: Optional[Callable] = None,
) -> dict:
    rec = resolve_recording(recording, config)
    fps = resolve_fps(rec, config)
    p = config.events

    # Use the filtered memmaps by default — they are what every other stage
    # consumes. If the filtered stage hasn't run yet, fall back to raw.
    prefix = config.signal_extraction["filtered_prefix"]
    try:
        dff, low, dt, T, N = open_s2p_memmaps(rec.plane0, prefix=prefix)
    except FileNotFoundError:
        prefix = config.signal_extraction["prefix"]
        dff, low, dt, T, N = open_s2p_memmaps(rec.plane0, prefix=prefix)

    out_dir = rec.plane0 / f"{prefix}events"
    out_dir.mkdir(exist_ok=True)

    with stage_scope(progress_callback, STAGE_NAME, total=N,
                     message=f"event detection on {rec.name} (N={N})"):
        t0 = time.monotonic()

        # ---- per-ROI hysteresis onsets ---------------------------------
        onsets_by_roi: list[np.ndarray] = []
        for j in range(N):
            check_cancelled(progress_callback)
            zj, _, _ = mad_z(dt[:, j])
            on_frames = hysteresis_onsets(
                zj,
                z_hi=p["z_enter"], z_lo=p["z_exit"],
                fps=fps, min_sep_s=p["min_sep_s"],
            )
            onsets_by_roi.append(on_frames.astype(np.float64) / fps)  # seconds

            if (j + 1) % max(1, N // 50) == 0:
                emit(progress_callback, ProgressEvent(
                    stage=STAGE_NAME,
                    message=f"per-ROI onsets {j+1}/{N}",
                    current=j + 1, total=N, kind="tick",
                ))

        # Save per-ROI onsets as an npz of ragged arrays.
        onsets_dict = {f"roi_{i}": a for i, a in enumerate(onsets_by_roi)}
        np.savez(out_dir / "onsets_by_roi.npz", **onsets_dict)

        # ---- population-density events ---------------------------------
        params = _event_params_from_config(p)
        event_windows, A, first_time, diagnostics = detect_event_windows(
            onsets_by_roi=onsets_by_roi,
            T=T, fps=fps, params=params,
            return_diagnostics=True,
        )

        np.save(out_dir / "event_windows.npy", event_windows.astype(np.float32))
        np.save(out_dir / "activation_matrix.npy", A)
        np.save(out_dir / "first_time.npy", first_time.astype(np.float32))
        np.savez(out_dir / "event_diagnostics.npz", **{
            k: np.asarray(v) for k, v in diagnostics.items()
        })

        emit(progress_callback, ProgressEvent(
            stage=STAGE_NAME, kind="log",
            message=f"{event_windows.shape[0]} population events detected",
        ))

        elapsed = time.monotonic() - t0

    return {
        "out_dir": out_dir,
        "n_events": int(event_windows.shape[0]),
        "event_windows": out_dir / "event_windows.npy",
        "activation_matrix": out_dir / "activation_matrix.npy",
        "first_time": out_dir / "first_time.npy",
        "elapsed_s": float(elapsed),
    }


__all__ = ["run", "STAGE_NAME"]
