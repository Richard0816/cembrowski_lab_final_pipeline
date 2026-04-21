"""
Shared helpers for stage modules:

* :func:`resolve_recording` — turn whatever the caller passed (``Recording``,
  ``Path``, or raw ``str`` rec-id) into a :class:`Recording`.
* :func:`resolve_fps` — honour ``config.fps_override`` over notes lookup.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

from ..config.schema import PipelineConfig
from ..io.recording import Recording


def resolve_recording(
    recording: Union[Recording, str, Path],
    config: PipelineConfig,
) -> Recording:
    """
    Accepts:
      * a :class:`Recording` instance → returned as-is.
      * a :class:`Path` or ``str`` that points at a recording folder → wrapped.
      * a bare recording ID (``"2024-07-01_00018"``) → resolved via
        ``config.paths.recording_path``.
    """
    if isinstance(recording, Recording):
        return recording

    if isinstance(recording, (str, Path)):
        p = Path(recording)
        if p.exists():
            return Recording(
                p,
                notes_root=config.paths.notes_root,
                fps_override=config.fps_override,
                zoom_override=config.zoom_override,
            )
        # Treat as a rec_id.
        return Recording.from_id(
            str(recording),
            config.paths,
            fps_override=config.fps_override,
            zoom_override=config.zoom_override,
        )

    raise TypeError(f"Cannot resolve recording from {recording!r}")


def resolve_fps(recording: Recording, config: PipelineConfig) -> float:
    if config.fps_override is not None:
        return float(config.fps_override)
    return recording.fps


__all__ = ["resolve_recording", "resolve_fps"]
