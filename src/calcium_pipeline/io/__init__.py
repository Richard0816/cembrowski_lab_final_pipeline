"""
calcium_pipeline.io
-------------------
Low-level I/O primitives: suite2p file loading, memmaps, notes-xlsx lookup,
tif/gif export, artifact paths. Kept separate from ``core`` so GPU / torch /
matplotlib aren't pulled in when a caller only needs to open a recording.
"""
from __future__ import annotations

__all__ = [
    "open_s2p_memmaps",
    "s2p_load_raw",
    "s2p_infer_orientation",
    "change_batch_according_to_free_ram",
    "find_recording_root",
    "get_fps_from_notes",
    "get_zoom_from_notes",
    "aav_cleanup_and_dictionary_lookup",
    "file_name_to_aav_to_dictionary_lookup",
    "Recording",
]


def __getattr__(name):
    if name in (
        "open_s2p_memmaps",
        "s2p_load_raw",
        "s2p_infer_orientation",
        "change_batch_according_to_free_ram",
    ):
        from . import memmap as _m
        return getattr(_m, name)
    if name in (
        "find_recording_root",
        "get_fps_from_notes",
        "get_zoom_from_notes",
        "aav_cleanup_and_dictionary_lookup",
        "file_name_to_aav_to_dictionary_lookup",
    ):
        from . import notes_lookup as _n
        return getattr(_n, name)
    if name == "Recording":
        from .recording import Recording
        return Recording
    raise AttributeError(name)
