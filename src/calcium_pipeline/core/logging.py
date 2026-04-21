"""
stdout/stderr tee helpers.

Used both by the CLI (to mirror pipeline output into per-run log files) and by
the GUI (to pipe stage output into an on-screen console widget). Intentionally
Unicode-safe: the underlying handle may not be able to encode ``≥`` / ``Δ``,
so we fall back to byte-level writes with ``errors="replace"``.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional, Union


# --------------------------------------------------------------------------

class Tee(io.TextIOBase):
    """
    Fan-out text stream. ``write`` forwards each chunk to every underlying
    stream in turn and flushes. If a stream's encoding can't represent a
    character (e.g. a Windows console that is cp1252), we transparently encode
    with ``errors="replace"`` so the write never raises.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except UnicodeEncodeError:
                enc = getattr(s, "encoding", None) or "utf-8"
                if hasattr(s, "buffer"):
                    s.buffer.write(data.encode(enc, errors="replace"))
                    s.flush()
                else:
                    s.write(
                        data.encode(enc, errors="replace").decode(enc, errors="replace")
                    )
                    s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


# --------------------------------------------------------------------------

def run_with_logging(
    logfile_name: Union[str, Path],
    func: Callable,
    *args,
    **kwargs,
):
    """
    Call ``func(*args, **kwargs)`` with stdout + stderr mirrored into
    ``logfile_name`` (opened for append) as well as the real stdout.

    The logfile is always opened UTF-8 with ``errors="replace"`` so writes
    never fail on exotic characters.
    """
    with open(logfile_name, "a", encoding="utf-8", errors="replace") as logfile:
        tee = Tee(sys.__stdout__, logfile)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            return func(*args, **kwargs)


# --------------------------------------------------------------------------

def run_on_folders(
    parent_folder: Union[str, Path],
    func: Callable,
    log_filename: Union[str, Path],
    addon_vals: Optional[Iterable] = None,
    leaf_folders_only: bool = False,
) -> None:
    """
    Apply ``func`` to every immediate subfolder of ``parent_folder``, logging
    each call's stdout/stderr into ``log_filename`` (appended).

    Parameters
    ----------
    parent_folder : path
        Folder whose subfolders will be iterated.
    func : callable
        Called as ``func(entry.path)`` or ``func(entry.path, addon_vals)``.
    log_filename : path
        All output is appended here (per-entry, opening a fresh handle so a
        crash in one entry doesn't lose preceding logs).
    addon_vals : iterable, optional
        Extra positional argument forwarded to ``func``.
    leaf_folders_only : bool
        If True, skip any subfolder that itself contains subfolders.
    """
    for entry in os.scandir(parent_folder):
        if not entry.is_dir():
            continue

        has_subfolders = any(sub.is_dir() for sub in os.scandir(entry.path))
        if leaf_folders_only and has_subfolders:
            continue

        with open(log_filename, "a", encoding="utf-8", errors="replace") as logfile:
            tee = Tee(sys.__stdout__, logfile)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                if addon_vals:
                    func(entry.path, addon_vals)
                else:
                    func(entry.path)


__all__ = [
    "Tee",
    "run_with_logging",
    "run_on_folders",
]
