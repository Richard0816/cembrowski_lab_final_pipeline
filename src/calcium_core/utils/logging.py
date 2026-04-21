"""Stream multiplexing and run-with-logging helpers."""
from __future__ import annotations

import contextlib
import io
import sys
from typing import Callable


class Tee(io.TextIOBase):
    """Write to multiple streams; tolerates UnicodeEncodeError on any stream.

    If a stream's encoding cannot represent a character (e.g. ``>=`` on a
    cp1252 Windows console), the data is re-encoded with ``errors="replace"``
    and written via the stream's underlying byte buffer when available.
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
                    # Last resort: replace problematic chars and retry
                    s.write(data.encode(enc, errors="replace").decode(enc, errors="replace"))
                    s.flush()
        return len(data)


def run_with_logging(logfile_path: str, func: Callable, *args, **kwargs):
    """Run ``func(*args, **kwargs)`` with stdout/stderr mirrored to a log file.

    The log file is opened append-mode UTF-8 with ``errors="replace"`` so that
    any character produced by the wrapped function can always be recorded.
    """
    with open(logfile_path, "a", encoding="utf-8", errors="replace") as logfile:
        tee = Tee(sys.__stdout__, logfile)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            return func(*args, **kwargs)
