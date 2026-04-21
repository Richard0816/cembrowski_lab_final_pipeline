"""
calcium_pipeline.orchestration
------------------------------
Thin glue between the stages and whatever is driving them (CLI, GUI, notebook).

* :mod:`.progress` — ``ProgressBus`` (Qt-signal-compatible, but also works
  without Qt) for streaming stage progress to any number of listeners.
* :mod:`.runner`   — ``run_pipeline(...)`` applies a stage sequence to one or
  many recordings, with optional per-stage logging and failure handling.
* :mod:`.notify`   — SMTP email + desktop-notification helpers for long runs.
"""
from __future__ import annotations

__all__ = [
    "ProgressBus",
    "ProgressEvent",
    "run_pipeline",
    "send_email",
]


def __getattr__(name):
    if name in ("ProgressBus", "ProgressEvent"):
        from . import progress as _p
        return getattr(_p, name)
    if name == "run_pipeline":
        from .runner import run_pipeline
        return run_pipeline
    if name == "send_email":
        from .notify import send_email
        return send_email
    raise AttributeError(name)
