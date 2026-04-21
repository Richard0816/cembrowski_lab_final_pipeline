"""Progress-reporting protocol — shared contract between library and any caller.

The library never imports a UI framework. Long-running steps accept an optional
`ProgressReporter`; callers plug in whatever they want (logging, tqdm, Qt
signal, web socket, ...).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ProgressReporter(Protocol):
    """Minimal contract for progress callbacks."""

    def report(self, phase: str, fraction: float, message: str = "") -> None: ...


@dataclass
class PrintReporter:
    """Default reporter for CLI usage — prints a single line per update."""

    def report(self, phase: str, fraction: float, message: str = "") -> None:
        pct = int(round(fraction * 100))
        extra = f" — {message}" if message else ""
        print(f"[{phase}] {pct:3d}%{extra}", flush=True)


class NullReporter:
    """No-op reporter."""

    def report(self, phase: str, fraction: float, message: str = "") -> None:
        return None
