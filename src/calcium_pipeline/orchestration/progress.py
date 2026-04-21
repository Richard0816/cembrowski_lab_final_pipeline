"""
Progress bus — the one abstraction the GUI needs from the compute code.

Every stage's ``run(folder, config, progress_callback=None)`` calls
``progress_callback(event)`` (or ignores it if None). The callback is any
zero-dependency callable: a print function, a ``ProgressBus`` instance, or a
``pyqtSignal.emit``-like bound method.

Design goals
------------
* **No Qt import in stage code.** Stages only know about the ``ProgressEvent``
  dataclass and call the callback. The GUI wires a Qt signal to the bus.
* **Safe cancellation.** Stages call ``bus.check_cancelled()`` in their inner
  loops; the GUI cancel button flips a flag via ``bus.request_cancel()`` and
  the stage raises :class:`CancelledError` on the next check.
* **Nested stages.** When the orchestrator runs stages serially, it uses
  ``with bus.stage(name, total):`` to scope progress to one stage, and the
  outer listener sees ``(stage_name, frac_within_stage, overall_frac)``.
"""
from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


# --------------------------------------------------------------------------

class CancelledError(RuntimeError):
    """Raised by ``ProgressBus.check_cancelled`` when the GUI has cancelled."""


@dataclass
class ProgressEvent:
    """
    One tick of progress emitted by a stage.

    Attributes
    ----------
    stage : str
        The enclosing stage name (e.g. ``"signal_extraction"``).
    message : str
        Human-readable status line.
    current : int
        Number of units of work completed so far.
    total : int
        Total units of work in the current stage (``0`` means indeterminate).
    kind : str
        One of ``"start"``, ``"tick"``, ``"finish"``, ``"error"``, ``"log"``.
    payload : dict
        Stage-specific extras — e.g. per-ROI probability for the cellfilter,
        or a ``fig`` handle for plot-producing stages. GUIs can use this to
        refresh thumbnails mid-run.
    wall_time : float
        Seconds elapsed since the stage started (filled in by ``ProgressBus``).
    """
    stage: str
    message: str = ""
    current: int = 0
    total: int = 0
    kind: str = "tick"
    payload: dict = field(default_factory=dict)
    wall_time: float = 0.0

    @property
    def fraction(self) -> float:
        """Fraction of this stage complete in ``[0.0, 1.0]``, or 0 if indeterminate."""
        if self.total <= 0:
            return 0.0
        return min(1.0, max(0.0, self.current / self.total))


# --------------------------------------------------------------------------

Listener = Callable[[ProgressEvent], None]


class ProgressBus:
    """
    Fan-out listener registry with cancellation support.

    Usage (stage side)::

        def run(folder, cfg, progress_callback=None):
            bus = progress_callback or (lambda ev: None)
            with bus_stage(bus, "signal_extraction", total=N):
                for i in range(N):
                    bus_check(bus)          # may raise CancelledError
                    ...compute...
                    bus(ProgressEvent("signal_extraction", current=i+1, total=N))

    Usage (GUI side)::

        bus = ProgressBus()
        bus.subscribe(my_qt_signal.emit)
        bus.subscribe(console_widget.append_event)
        # pass bus itself as the progress_callback:
        run_pipeline(folder, cfg, progress_callback=bus)
    """

    def __init__(self):
        self._listeners: list[Listener] = []
        self._cancel_flag = threading.Event()
        self._stage_name: Optional[str] = None
        self._stage_start: float = 0.0

    # --- subscription ---------------------------------------------------

    def subscribe(self, fn: Listener) -> None:
        self._listeners.append(fn)

    def unsubscribe(self, fn: Listener) -> None:
        try:
            self._listeners.remove(fn)
        except ValueError:
            pass

    # --- emitting -------------------------------------------------------

    def __call__(self, event: ProgressEvent) -> None:
        """Forward ``event`` to every subscriber. Exceptions in a listener are
        logged but don't propagate — one broken listener can't kill a run."""
        if event.wall_time == 0.0 and self._stage_start:
            event.wall_time = time.monotonic() - self._stage_start
        for fn in list(self._listeners):
            try:
                fn(event)
            except Exception as ex:  # noqa: BLE001
                # Print rather than re-raise — GUI crashes shouldn't abort compute.
                print(f"[ProgressBus] listener {fn!r} raised {ex!r}")

    # --- cancellation ---------------------------------------------------

    def request_cancel(self) -> None:
        """Mark the run as cancelled. Stages see this on the next ``check_cancelled``."""
        self._cancel_flag.set()

    def clear_cancel(self) -> None:
        self._cancel_flag.clear()

    @property
    def cancelled(self) -> bool:
        return self._cancel_flag.is_set()

    def check_cancelled(self) -> None:
        """Raise :class:`CancelledError` if cancellation has been requested."""
        if self._cancel_flag.is_set():
            raise CancelledError(f"Cancelled during stage {self._stage_name!r}")

    # --- stage scoping --------------------------------------------------

    @contextlib.contextmanager
    def stage(self, name: str, total: int = 0, message: str = ""):
        """
        Context manager that emits ``start`` / ``finish`` bookends and times
        the stage. Nested stages are allowed but only the innermost is
        reflected in ``wall_time`` / ``stage_name``.
        """
        prev_name, prev_start = self._stage_name, self._stage_start
        self._stage_name = name
        self._stage_start = time.monotonic()
        try:
            self(ProgressEvent(stage=name, message=message or f"starting {name}",
                               current=0, total=total, kind="start"))
            yield
            self(ProgressEvent(stage=name, message=f"finished {name}",
                               current=total, total=total, kind="finish"))
        except CancelledError as ex:
            self(ProgressEvent(stage=name, message=str(ex), kind="error"))
            raise
        except Exception as ex:  # noqa: BLE001
            self(ProgressEvent(stage=name, message=f"{name} failed: {ex!r}",
                               kind="error", payload={"exception": ex}))
            raise
        finally:
            self._stage_name, self._stage_start = prev_name, prev_start


# --------------------------------------------------------------------------
# Convenience wrappers that work whether ``bus`` is a ProgressBus or a bare
# callable or None. Stage code should prefer these over ``isinstance`` checks.

def emit(bus, event: ProgressEvent) -> None:
    if bus is None:
        return
    bus(event)


def check_cancelled(bus) -> None:
    if bus is None:
        return
    if hasattr(bus, "check_cancelled"):
        bus.check_cancelled()


@contextlib.contextmanager
def stage_scope(bus, name: str, total: int = 0, message: str = ""):
    """``with stage_scope(bus, name, total): ...`` — works even if bus is None/lambda."""
    if bus is not None and hasattr(bus, "stage"):
        with bus.stage(name, total=total, message=message):
            yield
    else:
        # Fallback for plain callables: emit manually.
        start = time.monotonic()
        emit(bus, ProgressEvent(stage=name, message=message or f"starting {name}",
                                current=0, total=total, kind="start"))
        try:
            yield
            emit(bus, ProgressEvent(stage=name, message=f"finished {name}",
                                    current=total, total=total, kind="finish",
                                    wall_time=time.monotonic() - start))
        except Exception as ex:  # noqa: BLE001
            emit(bus, ProgressEvent(stage=name, message=f"{name} failed: {ex!r}",
                                    kind="error",
                                    wall_time=time.monotonic() - start))
            raise


__all__ = [
    "ProgressEvent",
    "ProgressBus",
    "CancelledError",
    "emit",
    "check_cancelled",
    "stage_scope",
]
