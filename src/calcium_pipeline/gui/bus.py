"""
Bridge between :class:`orchestration.progress.ProgressBus` and Qt's signal
system.

The compute code only knows about ``ProgressBus``; the GUI code only wants
``pyqtSignal``. This adapter owns both:

* It IS a ``ProgressBus`` (stages call it exactly as they would the plain bus)
* When the bus emits an event, the adapter ``emit``s a Qt signal that the
  main thread's widgets connect to.

The stages themselves run on a ``QThread``; the bus adapter's ``__call__``
is thread-safe because Qt signals queue cross-thread emissions.
"""
from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal

from ..orchestration.progress import ProgressBus, ProgressEvent


class QtProgressBus(QObject, ProgressBus):
    """
    Thread-safe ``ProgressBus`` whose events are mirrored to a Qt signal.

    Connect widgets like this::

        bus = QtProgressBus()
        bus.event.connect(self.progress_bar.update_from_event)
        bus.event.connect(self.console.append_event)
        ... pass `bus` as `progress_callback` to any stage ...
    """

    event = pyqtSignal(object)   # one arg: ProgressEvent

    def __init__(self, parent: QObject | None = None):
        QObject.__init__(self, parent)
        ProgressBus.__init__(self)

    def __call__(self, ev: ProgressEvent) -> None:      # type: ignore[override]
        # Fan out to plain-Python listeners first (so any non-Qt consumers
        # see the event), then re-emit as a Qt signal for the GUI.
        ProgressBus.__call__(self, ev)
        self.event.emit(ev)


__all__ = ["QtProgressBus"]
