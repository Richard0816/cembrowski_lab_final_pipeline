"""
``RunWorker`` — a ``QThread`` that executes ``orchestration.run_pipeline`` off
the GUI thread, routing all progress events through a :class:`QtProgressBus`.

Widgets should never call ``run_pipeline`` directly from the main thread; they
construct a ``RunWorker``, connect to the bus's ``event`` signal, and start
the thread. Cancellation is a single call: ``worker.request_cancel()``.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Union

from PyQt6.QtCore import QThread, pyqtSignal

from ..config.schema import PipelineConfig
from ..io.recording import Recording
from ..orchestration.runner import run_pipeline
from .bus import QtProgressBus


class RunWorker(QThread):
    """
    Runs the pipeline on a background thread and emits:

    * ``started_stages`` — list of stage names, at QThread start
    * ``finished_ok``    — manifest dict, on successful completion
    * ``failed``         — (str, str) (message, traceback) on exception
    """

    started_stages = pyqtSignal(list)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str, str)

    def __init__(
        self,
        recording: Union[Recording, str, Iterable],
        config: PipelineConfig,
        bus: QtProgressBus,
        stages: Optional[Iterable[str]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._recording = recording
        self._config = config
        self._bus = bus
        self._stages = list(stages) if stages is not None else None

    # --- public control ---------------------------------------------------

    def request_cancel(self) -> None:
        self._bus.request_cancel()

    # --- QThread hook -----------------------------------------------------

    def run(self) -> None:  # noqa: D401 — QThread hook
        import traceback
        try:
            self.started_stages.emit(
                list(self._stages or self._config.orchestration["default_pipeline"])
            )
            self._bus.clear_cancel()
            manifest = run_pipeline(
                self._recording,
                self._config,
                stages=self._stages,
                progress_callback=self._bus,
            )
            self.finished_ok.emit(manifest)
        except Exception as ex:  # noqa: BLE001
            self.failed.emit(repr(ex), traceback.format_exc())


__all__ = ["RunWorker"]
