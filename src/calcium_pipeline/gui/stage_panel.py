"""
``StagePanel`` — a single-stage strip (progress bar + status + run/cancel).

One per stage; the main window stacks them vertically. Each panel owns a
``RunWorker`` when its stage is running and is wired to the shared
:class:`QtProgressBus`.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from ..config.schema import PipelineConfig
from ..io.recording import Recording
from ..orchestration.progress import ProgressEvent
from .bus import QtProgressBus
from .worker import RunWorker


class StagePanel(QFrame):
    """
    One row per stage. Methods:

    * :meth:`set_recording` — which recording to run against.
    * :meth:`run`           — fire up a ``RunWorker`` for just this stage.
    * :meth:`on_event`      — slot for ``QtProgressBus.event``; updates only
      if the event's ``stage`` matches this panel's.
    """

    def __init__(
        self,
        stage_name: str,
        config: PipelineConfig,
        bus: QtProgressBus,
        parent=None,
    ):
        super().__init__(parent)
        self.stage_name = stage_name
        self._config = config
        self._bus = bus
        self._recording: Recording | None = None
        self._worker: RunWorker | None = None

        self.setFrameShape(QFrame.Shape.StyledPanel)

        # --- widgets -----------------------------------------------------
        self.title = QLabel(stage_name)
        self.title.setStyleSheet("font-weight: bold;")

        self.status = QLabel("idle")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.run_btn = QPushButton("Run")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)

        self.run_btn.clicked.connect(self.run)
        self.cancel_btn.clicked.connect(self.cancel)

        row = QHBoxLayout()
        row.addWidget(self.title, 2)
        row.addWidget(self.status, 4)
        row.addWidget(self.progress, 4)
        row.addWidget(self.run_btn)
        row.addWidget(self.cancel_btn)

        lay = QVBoxLayout(self)
        lay.addLayout(row)

        # --- wire bus ----------------------------------------------------
        bus.event.connect(self.on_event)

    # --- public API -------------------------------------------------------

    def set_recording(self, rec: Recording | None) -> None:
        self._recording = rec
        self.run_btn.setEnabled(rec is not None)

    def run(self) -> None:
        if self._recording is None:
            self.status.setText("no recording selected")
            return
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status.setText("queued...")
        self._worker = RunWorker(
            self._recording, self._config, self._bus,
            stages=[self.stage_name], parent=self,
        )
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def cancel(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_cancel()
            self.status.setText("cancelling...")

    # --- slots -----------------------------------------------------------

    def on_event(self, ev: ProgressEvent) -> None:
        if ev.stage != self.stage_name:
            return
        if ev.total > 0:
            self.progress.setValue(int(100 * ev.fraction))
        if ev.message:
            self.status.setText(ev.message)

    def _on_finished(self, manifest: dict) -> None:
        self.status.setText("done")
        self.progress.setValue(100)
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _on_failed(self, msg: str, tb: str) -> None:
        self.status.setText(f"FAILED: {msg}")
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)


__all__ = ["StagePanel"]
