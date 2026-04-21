"""
Main window.

Layout
------
+------------------------+----------------------+
|  recording picker      |  per-stage panels    |
|  + run-all / cancel    |  (progress + status) |
+------------------------+----------------------+
|  scrolling console (full width)               |
+-----------------------------------------------+

Everything heavy runs on a :class:`RunWorker`; the GUI thread only handles
events delivered via the shared :class:`QtProgressBus`.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QSplitter, QVBoxLayout, QWidget,
)

from ..config.schema import PipelineConfig
from ..io.recording import Recording
from ..stages import STAGE_ORDER
from .bus import QtProgressBus
from .console import ConsoleWidget
from .stage_panel import StagePanel
from .worker import RunWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calcium Pipeline")
        self.resize(1100, 760)

        self.config = PipelineConfig()
        self.bus = QtProgressBus()
        self.recording: Recording | None = None
        self._run_all_worker: RunWorker | None = None

        # --- recording picker -------------------------------------------
        self.rec_label = QLabel("No recording selected")
        pick_btn = QPushButton("Pick recording...")
        pick_btn.clicked.connect(self._pick_recording)

        self.run_all_btn = QPushButton("Run full pipeline")
        self.run_all_btn.setEnabled(False)
        self.run_all_btn.clicked.connect(self._run_all)

        self.cancel_all_btn = QPushButton("Cancel")
        self.cancel_all_btn.setEnabled(False)
        self.cancel_all_btn.clicked.connect(self._cancel_all)

        top = QHBoxLayout()
        top.addWidget(pick_btn)
        top.addWidget(self.rec_label, 1)
        top.addWidget(self.run_all_btn)
        top.addWidget(self.cancel_all_btn)

        # --- per-stage panels -------------------------------------------
        self.panels = {
            name: StagePanel(name, self.config, self.bus)
            for name in STAGE_ORDER
        }
        stages_col = QVBoxLayout()
        for panel in self.panels.values():
            stages_col.addWidget(panel)
        stages_col.addStretch(1)

        stages_wrap = QWidget()
        stages_wrap.setLayout(stages_col)

        # --- console ----------------------------------------------------
        self.console = ConsoleWidget()
        self.bus.event.connect(self.console.append_event)

        # --- splitter layout -------------------------------------------
        splitter = QSplitter()
        splitter.setOrientation(1)  # vertical
        splitter.addWidget(stages_wrap)
        splitter.addWidget(self.console)
        splitter.setSizes([500, 260])

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.addLayout(top)
        lay.addWidget(splitter, 1)
        self.setCentralWidget(central)

    # --- slots ----------------------------------------------------------

    def _pick_recording(self) -> None:
        start_dir = str(self.config.paths.data_root)
        folder = QFileDialog.getExistingDirectory(
            self, "Select recording folder", start_dir,
        )
        if not folder:
            return
        try:
            rec = Recording(
                folder,
                notes_root=self.config.paths.notes_root,
                fps_override=self.config.fps_override,
                zoom_override=self.config.zoom_override,
            )
        except FileNotFoundError as ex:
            self.rec_label.setText(f"Invalid: {ex}")
            return

        self.recording = rec
        self.rec_label.setText(str(rec.path))
        self.run_all_btn.setEnabled(True)
        for panel in self.panels.values():
            panel.set_recording(rec)

    def _run_all(self) -> None:
        if self.recording is None:
            return
        self.run_all_btn.setEnabled(False)
        self.cancel_all_btn.setEnabled(True)
        self._run_all_worker = RunWorker(
            self.recording, self.config, self.bus,
            stages=self.config.orchestration["default_pipeline"], parent=self,
        )
        self._run_all_worker.finished_ok.connect(self._on_run_all_ok)
        self._run_all_worker.failed.connect(self._on_run_all_failed)
        self._run_all_worker.start()

    def _cancel_all(self) -> None:
        if self._run_all_worker is not None:
            self._run_all_worker.request_cancel()

    def _on_run_all_ok(self, manifest: dict) -> None:
        self.run_all_btn.setEnabled(True)
        self.cancel_all_btn.setEnabled(False)

    def _on_run_all_failed(self, msg: str, tb: str) -> None:
        self.run_all_btn.setEnabled(True)
        self.cancel_all_btn.setEnabled(False)


def launch(argv=None):
    """Entry point — ``python -m calcium_pipeline.gui.app``."""
    app = QApplication(argv if argv is not None else sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(launch(sys.argv))
