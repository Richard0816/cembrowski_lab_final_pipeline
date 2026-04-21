"""
Scrolling console widget that subscribes to a :class:`QtProgressBus`.

Color-codes ``log`` / ``error`` / ``tick`` events; auto-scrolls to the bottom
unless the user has scrolled up manually (then respects their position).
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import QTextEdit

from ..orchestration.progress import ProgressEvent


_COLOURS = {
    "start":  "#4a6",
    "tick":   "#888",
    "finish": "#4a6",
    "log":    "#333",
    "error":  "#c33",
}


class ConsoleWidget(QTextEdit):
    """
    Drop-in read-only console. Connect with
    ``bus.event.connect(console.append_event)``.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)

    def append_event(self, ev: ProgressEvent) -> None:
        colour = _COLOURS.get(ev.kind, "#333")
        frac = f"{100 * ev.fraction:5.1f}%" if ev.total > 0 else "     "
        wall = f"{ev.wall_time:6.2f}s" if ev.wall_time else "       "
        html = (
            f'<span style="color:{colour}">'
            f"[{ev.stage:>18}] {wall} {frac}  {ev.message}"
            f"</span>"
        )
        self.append(html)

        # Auto-scroll only if we were already at the bottom.
        scrollbar = self.verticalScrollBar()
        if scrollbar.value() >= scrollbar.maximum() - 2:
            self.moveCursor(QTextCursor.MoveOperation.End)


__all__ = ["ConsoleWidget"]
