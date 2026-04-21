"""
calcium_pipeline.gui
--------------------
PyQt6 front-end that binds to every stage via
:class:`orchestration.progress.ProgressBus`.

The GUI is entirely optional — every stage and orchestrator function works
without a display. To launch::

    python -m calcium_pipeline.gui.app

Module layout
-------------
* :mod:`.app`          — main window, menu bar, recording picker
* :mod:`.bus`          — ``QtProgressBus`` adapter (pyqtSignal <-> ProgressEvent)
* :mod:`.config_panel` — widget that binds to a :class:`PipelineConfig`
* :mod:`.stage_widgets.*` — one widget per stage (progress bar + results pane)
* :mod:`.console`      — scrolling ``QTextEdit`` that subscribes to the bus

PyQt6 is imported lazily in each submodule so that
``import calcium_pipeline`` keeps working on headless servers.
"""
from __future__ import annotations

__all__ = ["launch"]


def launch(*args, **kwargs):
    """Start the GUI. Imports PyQt6 lazily."""
    from .app import launch as _launch
    return _launch(*args, **kwargs)
