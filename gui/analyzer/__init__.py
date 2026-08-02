"""Bayesian Analyzer application — formerly ``ui_pyqt5.py``.

Public API
----------
- :class:`ImageViewer` — top-level :class:`QMainWindow` to ``.show()``.

The thin launcher at the repo root imports this and runs the Qt event
loop after applying Linux-specific Qt platform fixes (same pattern as
:mod:`gui.dataclean`).
"""

from __future__ import annotations

from .main_window import ImageViewer

__all__ = ["ImageViewer"]
