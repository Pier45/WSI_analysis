"""Datacleaning application — formerly ``ui_dataclean.py``.

Public API
----------
- :class:`MainWindow` — top-level :class:`QMainWindow` to ``.show()``.

The thin launcher at the repo root imports this and runs the Qt event
loop after applying Linux-specific Qt platform fixes.
"""

from __future__ import annotations

from .main_window import MainWindow

__all__ = ["MainWindow"]
