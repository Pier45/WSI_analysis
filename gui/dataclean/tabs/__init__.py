"""Datacleaning application tabs.

Each tab is a self-contained ``QWidget`` that takes the shared
:class:`~gui.dataclean.state.DataCleanState` instance (and an optional
parent) and emits a ``worker_started`` signal whenever it launches a
background worker. The :class:`~gui.dataclean.main_tab_widget.MainTabWidget`
wires those signals to the application-wide worker-result callbacks.
"""

from __future__ import annotations

from .tab_cleaning import CleaningTab
from .tab_testing import TestingTab
from .tab_tiles import GetTilesTab
from .tab_training import TrainingTab
from .tab_uncertainty import UncertaintyTab

__all__ = [
    "CleaningTab",
    "GetTilesTab",
    "TestingTab",
    "TrainingTab",
    "UncertaintyTab",
]
