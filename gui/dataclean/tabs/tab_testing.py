"""Testing tab — thin wrapper around :class:`PerformanceTab`.

The confusion-matrix widget proper lives in
:mod:`src.performance_widget` and exposes ``traincm`` / ``valcm`` /
``testcm`` buttons plus a ``get_paths(...)`` helper. The Uncertainty tab
calls back into this wrapper to enable the right button after each
dataset's MC classification completes.
"""

from __future__ import annotations

from src.performance_widget import PerformanceTab

from ..state import DataCleanState


class TestingTab(PerformanceTab):
    """Tab 5 — wraps :class:`PerformanceTab` with shared-state awareness.

    ``PerformanceTab`` originally accepted a ``parent`` pointing at the
    monolithic ``MainTabWidget`` so the Uncertainty slots could poke
    ``parent.tab_testing.traincm.setEnabled(True)``. After the split, the
    Uncertainty tab talks to this wrapper via the
    :class:`UncertaintyTab.dataset_classified` signal — no parent poking
    any more. The class is preserved (rather than using
    ``PerformanceTab`` directly) so it can grow state-aware helpers.
    """

    def __init__(self, state: DataCleanState, parent=None) -> None:
        super().__init__(parent=parent)
        self.state = state

    def on_dataset_classified(self, dataset: str) -> None:
        """Enable the right cm button + register the JSON path.

        Called by :class:`MainTabWidget` when the Uncertainty tab emits
        ``dataset_classified``. The JSON paths are read straight out of
        shared state.
        """
        if dataset == "train":
            self.traincm.setEnabled(True)
            self.get_paths(train=self.state.train_json)
        elif dataset == "val":
            self.valcm.setEnabled(True)
            self.get_paths(val=self.state.val_json)
        elif dataset == "test":
            self.testcm.setEnabled(True)
            self.get_paths(test=self.state.test_json)
