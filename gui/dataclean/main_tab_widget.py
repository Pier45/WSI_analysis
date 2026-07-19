"""MainTabWidget — coordinator of the 5 Datacleaning tabs.

Holds the shared :class:`DataCleanState`, instantiates one widget per
tab, and wires up the per-tab ``worker_started`` signals to the
application-wide worker result/progress/finished callbacks. Also
wires the Uncertainty tab's ``dataset_classified`` signal to the
Testing tab so confusion-matrix buttons light up as JSONs become
available.
"""

from __future__ import annotations

import time

from PyQt5.QtCore import QThreadPool
from PyQt5.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from .state import DataCleanState
from .tabs import (
    CleaningTab,
    GetTilesTab,
    TestingTab,
    TrainingTab,
    UncertaintyTab,
)


class MainTabWidget(QWidget):
    """Central widget that coordinates all application tabs.

    Manages shared state across tabs (paths, training parameters, analysis
    results) and orchestrates background threads.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.state = DataCleanState(training_log=f"Start  {time.asctime(time.localtime(time.time()))}")
        self.thread_pool = QThreadPool()

        self.tab_tiles: GetTilesTab
        self.tab_training: TrainingTab
        self.tab_uncertainty: UncertaintyTab
        self.tab_cleaning: CleaningTab
        self.tab_testing: TestingTab

        self._build_ui()
        self._wire_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()

        self.tab_tiles = GetTilesTab(self.state, parent=self)
        self.tab_training = TrainingTab(self.state, parent=self)
        self.tab_uncertainty = UncertaintyTab(self.state, parent=self)
        self.tab_cleaning = CleaningTab(self.state, parent=self)
        self.tab_testing = TestingTab(self.state, parent=self)

        self.tabs.addTab(self.tab_tiles, "Get Tiles")
        self.tabs.addTab(self.tab_training, "Training")
        self.tabs.addTab(self.tab_uncertainty, "Uncertainty analysis")
        self.tabs.addTab(self.tab_cleaning, "Data cleaning")
        self.tabs.addTab(self.tab_testing, "Testing")

        root_layout.addWidget(self.tabs)
        self.setLayout(root_layout)

    def _wire_signals(self) -> None:
        """Connect each tab's worker_started signal to the shared handlers."""
        for tab in (
            self.tab_tiles,
            self.tab_training,
            self.tab_uncertainty,
            self.tab_cleaning,
        ):
            tab.worker_started.connect(self._start_worker)

        # Uncertainty -> Testing: enable cm buttons + register JSON paths.
        self.tab_uncertainty.dataset_classified.connect(self.tab_testing.on_dataset_classified)

        # Training tab's intermediate "view" signal updates its log label.
        # _start_worker hooks the intermediate_result to the tab's update method.

    # ------------------------------------------------------------------
    # Shared worker lifecycle handlers
    # ------------------------------------------------------------------

    def _start_worker(self, worker) -> None:
        """Connect a worker to the shared handlers and start it.

        For a training worker, we additionally forward the intermediate
        result signal to ``TrainingTab.update_training_log``. We can identify
        training workers by the callable they wrap via ``worker._fn``:
        the training's fn is the model ``start_train`` bound method, while
        the tiling / uncertainty / cleaning workers wrap
        ``tiler.list_files`` / ``classifier.classify`` / ``clean_js``.
        """
        worker.signals.result.connect(self._on_worker_result)
        worker.signals.progress.connect(self._on_worker_progress)
        worker.signals.finished.connect(self._on_worker_finished)

        fn_name = getattr(getattr(worker, "_fn", None), "__name__", "")
        if fn_name == "start_train":
            worker.signals.progress.connect(self.tab_training.training_progress_bar.setValue)
            worker.signals.intermediate_result.connect(self.tab_training.update_training_log)
            worker.signals.finished.connect(self._on_training_complete)
        elif fn_name == "classify":
            worker.signals.progress.connect(self.tab_uncertainty.uncertainty_progress_bar.setValue)
        elif fn_name == "clean_js":
            worker.signals.progress.connect(self.tab_cleaning.copy_progress_bar.setValue)

        self.thread_pool.start(worker)

    def _on_worker_result(self, result) -> None:
        self.tab_training.training_log_label.setText(str(result))
        print(result)

    def _on_worker_progress(self, value: int) -> None:
        print(f"{value}% completato")

    def _on_worker_finished(self) -> None:
        print("Thread completato.")

    def _on_training_complete(self) -> None:
        self.tab_training.training_log_label.setText(
            "Training completato! La history è stata salvata nella cartella di lavoro."
        )
        print("Training completato.")
