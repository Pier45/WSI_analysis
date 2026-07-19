"""Uncertainty analysis tab — run MC-Dropout classification per dataset.

Picks the previously-trained model and runs an N-sample Monte Carlo
inference pass over each dataset (train / val / test). The resulting
``dictionary_monte_<N>_js.txt`` JSON files become the input of the
Testing tab (for confusion matrices) and the Cleaning tab (for
histograms + thresholding).
"""

from __future__ import annotations

import os

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.classification import Classification
from src.qt_workers import LongRunningWorker

from ..components import HorizontalLine
from ..state import DataCleanState


class UncertaintyTab(QWidget):
    """Tab 3 — run Monte-Carlo classification on a chosen dataset."""

    # Forwarded to MainTabWidget for connection to result/progress/finished.
    worker_started = pyqtSignal(object)

    # Emitted when a dataset's MC run completes. Carries the string label
    # ("train" / "val" / "test") so a connected listener (e.g. the Testing
    # tab) can refresh only the affected confusion matrix.
    dataset_classified = pyqtSignal(str)

    def __init__(self, state: DataCleanState, parent=None) -> None:
        super().__init__(parent)
        self.state = state

        self.monte_input: QLineEdit
        self.monte_label: QLabel
        self.uncertainty_progress_bar: QProgressBar

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        bold_font = QFont("Helvetica", 15, QFont.Bold)

        description = QLabel(
            "This step computes uncertainty values (aleatoric, epistemic, total)."
        )

        # Monte Carlo samples
        self.monte_input = QLineEdit()
        self.monte_label = QLabel(f"Default value: {self.state.monte_carlo_samples}")
        monte_ok_btn = QPushButton("Ok")
        monte_ok_btn.clicked.connect(self._confirm_monte_carlo)

        monte_row = QHBoxLayout()
        monte_row.addWidget(QLabel("Monte Carlo samples:"))
        monte_row.addWidget(self.monte_input)
        monte_row.addWidget(monte_ok_btn)
        monte_row.addWidget(self.monte_label)

        self.uncertainty_progress_bar = QProgressBar()

        # One Start button per dataset
        dataset_labels = ["TRAINING SET", "VALIDATION SET", "TEST SET"]
        start_fns = [self._run_uncertainty_train, self._run_uncertainty_val, self._run_uncertainty_test]

        layout.addWidget(description, alignment=Qt.AlignTop)
        layout.addWidget(HorizontalLine())
        layout.addLayout(monte_row)

        for label_text, fn in zip(dataset_labels, start_fns, strict=True):
            title = QLabel(label_text)
            title.setFont(bold_font)
            start_btn = QPushButton("Start")
            start_btn.clicked.connect(fn)
            layout.addWidget(title)
            layout.addWidget(start_btn)
            layout.addWidget(HorizontalLine())

        layout.addStretch(1)
        layout.addWidget(self.uncertainty_progress_bar)
        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Slot — parameter setter
    # ------------------------------------------------------------------

    def _confirm_monte_carlo(self) -> None:
        text = self.monte_input.text()
        if text.isdecimal() and int(text) > 0:
            self.state.monte_carlo_samples = int(text)
            self.monte_label.setText(f"MC samples: {self.state.monte_carlo_samples}")

    # ------------------------------------------------------------------
    # Slots — start uncertainty analysis per dataset
    # ------------------------------------------------------------------

    def _run_uncertainty_train(self) -> None:
        self._run_uncertainty_analysis(self.state.tiles_train_path, "train")
        self.state.train_json = self._build_json_path(self.state.tiles_train_path)
        self.dataset_classified.emit("train")

    def _run_uncertainty_val(self) -> None:
        self._run_uncertainty_analysis(self.state.tiles_val_path, "val")
        self.state.val_json = self._build_json_path(self.state.tiles_val_path)
        self.dataset_classified.emit("val")

    def _run_uncertainty_test(self) -> None:
        self._run_uncertainty_analysis(self.state.tiles_test_path, "test")
        self.state.test_json = self._build_json_path(self.state.tiles_test_path)
        self.dataset_classified.emit("test")

    def _run_uncertainty_analysis(self, data_path: str, dataset_name: str) -> None:
        """Spin up a :class:`WorkerLong` running MC classification."""
        classifier = Classification(data_path, ty="datacleaning")
        worker = LongRunningWorker(
            classifier.classify,
            "datacleaning",
            self.state.monte_carlo_samples,
            self.state.model_path,
        )
        self.worker_started.emit(worker)

    def _build_json_path(self, base_path: str) -> str:
        return os.path.join(
            base_path,
            f"dictionary_monte_{self.state.monte_carlo_samples}_js.txt",
        )
