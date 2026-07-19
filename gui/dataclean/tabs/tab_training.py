"""Training tab — model type selection, hyperparameter entry, training log.

Owns three small parameter setters (epochs, batch size, augmentation
toggle, model type). When the user presses Start a dropout or KL model
is instantiated with the shared state's paths and parameters and run
inside a :class:`WorkerLong`. The intermediate ``view`` signal is
forwarded onto the tab's training-log label.

The tab does not own the model object — once the worker is created the
coordinator takes care of starting it and (when needed) updating the
final log line.
"""

from __future__ import annotations

import time

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from models.drop_out import BayesianDropoutCNN
from models.kl import ModelKl
from src.qt_workers import LongRunningWorker

from ..components import HorizontalLine
from ..state import DataCleanState


class TrainingTab(QWidget):
    """Tab 2 — pick dropout vs KL, set epochs / batch size, run training."""

    # Forwarded to MainTabWidget so the coordinator can connect
    # worker.signals.result / progress / intermediate_result / finished.
    worker_started = pyqtSignal(object)

    def __init__(self, state: DataCleanState, parent=None) -> None:
        super().__init__(parent)
        self.state = state

        self.kl_radio: QRadioButton
        self.dropout_radio: QRadioButton
        self.epoch_input: QLineEdit
        self.epoch_label: QLabel
        self.batch_input: QLineEdit
        self.batch_label: QLabel
        self.augmentation_checkbox: QCheckBox
        self.training_log_label: QLabel
        self.training_progress_bar: QProgressBar

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout()

        # Model selection
        self.kl_radio = QRadioButton("Kl divergence")
        self.kl_radio.toggled.connect(lambda: self._set_model_type("kl"))
        self.dropout_radio = QRadioButton("Drop-Out")
        self.dropout_radio.setChecked(True)
        self.dropout_radio.toggled.connect(lambda: self._set_model_type("drop"))

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Select one of the 2 available models:"))
        model_row.addWidget(self.kl_radio)
        model_row.addWidget(self.dropout_radio)

        # Epochs
        self.epoch_input = QLineEdit()
        self.epoch_label = QLabel(f"Default epochs: {self.state.epochs}")
        epoch_ok_btn = QPushButton("Ok")
        epoch_ok_btn.clicked.connect(self._confirm_epochs)

        epoch_row = QHBoxLayout()
        epoch_row.addWidget(QLabel("Number of epochs:  "))
        epoch_row.addWidget(self.epoch_input)
        epoch_row.addWidget(epoch_ok_btn)
        epoch_row.addWidget(self.epoch_label)

        # Batch size
        self.batch_input = QLineEdit()
        self.batch_label = QLabel(f"Default value: {self.state.batch_size}")
        batch_ok_btn = QPushButton("Ok")
        batch_ok_btn.clicked.connect(self._confirm_batch_size)

        batch_row = QHBoxLayout()
        batch_row.addWidget(QLabel("Batch size:  "))
        batch_row.addWidget(self.batch_input)
        batch_row.addWidget(batch_ok_btn)
        batch_row.addWidget(self.batch_label)

        # Data augmentation
        self.augmentation_checkbox = QCheckBox("Data Augmentation")
        self.augmentation_checkbox.stateChanged.connect(self._toggle_augmentation)

        # Optional folders — retrieve previously-extracted tiles
        optional_label = QLabel(
            "Optional: select train/val folders to retrieve existing tiles:"
        )
        retrieve_train_btn = QPushButton("Train")
        retrieve_train_btn.clicked.connect(self._retrieve_train_folder)
        retrieve_val_btn = QPushButton("Val")
        retrieve_val_btn.clicked.connect(self._retrieve_val_folder)
        retrieve_row = QHBoxLayout()
        retrieve_row.addWidget(retrieve_train_btn)
        retrieve_row.addWidget(retrieve_val_btn)
        retrieve_row.addStretch(1)

        # Training log and progress bar
        self.training_log_label = QLabel("Press Start to begin training.")
        self.training_log_label.setMargin(10)
        self.training_log_label.setFixedWidth(900)
        self.training_log_label.setFixedHeight(1500)
        self.training_log_label.setAlignment(Qt.AlignTop)

        scroll = QScrollArea()
        scroll.setAlignment(Qt.AlignTop)
        scroll.setWidget(self.training_log_label)

        self.training_progress_bar = QProgressBar()
        start_train_btn = QPushButton("Start")
        start_train_btn.clicked.connect(self._start_training)

        for widget in (model_row, epoch_row, batch_row):
            layout.addLayout(widget)
        layout.addWidget(self.augmentation_checkbox)
        layout.addWidget(HorizontalLine())
        layout.addWidget(optional_label)
        layout.addLayout(retrieve_row)
        layout.addWidget(HorizontalLine())
        layout.addWidget(start_train_btn)
        layout.addWidget(scroll)
        layout.addWidget(self.training_progress_bar)

        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Slot — parameter setters
    # ------------------------------------------------------------------

    def _set_model_type(self, model_type: str) -> None:
        self.state.model_type = model_type

    def _toggle_augmentation(self, state: int) -> None:
        self.state.use_augmentation = state == Qt.Checked

    def _confirm_epochs(self) -> None:
        text = self.epoch_input.text()
        if text.isdecimal() and int(text) > 0:
            self.state.epochs = int(text)
            self.epoch_label.setText(f"Epoche: {self.state.epochs}")

    def _confirm_batch_size(self) -> None:
        text = self.batch_input.text()
        if text.isdecimal() and int(text) > 0:
            self.state.batch_size = int(text)
            self.batch_label.setText(f"Batch: {self.state.batch_size}")

    def _retrieve_train_folder(self) -> None:
        from PyQt5.QtWidgets import QFileDialog

        folder = QFileDialog.getExistingDirectory(self, "Seleziona cartella train")
        if folder:
            self.state.tiles_train_path = folder

    def _retrieve_val_folder(self) -> None:
        from PyQt5.QtWidgets import QFileDialog

        folder = QFileDialog.getExistingDirectory(self, "Seleziona cartella val")
        if folder:
            self.state.tiles_val_path = folder

    # ------------------------------------------------------------------
    # Slot — start training
    # ------------------------------------------------------------------

    def _start_training(self) -> None:
        """Instantiate the chosen model and run it inside a WorkerLong."""
        self.training_log_label.setText(
            "Avvio del training in corso, ulteriori informazioni saranno disponibili a breve..."
        )
        timestamp = time.strftime("%Y_%m%d_%H%M%S")
        aug = bool(self.state.use_augmentation)

        model_filename = f"Model{self.state.model_type.capitalize()}-{timestamp}.h5"
        self.state.model_path = f"{self.state.work_path}/{model_filename}"

        if self.state.model_type == "drop":
            model_obj = BayesianDropoutCNN(
                model_save_path=self.state.model_path,
                epochs=self.state.epochs,
                path_train=self.state.tiles_train_path,
                path_val=self.state.tiles_val_path,
                batch_size=self.state.batch_size,
                augment=aug,
            )
        else:
            model_obj = ModelKl(
                model_save_path=self.state.model_path,
                epochs=self.state.epochs,
                path_train=self.state.tiles_train_path,
                path_val=self.state.tiles_val_path,
                batch_size=self.state.batch_size,
                augment=aug,
            )

        worker = LongRunningWorker(model_obj.start_train)
        self.worker_started.emit(worker)

    # ------------------------------------------------------------------
    # Coordinator callback — accept the intermediate "view" signal
    # ------------------------------------------------------------------

    def update_training_log(self, value: object) -> None:
        """Append epoch rows to the training log label.

        The Keras ``TrainingProgressCallback`` calls ``view.emit(...)`` with
        either a full "Epoch N/..." line or a status string. We keep both
        the persisted ``state.training_log`` (used across tab switches)
        and the visible label in sync.
        """
        if "Epoch" in str(value):
            self.state.training_log += f"\n{value}"
            self.training_log_label.setText(self.state.training_log)
        else:
            self.training_log_label.setText(f"{self.state.training_log}\n{value}")
