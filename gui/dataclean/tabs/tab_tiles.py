"""Get-Tiles tab — dataset folder selection and tile extraction.

The user selects the working folder plus per-set (train/val/test) source
folders; the SVS files inside each per-class sub-folder (``AC``, ``AD``,
``H``) are listed for the user's reference. Pressing Start launches a
:class:`StartAnalysis` worker per dataset that extracts 256x256 tiles
into ``<work_path>/<dataset>/<class>``.

The tab reads ``train_path`` / ``val_path`` / ``test_path`` from the
shared state and writes back the resolved ``tiles_*_path``s; the
Training tab reads those paths when it kicks off training.
"""

from __future__ import annotations

import os

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.multi_processing_analysis import StartAnalysis
from src.qt_workers import LongRunningWorker

from ..components import HorizontalLine
from ..constants import DEFAULT_TILE_LEVEL, DEFAULT_TILE_SIZE, KNOWN_CLASSES
from ..state import DataCleanState


class GetTilesTab(QWidget):
    """Tab 1 — pick source folders and launch per-dataset tiling workers."""

    # Emitted whenever a worker is started. The coordinator connects this to
    # the shared worker-result / progress / finished slots so the tab does
    # not need to know about the worker's signal shape.
    worker_started = pyqtSignal(object)

    def __init__(self, state: DataCleanState, parent=None) -> None:
        super().__init__(parent)
        self.state = state

        self.dataset_list_widgets: dict[str, dict[str, QListWidget]] = {}
        self.progress_bars: dict[str, list[QProgressBar]] = {}

        self.start_tiles_btn: QPushButton

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        bold_font = QFont("Helvetica", 15, QFont.Bold)

        work_folder_row = self._make_label_button_row(
            "First things first, select a folder to save all data:",
            "Select folder",
            self._select_work_folder,
        )

        sections = [
            ("TRAINING SET", self._select_train_folder),
            ("VALIDATION SET", self._select_val_folder),
            ("TEST SET", self._select_test_folder),
        ]
        dataset_names = ["train", "val", "test"]

        layout.addLayout(work_folder_row)

        for name, select_fn in zip(dataset_names, [s[1] for s in sections], strict=True):
            title_label = QLabel(sections[dataset_names.index(name)][0])
            title_label.setFont(bold_font)

            select_row = self._make_label_button_row(
                f"Select the {name} folder:",
                "Select folder",
                select_fn,
            )
            list_ac = QListWidget()
            list_ad = QListWidget()
            list_h = QListWidget()
            self.dataset_list_widgets[name] = {"AC": list_ac, "AD": list_ad, "H": list_h}

            lists_row = QHBoxLayout()
            for w in (list_ac, list_ad, list_h):
                lists_row.addWidget(w)

            pb1, pb2, pb3 = QProgressBar(), QProgressBar(), QProgressBar()
            self.progress_bars[name] = [pb1, pb2, pb3]

            progress_row = QHBoxLayout()
            for pb in (pb1, pb2, pb3):
                progress_row.addWidget(pb)

            layout.addWidget(HorizontalLine())
            layout.addWidget(title_label)
            layout.addLayout(select_row)
            layout.addLayout(lists_row)
            layout.addLayout(progress_row)

        self.start_tiles_btn = QPushButton("Start")
        self.start_tiles_btn.clicked.connect(self._start_tiling)
        layout.addWidget(HorizontalLine())
        layout.addWidget(self.start_tiles_btn)

        self.setLayout(layout)

    def _make_label_button_row(self, label_text: str, button_text: str, callback) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel(label_text))
        btn = QPushButton(button_text)
        btn.clicked.connect(callback)
        row.addWidget(btn)
        return row

    # ------------------------------------------------------------------
    # Slot — folder selection
    # ------------------------------------------------------------------

    def _select_work_folder(self) -> None:
        from PyQt5.QtWidgets import QFileDialog

        folder = QFileDialog.getExistingDirectory(self, "Seleziona cartella di lavoro")
        if folder:
            self.state.work_path = folder
            self.state.tiles_train_path = os.path.join(folder, "train")
            self.state.tiles_val_path = os.path.join(folder, "val")
            self.state.tiles_test_path = os.path.join(folder, "test")

    def _select_train_folder(self) -> None:
        self.state.train_path = self._select_dataset_folder("train")

    def _select_val_folder(self) -> None:
        self.state.val_path = self._select_dataset_folder("val")

    def _select_test_folder(self) -> None:
        self.state.test_path = self._select_dataset_folder("test")

    def _select_dataset_folder(self, dataset_name: str) -> str:
        """Open a folder dialog and populate the QListWidgets with files in
        each per-class sub-folder (AC / AD / H).

        Returns the selected folder (or the previous path if canceled).
        """
        from PyQt5.QtWidgets import QFileDialog

        folder = QFileDialog.getExistingDirectory(self, "Seleziona cartella")
        if not folder:
            return getattr(self.state, f"{dataset_name}_path", "")

        list_widgets = self.dataset_list_widgets.get(dataset_name, {})

        try:
            for class_name in os.listdir(folder):
                key = class_name.upper()
                if key not in {k.upper() for k in KNOWN_CLASSES}:
                    print(
                        f"||||| > Ignoro elemento non-class: {repr(key)} "
                        f"in {KNOWN_CLASSES} in dataset {class_name}"
                    )
                    continue
                files = os.listdir(os.path.join(folder, class_name))
                list_widgets[key].addItems(files)
                print(
                    f"||||| > Popolamento widget per {dataset_name} - "
                    f"classe {class_name} - files {files}"
                )
        except OSError as e:
            print(f"Errore nella lettura della cartella {folder}: {e}")

        return folder

    # ------------------------------------------------------------------
    # Slot — start tiling
    # ------------------------------------------------------------------

    def _start_tiling(self) -> None:
        """Launch per-dataset tiling workers (one worker per class sub-folder)."""
        datasets = [
            ("train", self.state.train_path, self.progress_bars["train"]),
            ("val", self.state.val_path, self.progress_bars["val"]),
            ("test", self.state.test_path, self.progress_bars["test"]),
        ]
        for name, path, progress_bar_list in datasets:
            self._tile_dataset(name, path, progress_bar_list)

    def _tile_dataset(self, dataset_name: str, source_path: str, progress_bars: list) -> None:
        if not source_path or not os.path.isdir(source_path):
            print(f"Percorso non valido per {dataset_name}: {source_path}")
            return

        try:
            class_dirs = os.listdir(source_path)
        except OSError as e:
            print(f"Impossibile leggere {source_path}: {e}")
            return

        for idx, class_dir in enumerate(class_dirs):
            save_folder = os.path.join(self.state.work_path, dataset_name, class_dir)
            class_path = os.path.join(source_path, class_dir)
            os.makedirs(save_folder, exist_ok=True)

            tiler = StartAnalysis(tile_size=DEFAULT_TILE_SIZE, lev_sec=DEFAULT_TILE_LEVEL)
            worker = LongRunningWorker(tiler.list_files, class_path, save_folder)
            self.worker_started.emit(worker)
            if idx < len(progress_bars):
                worker.signals.progress.connect(progress_bars[idx].setValue)
