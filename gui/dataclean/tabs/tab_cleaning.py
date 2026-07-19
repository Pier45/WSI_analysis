"""Data cleaning tab — uncertainty histograms + manual / auto thresholding.

Picks up the JSON dictionaries produced by the Uncertainty tab, draws
three histograms (aleatoric / epistemic / total), runs Otsu or the
"new" custom threshold (or accepts a manual one), shows the number of
retained tiles, and copies the surviving tiles into a new clean dataset
folder.

The tab owns the live :class:`Th` instance via the shared state — that
attribute is the bridge to the Testing tab if it ever needs the cleaned
counts.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from src.config import CLASS_NAMES
from src.qt_workers import LongRunningWorker
from src.uncertainty_analysis import Th

from ..components import HorizontalLine, MatplotlibCanvas
from ..constants import MANUAL_THRESHOLD_MAX, MANUAL_THRESHOLD_MIN
from ..state import DataCleanState


class CleaningTab(QWidget):
    """Tab 4 — histograms, threshold selection, clean-dataset export."""

    # Forwarded to the coordinator for connection to the worker signals.
    worker_started = pyqtSignal(object)

    def __init__(self, state: DataCleanState, parent=None) -> None:
        super().__init__(parent)
        self.state = state

        # Dataset selector
        self.train_dataset_radio: QRadioButton
        self.val_dataset_radio: QRadioButton

        # Histograms
        self.hist_aleatoric: MatplotlibCanvas
        self.hist_epistemic: MatplotlibCanvas
        self.hist_total: MatplotlibCanvas

        # Stats / buttons
        self.before_count_label: QLabel
        self.after_count_label: QLabel
        self.show_removed_btn: QPushButton

        # Mode selector
        self.auto_radio: QRadioButton
        self.manual_radio: QRadioButton
        self._control_radio: QRadioButton
        self.manual_threshold_input: QLineEdit
        self.apply_manual_threshold_btn: QPushButton

        # Threshold selector
        self.otsu_radio: QRadioButton
        self.new_th_radio: QRadioButton
        self.manual_th_radio: QRadioButton

        # Save / action
        self.save_folder_btn: QPushButton
        self.create_dataset_btn: QPushButton
        self.copy_progress_bar: QProgressBar

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout()

        # Dataset selection
        self.train_dataset_radio = QRadioButton("Training set")
        self.train_dataset_radio.toggled.connect(self._load_train_histograms)
        self.val_dataset_radio = QRadioButton("Validation set")
        self.val_dataset_radio.toggled.connect(self._load_val_histograms)
        dataset_group = QButtonGroup(self)
        dataset_group.addButton(self.train_dataset_radio)
        dataset_group.addButton(self.val_dataset_radio)

        select_dataset_row = QHBoxLayout()
        select_dataset_row.addWidget(QLabel("Seleziona il dataset da analizzare:"))
        select_dataset_row.addWidget(self.train_dataset_radio)
        select_dataset_row.addWidget(self.val_dataset_radio)

        # Histograms
        self.hist_aleatoric = MatplotlibCanvas("Aleatoric uncertainty", self, width=5, height=4)
        self.hist_epistemic = MatplotlibCanvas("Epistemic uncertainty", self, width=5, height=4)
        self.hist_total = MatplotlibCanvas("Total uncertainty", self, width=5, height=4)

        right_hists = QVBoxLayout()
        right_hists.addWidget(self.hist_aleatoric)
        right_hists.addWidget(self.hist_epistemic)

        hists_row = QHBoxLayout()
        hists_row.addWidget(self.hist_total)
        hists_row.addLayout(right_hists)

        # Stats
        self.before_count_label = QLabel()
        self.before_count_label.hide()
        self.after_count_label = QLabel()
        self.after_count_label.hide()
        self.show_removed_btn = QPushButton("Mostra numero di tile rimosse per classe")
        self.show_removed_btn.hide()
        self.show_removed_btn.clicked.connect(self._show_removed_tiles)

        stats_col = QVBoxLayout()
        stats_col.addWidget(self.before_count_label)
        stats_col.addWidget(self.after_count_label)
        stats_row = QHBoxLayout()
        stats_row.addLayout(stats_col)
        stats_row.addWidget(self.show_removed_btn)

        # Mode selection (auto/manual)
        description_modes = QLabel(
            "Auto: il software calcola automaticamente la soglia ottimale. "
            "Manual: inserire manualmente il valore desiderato."
        )

        self.auto_radio = QRadioButton("Auto")
        self.auto_radio.setEnabled(False)
        self.manual_radio = QRadioButton("Manual")
        self.manual_radio.setEnabled(False)
        self._control_radio = QRadioButton("Control")
        self._control_radio.hide()
        self.auto_radio.toggled.connect(self._on_auto_mode_selected)
        self.manual_radio.toggled.connect(self._on_manual_mode_selected)

        mode_group = QButtonGroup(self)
        mode_group.addButton(self.auto_radio)
        mode_group.addButton(self.manual_radio)
        mode_group.addButton(self._control_radio)

        self.manual_threshold_input = QLineEdit()
        self.manual_threshold_input.hide()
        self.apply_manual_threshold_btn = QPushButton("Applica soglia manuale")
        self.apply_manual_threshold_btn.clicked.connect(self._apply_manual_threshold)
        self.apply_manual_threshold_btn.hide()

        mode_row = QHBoxLayout()
        mode_row.addWidget(self.auto_radio, alignment=Qt.AlignCenter)
        mode_row.addWidget(self.manual_radio, alignment=Qt.AlignCenter)
        mode_row.addWidget(self.manual_threshold_input)
        mode_row.addWidget(self.apply_manual_threshold_btn)
        mode_row.addStretch(1)

        # Threshold selector
        self.otsu_radio = QRadioButton("Soglia Otsu")
        self.otsu_radio.setEnabled(False)
        self.otsu_radio.toggled.connect(lambda: self._set_threshold_type("otsu"))
        self.new_th_radio = QRadioButton("Nuova soglia")
        self.new_th_radio.setEnabled(False)
        self.new_th_radio.toggled.connect(lambda: self._set_threshold_type("new"))
        self.manual_th_radio = QRadioButton("Soglia manuale")
        self.manual_th_radio.setEnabled(False)
        self.manual_th_radio.toggled.connect(lambda: self._set_threshold_type("manual"))

        threshold_group = QButtonGroup(self)
        threshold_group.addButton(self.otsu_radio)
        threshold_group.addButton(self.new_th_radio)
        threshold_group.addButton(self.manual_th_radio)

        threshold_row = QHBoxLayout()
        for radio in (self.otsu_radio, self.new_th_radio, self.manual_th_radio):
            threshold_row.addWidget(radio)
        threshold_row.addStretch(1)

        # Save folder + action
        self.save_folder_btn = QPushButton("Seleziona cartella vuota")
        self.save_folder_btn.clicked.connect(self._select_save_folder)
        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("Cartella dove creare il dataset pulito:"))
        folder_row.addWidget(self.save_folder_btn)

        self.create_dataset_btn = QPushButton("Crea nuovo dataset")
        self.create_dataset_btn.clicked.connect(self._create_clean_dataset)
        self.create_dataset_btn.setEnabled(False)

        self.copy_progress_bar = QProgressBar()

        for item in (
            select_dataset_row, hists_row, stats_row,
            (HorizontalLine(),), (description_modes,), mode_row,
            (HorizontalLine(),), folder_row, threshold_row,
        ):
            if isinstance(item, tuple):
                layout.addWidget(item[0])
            else:
                layout.addLayout(item)

        layout.addWidget(self.create_dataset_btn)
        layout.addStretch(1)
        layout.addWidget(self.copy_progress_bar)
        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Slot — pick save folder
    # ------------------------------------------------------------------

    def _select_save_folder(self) -> None:
        from PyQt5.QtWidgets import QFileDialog

        folder = QFileDialog.getExistingDirectory(self, "Seleziona cartella di salvataggio")
        if folder:
            self.state.clean_save_path = folder
            self.create_dataset_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Slot — threshold type
    # ------------------------------------------------------------------

    def _set_threshold_type(self, threshold_type: str) -> None:
        self.state.selected_threshold = threshold_type

    # ------------------------------------------------------------------
    # Slot — load and draw histograms
    # ------------------------------------------------------------------

    def _load_train_histograms(self) -> None:
        self._draw_histograms(self.state.train_json, "train")

    def _load_val_histograms(self) -> None:
        self._draw_histograms(self.state.val_json, "val")

    def _draw_histograms(self, json_path: str, dataset_name: str) -> None:
        """Load uncertainty values from the JSON, refresh the histograms."""
        self.state.cleaning_obj = Th(json_path, dataset_name)
        (
            self.state.aleatoric_values,
            self.state.epistemic_values,
            self.state.total_uncertainty_values,
        ) = self.state.cleaning_obj.create_list()
        total_count = len(self.state.total_uncertainty_values)
        max_val = max(self.state.total_uncertainty_values)
        n_bins = max(1, round(total_count / 100))

        for canvas, values, title in (
            (self.hist_total,     self.state.total_uncertainty_values, "Total uncertainty"),
            (self.hist_aleatoric, self.state.aleatoric_values,         "Aleatoric uncertainty"),
            (self.hist_epistemic, self.state.epistemic_values,         "Epistemic uncertainty"),
        ):
            canvas.axes.clear()
            canvas.axes.set_xlim(0, max_val)
            canvas.axes.set_title(title)
            canvas.axes.hist(values, bins=n_bins, color="#FFA420")
            canvas.draw()

        self.before_count_label.setText(
            f"Totale tile prima del cleaning: {total_count}"
        )
        self.before_count_label.show()
        self.after_count_label.hide()
        self.show_removed_btn.hide()
        self._unlock_threshold_mode()

    def _unlock_threshold_mode(self) -> None:
        """After the histograms are loaded, enable the auto/manual radios."""
        self.auto_radio.setEnabled(True)
        self.manual_radio.setEnabled(True)
        self.auto_radio.setChecked(False)
        self.manual_radio.setChecked(False)
        self._control_radio.setChecked(True)
        self.state.threshold_flag = 0

    # ------------------------------------------------------------------
    # Slots — auto / manual threshold
    # ------------------------------------------------------------------

    def _on_auto_mode_selected(self) -> None:
        """Compute Otsu + new thresholds and draw vertical lines on the total histogram."""
        if not self.state.cleaning_obj:
            return
        self.state.cleaning_obj.otsu()
        new_th, otsu_th, count_new, count_otsu = self.state.cleaning_obj.th_managment()

        self.after_count_label.setText(
            f"Totale tile dopo il cleaning:\n"
            f"  Soglia Otsu:   {count_otsu:>10}\n"
            f"  Nuova soglia:  {count_new:>10}"
        )
        self.after_count_label.show()
        self.show_removed_btn.show()

        self.hist_total.axes.axvline(x=new_th,  ls="--", color="k",   label="Nuova soglia")
        self.hist_total.axes.axvline(x=otsu_th,         color="red",  label="Soglia Otsu")
        self.hist_total.axes.axvline(x=-3,       ls="--", color="y",   label="Soglia manuale")

        if self.state.threshold_flag == 0:
            self.hist_total.axes.legend(prop={"size": 10})
            self.state.threshold_flag = 1
        self.hist_total.draw()

        if self.auto_radio.isChecked():
            self.manual_threshold_input.hide()
            self.apply_manual_threshold_btn.hide()
            self._enable_threshold_radios()

    def _on_manual_mode_selected(self) -> None:
        if self.manual_radio.isChecked():
            self.manual_threshold_input.show()
            self.apply_manual_threshold_btn.show()

    def _apply_manual_threshold(self) -> None:
        try:
            threshold = float(self.manual_threshold_input.text())
        except ValueError:
            return

        if not (MANUAL_THRESHOLD_MIN < threshold < MANUAL_THRESHOLD_MAX):
            return

        self.state.selected_threshold = threshold
        self.state.cleaning_obj.otsu()
        manual_th, otsu_th, count_new, count_otsu = self.state.cleaning_obj.th_managment(threshold)

        self.after_count_label.setText(
            f"Totale tile dopo il cleaning:\n"
            f"  Soglia Otsu:     {count_otsu:>10}\n"
            f"  Soglia manuale:  {count_new:>10}"
        )
        self.after_count_label.show()
        self.show_removed_btn.show()
        self._enable_threshold_radios()

        self.hist_total.axes.axvline(x=manual_th, ls="--", color="y", label="Soglia manuale")
        self.hist_total.draw()

    def _enable_threshold_radios(self) -> None:
        for radio in (self.otsu_radio, self.new_th_radio, self.manual_th_radio):
            radio.setEnabled(True)

    def _show_removed_tiles(self) -> None:
        if self.state.cleaning_obj:
            self.state.cleaning_obj.removed_class()

    # ------------------------------------------------------------------
    # Slot — create clean dataset
    # ------------------------------------------------------------------

    def _create_clean_dataset(self) -> None:
        """Copy the surviving tiles into the chosen folder under per-class subdirs."""
        import os

        for class_name in CLASS_NAMES:
            class_path = os.path.join(self.state.clean_save_path, class_name)
            os.makedirs(class_path, exist_ok=True)

        worker = LongRunningWorker(
            self.state.cleaning_obj.clean_js,
            self.state.selected_threshold,
            self.state.clean_save_path,
        )
        self.worker_started.emit(worker)
