"""
Bayesian Datacleaning Application
Application for cleaning medical image datasets via Bayesian uncertainty analysis.
"""

import sys
import os
import time
import traceback
import matplotlib

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QApplication,
    QLabel, QScrollArea, QMenu, QAction, QFileDialog, QProgressBar,
    QListWidget, QLineEdit, QButtonGroup, QMessageBox, QPushButton,
    QTabWidget, QRadioButton, QFrame, QCheckBox
)
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap, QFont, QIcon
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QObject, QRunnable, QThreadPool, Qt

from multi_processing_analysis import StartAnalysis
from DropOut import BayesianDropoutCNN
from Kl import ModelKl
from Classification import Classification
from uncertainty_analysis import Th
from test_widget import TestTab

matplotlib.use('Qt5Agg')

# --- Constants ---
DEFAULT_TRAIN_PATH = "test/train"
DEFAULT_VAL_PATH = "test/val"
DEFAULT_MODEL_PATH = "Model_1_85aug.h5"
DEFAULT_WORK_PATH = "test"
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 100
DEFAULT_MONTE_CARLO_SAMPLES = 5
DEFAULT_TILE_SIZE = 256
DEFAULT_TILE_LEVEL = 0
MANUAL_THRESHOLD_MIN = 0.1
MANUAL_THRESHOLD_MAX = 1.0
KNOWN_CLASSES = {"ac", "ad", "h"}
APP_ICON_PATH = "icons/target.ico"
APP_STYLE_PATH = "styles/stileor.css"
TUTORIAL_MESSAGE = (
    "The program is divided into tabs; follow the tab sequence to ensure everything works correctly.\n\n"
    "Tab 'Get tiles': select the working folder, then the train/val/test folders "
    "(SVS files must be organised into subfolders AC, AD, H). Press Start.\n\n"
    "Tab 'Training': configure the model parameters and start training.\n\n"
    "Press OK to continue."
)


# ---------------------------------------------------------------------------
# Reusable UI components
# ---------------------------------------------------------------------------

class HorizontalLine(QFrame):
    """Decorative horizontal separator."""

    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)


class VerticalLine(QFrame):
    """Decorative vertical separator."""

    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)


class MatplotlibCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for rendering inline histograms."""

    def __init__(self, title: str, parent=None, width: int = 10, height: int = 8, dpi: int = 100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.set_facecolor("#323232")
        self.axes.set_title(title)
        self.axes.set_xlim(0, 1)
        super().__init__(fig)


# ---------------------------------------------------------------------------
# Threading infrastructure
# ---------------------------------------------------------------------------

class WorkerSignals(QObject):
    """Signals emitted by LongRunningWorker to communicate with the main thread."""

    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)
    intermediate_result = pyqtSignal(object)


class LongRunningWorker(QRunnable):
    """
    Generic worker for long-running operations on background threads.

    Accepts a callable and its arguments; emits progress, result, and error
    signals without blocking the UI thread.
    """

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.kwargs["progress_callback"] = self.signals.progress
        self.kwargs["view"] = self.signals.intermediate_result

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
            exc_type, value = sys.exc_info()[:2]
            self.signals.error.emit((exc_type, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Main application window for Bayesian Datacleaning."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bayesian Datacleaning")
        self.setWindowIcon(QIcon(APP_ICON_PATH))
        if os.path.exists(APP_STYLE_PATH):
            self.setStyleSheet(open(APP_STYLE_PATH).read())
        else:
            print(f"Warning: style file {APP_STYLE_PATH} not found. Using default style.")

        self.central_widget = MainTabWidget(self)
        self.setCentralWidget(self.central_widget)

        self._create_actions()
        self._create_menus()
        self.showMaximized()

    def _create_actions(self):
        self.tutorial_action = QAction("&Tutorial", self, triggered=self._show_tutorial)
        self.exit_action = QAction("Exit", self, triggered=self.close)

    def _create_menus(self):
        file_menu = QMenu("&File", self)
        file_menu.addAction(self.exit_action)

        about_menu = QMenu("About", self)
        about_menu.addAction(self.tutorial_action)

        self.menuBar().addMenu(file_menu)
        self.menuBar().addMenu(about_menu)

    def _show_tutorial(self):
        QMessageBox.information(self, "Bayesian Datacleaner", TUTORIAL_MESSAGE)


# ---------------------------------------------------------------------------
# Main tab widget
# ---------------------------------------------------------------------------

class MainTabWidget(QWidget):
    """
    Central widget that coordinates all application tabs.

    Manages shared state across tabs (paths, training parameters, analysis
    results) and orchestrates background threads.
    """

    def __init__(self, parent):
        super().__init__(parent)

        # --- Shared state ---
        self.train_path = DEFAULT_TRAIN_PATH
        self.val_path = DEFAULT_VAL_PATH
        self.test_path = ""
        self.model_path = DEFAULT_MODEL_PATH
        self.work_path = DEFAULT_WORK_PATH
        self.tiles_train_path = ""
        self.tiles_val_path = ""
        self.tiles_test_path = ""
        self.clean_save_path = ""

        self.epochs = DEFAULT_EPOCHS
        self.batch_size = DEFAULT_BATCH_SIZE
        self.monte_carlo_samples = DEFAULT_MONTE_CARLO_SAMPLES
        self.model_type = "drop"
        self.use_augmentation = False
        self.selected_threshold = ""

        self.train_json = "new_train_js.txt"
        self.val_json = "new_val_js.txt"
        self.test_json = ""

        self.aleatoric_values = []
        self.epistemic_values = []
        self.total_uncertainty_values = []
        self.threshold_flag = 0
        self.cleaning_obj = None

        self.thread_pool = QThreadPool()
        self.training_log = f"Start  {time.asctime(time.localtime(time.time()))}"

        self._build_ui()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        root_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()

        self.tab_tiles = QWidget()
        self.tab_training = QWidget()
        self.tab_uncertainty = QWidget()
        self.tab_cleaning = QWidget()
        self.tab_testing = TestTab(parent=self)

        self.tabs.addTab(self.tab_tiles, "Get Tiles")
        self.tabs.addTab(self.tab_training, "Training")
        self.tabs.addTab(self.tab_uncertainty, "Uncertainty analysis")
        self.tabs.addTab(self.tab_cleaning, "Data cleaning")
        self.tabs.addTab(self.tab_testing, "Testing")

        self._build_tab_tiles()
        self._build_tab_training()
        self._build_tab_uncertainty()
        self._build_tab_cleaning()

        root_layout.addWidget(self.tabs)
        self.setLayout(root_layout)

    def _build_tab_tiles(self):
        """Builds the 'Get Tiles' tab for dataset folder selection and tiling."""
        layout = QVBoxLayout()
        bold_font = QFont("Helvetica", 15, QFont.Bold)

        # Working folder selection
        work_folder_row = self._make_label_button_row(
            "First things first, select a folder to save all data:",
            "Select folder",
            self._select_work_folder,
        )

        # One section per dataset
        sections = [
            ("TRAINING SET", self._select_train_folder),
            ("VALIDATION SET", self._select_val_folder),
            ("TEST SET", self._select_test_folder),
        ]
        self.dataset_list_widgets = {}
        self.progress_bars = {}
        dataset_names = ["train", "val", "test"]

        layout.addLayout(work_folder_row)

        for name, select_fn in zip(dataset_names, [s[1] for s in sections]):
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

        self.tab_tiles.setLayout(layout)

    def _build_tab_training(self):
        """Builds the 'Training' tab for model configuration and launch."""
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
        self.epoch_label = QLabel(f"Default epochs: {self.epochs}")
        epoch_ok_btn = QPushButton("Ok")
        epoch_ok_btn.clicked.connect(self._confirm_epochs)

        epoch_row = QHBoxLayout()
        epoch_row.addWidget(QLabel("Number of epochs:  "))
        epoch_row.addWidget(self.epoch_input)
        epoch_row.addWidget(epoch_ok_btn)
        epoch_row.addWidget(self.epoch_label)

        # Batch size
        self.batch_input = QLineEdit()
        self.batch_label = QLabel(f"Default value: {self.batch_size}")
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

        # Optional folders
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

        self.tab_training.setLayout(layout)

    def _build_tab_uncertainty(self):
        """Builds the 'Uncertainty analysis' tab for MC Dropout classification."""
        layout = QVBoxLayout()
        bold_font = QFont("Helvetica", 15, QFont.Bold)

        description = QLabel(
            "This step computes uncertainty values (aleatoric, epistemic, total)."
        )

        # Monte Carlo samples
        self.monte_input = QLineEdit()
        self.monte_label = QLabel(f"Default value: {self.monte_carlo_samples}")
        monte_ok_btn = QPushButton("Ok")
        monte_ok_btn.clicked.connect(self._confirm_monte_carlo)

        monte_row = QHBoxLayout()
        monte_row.addWidget(QLabel("Monte Carlo samples:"))
        monte_row.addWidget(self.monte_input)
        monte_row.addWidget(monte_ok_btn)
        monte_row.addWidget(self.monte_label)

        self.uncertainty_progress_bar = QProgressBar()

        # Start buttons for each dataset
        dataset_labels = ["TRAINING SET", "VALIDATION SET", "TEST SET"]
        start_fns = [self._run_uncertainty_train, self._run_uncertainty_val, self._run_uncertainty_test]

        layout.addWidget(description, alignment=Qt.AlignTop)
        layout.addWidget(HorizontalLine())
        layout.addLayout(monte_row)

        for label_text, fn in zip(dataset_labels, start_fns):
            title = QLabel(label_text)
            title.setFont(bold_font)
            start_btn = QPushButton("Start")
            start_btn.clicked.connect(fn)
            layout.addWidget(title)
            layout.addWidget(start_btn)
            layout.addWidget(HorizontalLine())

        layout.addStretch(1)
        layout.addWidget(self.uncertainty_progress_bar)
        self.tab_uncertainty.setLayout(layout)

    def _build_tab_cleaning(self):
        """Costruisce il tab 'Data cleaning' con istogrammi e selezione soglia."""
        layout = QVBoxLayout()

        # Selezione dataset
        self.train_dataset_radio = QRadioButton("Training set")
        self.train_dataset_radio.toggled.connect(self._load_train_histograms)
        self.val_dataset_radio = QRadioButton("Validation set")
        self.val_dataset_radio.toggled.connect(self._load_val_histograms)
        dataset_group = QButtonGroup()
        dataset_group.addButton(self.train_dataset_radio)
        dataset_group.addButton(self.val_dataset_radio)

        select_dataset_row = QHBoxLayout()
        select_dataset_row.addWidget(QLabel("Seleziona il dataset da analizzare:"))
        select_dataset_row.addWidget(self.train_dataset_radio)
        select_dataset_row.addWidget(self.val_dataset_radio)

        # Canvas istogrammi
        self.hist_aleatoric = MatplotlibCanvas("Aleatoric uncertainty", self, width=5, height=4)
        self.hist_epistemic = MatplotlibCanvas("Epistemic uncertainty", self, width=5, height=4)
        self.hist_total = MatplotlibCanvas("Total uncertainty", self, width=5, height=4)

        right_hists = QVBoxLayout()
        right_hists.addWidget(self.hist_aleatoric)
        right_hists.addWidget(self.hist_epistemic)

        hists_row = QHBoxLayout()
        hists_row.addWidget(self.hist_total)
        hists_row.addLayout(right_hists)

        # Statistiche dataset
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

        # Selezione modalità soglia
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

        mode_group = QButtonGroup()
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

        # Selezione soglia finale
        self.otsu_radio = QRadioButton("Soglia Otsu")
        self.otsu_radio.setEnabled(False)
        self.otsu_radio.toggled.connect(lambda: self._set_threshold_type("otsu"))
        self.new_th_radio = QRadioButton("Nuova soglia")
        self.new_th_radio.setEnabled(False)
        self.new_th_radio.toggled.connect(lambda: self._set_threshold_type("new"))
        self.manual_th_radio = QRadioButton("Soglia manuale")
        self.manual_th_radio.setEnabled(False)
        self.manual_th_radio.toggled.connect(lambda: self._set_threshold_type("manual"))

        threshold_group = QButtonGroup()
        threshold_group.addButton(self.otsu_radio)
        threshold_group.addButton(self.new_th_radio)
        threshold_group.addButton(self.manual_th_radio)

        threshold_row = QHBoxLayout()
        for radio in (self.otsu_radio, self.new_th_radio, self.manual_th_radio):
            threshold_row.addWidget(radio)
        threshold_row.addStretch(1)

        # Cartella di destinazione e avvio
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
        self.tab_cleaning.setLayout(layout)

    # -----------------------------------------------------------------------
    # Helper UI
    # -----------------------------------------------------------------------

    def _make_label_button_row(self, label_text: str, button_text: str, callback) -> QHBoxLayout:
        """Crea una riga orizzontale con etichetta e pulsante."""
        row = QHBoxLayout()
        row.addWidget(QLabel(label_text))
        btn = QPushButton(button_text)
        btn.clicked.connect(callback)
        row.addWidget(btn)
        return row

    # -----------------------------------------------------------------------
    # Slot — selezione cartelle
    # -----------------------------------------------------------------------

    def _select_work_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleziona cartella di lavoro")
        if folder:
            self.work_path = folder
            self.tiles_train_path = os.path.join(folder, "train")
            self.tiles_val_path = os.path.join(folder, "val")
            self.tiles_test_path = os.path.join(folder, "test")

    def _select_train_folder(self):
        self.train_path = self._select_dataset_folder("train")

    def _select_val_folder(self):
        self.val_path = self._select_dataset_folder("val")

    def _select_test_folder(self):
        self.test_path = self._select_dataset_folder("test")

    def _select_dataset_folder(self, dataset_name: str) -> str:
        """
        Apre un dialogo di selezione cartella e popola le QListWidget
        con i file trovati nelle sottocartelle per classe (AC, AD, H).
        """
        folder = QFileDialog.getExistingDirectory(self, "Seleziona cartella")
        if not folder:
            return getattr(self, f"{dataset_name}_path", "")

        list_widgets = self.dataset_list_widgets.get(dataset_name, {})
        known_classes = {k.strip().upper() for k in KNOWN_CLASSES}

        try:
            for class_name in os.listdir(folder):
                key = class_name.upper()
                if key not in {k.upper() for k in KNOWN_CLASSES}:
                    print(f"||||| > Ignoro elemento non-class: {repr(key)} in {KNOWN_CLASSES} in dataset {class_name}")
                    continue
                files = os.listdir(os.path.join(folder, class_name))
                # if widget:
                list_widgets[key].addItems(files) # <--- Qui avviene il popolamento del widget con i nomi dei file BUG
                print(f"||||| > Popolamento widget per {dataset_name} - classe {class_name} - files {files}")
        except OSError as e:
            print(f"Errore nella lettura della cartella {folder}: {e}")

        return folder

    def _retrieve_train_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleziona cartella train")
        if folder:
            self.tiles_train_path = folder

    def _retrieve_val_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleziona cartella val")
        if folder:
            self.tiles_val_path = folder

    def _select_save_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleziona cartella di salvataggio")
        if folder:
            self.clean_save_path = folder
            self.create_dataset_btn.setEnabled(True)

    # -----------------------------------------------------------------------
    # Slot — configurazione parametri
    # -----------------------------------------------------------------------

    def _set_model_type(self, model_type: str):
        self.model_type = model_type

    def _toggle_augmentation(self, state: int):
        self.use_augmentation = state == Qt.Checked

    def _confirm_epochs(self):
        text = self.epoch_input.text()
        if text.isdecimal() and int(text) > 0:
            self.epochs = int(text)
            self.epoch_label.setText(f"Epoche: {self.epochs}")

    def _confirm_batch_size(self):
        text = self.batch_input.text()
        if text.isdecimal() and int(text) > 0:
            self.batch_size = int(text)
            self.batch_label.setText(f"Batch: {self.batch_size}")

    def _confirm_monte_carlo(self):
        text = self.monte_input.text()
        if text.isdecimal() and int(text) > 0:
            self.monte_carlo_samples = int(text)
            self.monte_label.setText(f"MC samples: {self.monte_carlo_samples}")

    def _set_threshold_type(self, threshold_type: str):
        self.selected_threshold = threshold_type

    # -----------------------------------------------------------------------
    # Slot — Tab 1: tiling
    # -----------------------------------------------------------------------

    def _start_tiling(self):
        """Avvia la segmentazione in tile per tutti e tre i dataset in parallelo."""
        datasets = [
            ("train", self.train_path, self.progress_bars["train"]),
            ("val",   self.val_path,   self.progress_bars["val"]),
            ("test",  self.test_path,  self.progress_bars["test"]),
        ]
        for name, path, progress_bar_list in datasets:
            self._tile_dataset(name, path, progress_bar_list)

    def _tile_dataset(self, dataset_name: str, source_path: str, progress_bars: list):
        """Lancia un worker per la segmentazione di un singolo dataset."""
        if not source_path or not os.path.isdir(source_path):
            print(f"Percorso non valido per {dataset_name}: {source_path}")
            return

        try:
            class_dirs = os.listdir(source_path)
        except OSError as e:
            print(f"Impossibile leggere {source_path}: {e}")
            return

        for idx, class_dir in enumerate(class_dirs):
            save_folder = os.path.join(self.work_path, dataset_name, class_dir)
            class_path = os.path.join(source_path, class_dir)
            os.makedirs(save_folder, exist_ok=True)

            tiler = StartAnalysis(tile_size=DEFAULT_TILE_SIZE, lev_sec=DEFAULT_TILE_LEVEL)
            worker = LongRunningWorker(tiler.list_files, class_path, save_folder)
            worker.signals.result.connect(self._on_worker_result)
            worker.signals.progress.connect(self._on_worker_progress)
            if idx < len(progress_bars):
                worker.signals.progress.connect(progress_bars[idx].setValue)
            worker.signals.finished.connect(self._on_worker_finished)
            self.thread_pool.start(worker)

    # -----------------------------------------------------------------------
    # Slot — Tab 2: training
    # -----------------------------------------------------------------------

    def _start_training(self):
        """Avvia il training del modello selezionato in un thread separato."""
        self.training_log_label.setText(
            "Avvio del training in corso, ulteriori informazioni saranno disponibili a breve..."
        )
        timestamp = time.strftime("%Y_%m%d_%H%M%S")
        aug = 1 if self.use_augmentation else 0

        if self.model_type == "drop":
            model_filename = f"ModelDrop-{timestamp}.h5"
            self.model_path = os.path.join(self.work_path, model_filename)
            model_obj = BayesianDropoutCNN(
                n_model=self.model_path, epochs=self.epochs,
                path_train=self.tiles_train_path, path_val=self.tiles_val_path,
                b_dim=self.batch_size, aug=aug,
            )
        else:
            model_filename = f"ModelKl-{timestamp}.h5"
            self.model_path = os.path.join(self.work_path, model_filename)
            model_obj = ModelKl(
                n_model=self.model_path, epochs=self.epochs,
                path_train=self.tiles_train_path, path_val=self.tiles_val_path,
                b_dim=self.batch_size, aug=aug,
            )

        worker = LongRunningWorker(model_obj.start_train)
        worker.signals.result.connect(self._on_worker_result)
        worker.signals.progress.connect(self.training_progress_bar.setValue)
        worker.signals.intermediate_result.connect(self._update_training_log)
        worker.signals.finished.connect(self._on_training_complete)
        self.thread_pool.start(worker)

    def _update_training_log(self, value):
        """Aggiorna il log del training, filtrando le righe di epoca."""
        if "Epoch" in str(value):
            self.training_log += f"\n{value}"
            self.training_log_label.setText(self.training_log)
        else:
            self.training_log_label.setText(f"{self.training_log}\n{value}")

    # -----------------------------------------------------------------------
    # Slot — Tab 3: uncertainty analysis
    # -----------------------------------------------------------------------

    def _run_uncertainty_train(self):
        self._run_uncertainty_analysis(self.tiles_train_path, "train")
        self.train_json = self._build_json_path(self.tiles_train_path)
        self.tab_testing.traincm.setEnabled(True)
        self.tab_testing.get_paths(train=self.train_json)

    def _run_uncertainty_val(self):
        self._run_uncertainty_analysis(self.tiles_val_path, "val")
        self.val_json = self._build_json_path(self.tiles_val_path)
        self.tab_testing.valcm.setEnabled(True)
        self.tab_testing.get_paths(val=self.val_json)

    def _run_uncertainty_test(self):
        self._run_uncertainty_analysis(self.tiles_test_path, "test")
        self.test_json = self._build_json_path(self.tiles_test_path)
        self.tab_testing.testcm.setEnabled(True)
        self.tab_testing.get_paths(test=self.test_json)

    def _run_uncertainty_analysis(self, data_path: str, dataset_name: str):
        """Lancia il worker per la classificazione Monte Carlo su un dataset."""
        classifier = Classification(data_path, ty="datacleaning")
        worker = LongRunningWorker(
            classifier.classify, "datacleaning",
            self.monte_carlo_samples, self.model_path,
        )
        worker.signals.result.connect(self._on_worker_result)
        worker.signals.progress.connect(self._on_worker_progress)
        worker.signals.progress.connect(self.uncertainty_progress_bar.setValue)
        worker.signals.finished.connect(self._on_worker_finished)
        self.thread_pool.start(worker)

    def _build_json_path(self, base_path: str) -> str:
        return os.path.join(
            base_path,
            f"dictionary_monte_{self.monte_carlo_samples}_js.txt",
        )

    # -----------------------------------------------------------------------
    # Slot — Tab 4: data cleaning
    # -----------------------------------------------------------------------

    def _load_train_histograms(self):
        self._draw_histograms(self.train_json, "train")

    def _load_val_histograms(self):
        self._draw_histograms(self.val_json, "val")

    def _draw_histograms(self, json_path: str, dataset_name: str):
        """Carica i dati di incertezza dal JSON e aggiorna i tre istogrammi."""
        self.cleaning_obj = Th(json_path, dataset_name)
        self.aleatoric_values, self.epistemic_values, self.total_uncertainty_values = (
            self.cleaning_obj.create_list()
        )
        total_count = len(self.total_uncertainty_values)
        max_val = max(self.total_uncertainty_values)
        n_bins = max(1, round(total_count / 100))

        for canvas, values, title in (
            (self.hist_total,     self.total_uncertainty_values, "Total uncertainty"),
            (self.hist_aleatoric, self.aleatoric_values,         "Aleatoric uncertainty"),
            (self.hist_epistemic, self.epistemic_values,         "Epistemic uncertainty"),
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

    def _unlock_threshold_mode(self):
        """Abilita i radio button di selezione modalità dopo il caricamento dei dati."""
        self.auto_radio.setEnabled(True)
        self.manual_radio.setEnabled(True)
        self.auto_radio.setChecked(False)
        self.manual_radio.setChecked(False)
        self._control_radio.setChecked(True)
        self.threshold_flag = 0

    def _on_auto_mode_selected(self):
        """Calcola le soglie automatiche (Otsu + nuova) e aggiorna il grafico."""
        if not self.cleaning_obj:
            return
        self.cleaning_obj.otsu()
        new_th, otsu_th, count_new, count_otsu = self.cleaning_obj.th_managment()

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

        if self.threshold_flag == 0:
            self.hist_total.axes.legend(prop={"size": 10})
            self.threshold_flag = 1
        self.hist_total.draw()

        if self.auto_radio.isChecked():
            self.manual_threshold_input.hide()
            self.apply_manual_threshold_btn.hide()
            self._enable_threshold_radios()

    def _on_manual_mode_selected(self):
        if self.manual_radio.isChecked():
            self.manual_threshold_input.show()
            self.apply_manual_threshold_btn.show()

    def _apply_manual_threshold(self):
        """Applica il valore di soglia inserito manualmente dall'utente."""
        try:
            threshold = float(self.manual_threshold_input.text())
        except ValueError:
            return

        if not (MANUAL_THRESHOLD_MIN < threshold < MANUAL_THRESHOLD_MAX):
            return

        self.selected_threshold = threshold
        self.cleaning_obj.otsu()
        manual_th, otsu_th, count_new, count_otsu = self.cleaning_obj.th_managment(threshold)

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

    def _enable_threshold_radios(self):
        for radio in (self.otsu_radio, self.new_th_radio, self.manual_th_radio):
            radio.setEnabled(True)

    def _show_removed_tiles(self):
        if self.cleaning_obj:
            self.cleaning_obj.removed_class()

    def _create_clean_dataset(self):
        """Crea il dataset pulito copiando le tile nella cartella di destinazione."""
        for class_name in ("AC", "H", "AD"):
            class_path = os.path.join(self.clean_save_path, class_name)
            os.makedirs(class_path, exist_ok=True)

        worker = LongRunningWorker(
            self.cleaning_obj.clean_js,
            self.selected_threshold,
            self.clean_save_path,
        )
        worker.signals.result.connect(self._on_worker_result)
        worker.signals.progress.connect(self._on_worker_progress)
        worker.signals.progress.connect(self.copy_progress_bar.setValue)
        worker.signals.finished.connect(self._on_worker_finished)
        self.thread_pool.start(worker)

    # -----------------------------------------------------------------------
    # Slot generici per i worker
    # -----------------------------------------------------------------------

    def _on_worker_result(self, result):
        self.training_log_label.setText(str(result))
        print(result)

    def _on_worker_progress(self, value: int):
        print(f"{value}% completato")

    def _on_worker_finished(self):
        print("Thread completato.")

    def _on_training_complete(self):
        self.training_log_label.setText(
            "Training completato! La history è stata salvata nella cartella di lavoro."
        )
        print("Training completato.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())