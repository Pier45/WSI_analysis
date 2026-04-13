"""
Bayesian Analyzer — main application window.

Entry point:
    python image_viewer.py

Dependencies:
    PyQt5, multi_processing_analysis.StartAnalysis,
    progress_bar.Actions, Classification.Classification
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
import webbrowser
from typing import List, Optional, Tuple

from PyQt5.QtCore import QObject, QRunnable, QThreadPool, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QIcon, QImage, QPainter, QPalette, QPixmap
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDialog,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QToolBar,
)

from multi_processing_analysis import StartAnalysis
from progress_bar import Actions


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_TITLE = "Bayesian Analyzer"
APP_ICON = "icons/target.ico"
DEFAULT_MODEL = "Model_1_85aug.h5"
DEEPZOOM_URL = "http://127.0.0.1:5000/"
DEEPZOOM_SERVER_SCRIPT = "deepzoom/deepzoom_server.py"

MONTE_CARLO_OPTIONS: Tuple[int, ...] = (5, 25, 50)
DEFAULT_MONTE_CARLO = 5

ZOOM_IN_FACTOR = 1.25
ZOOM_OUT_FACTOR = 0.8
ZOOM_MIN = 0.2
ZOOM_MAX = 4.0

WELCOME_MESSAGE = (
    "Steps to start the analysis:\n\n"
    "1) File         → Select .svs  (or click the folder icon in the toolbar)\n\n"
    "2) Analysis → Start analysis  (or click the green arrow in the toolbar)"
)

# ---------------------------------------------------------------------------
# Thread infrastructure
# ---------------------------------------------------------------------------


class WorkerSignals(QObject):
    """Signals emitted by :class:`WorkerLong` during its lifecycle."""

    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    """
    Fire-and-forget thread wrapper.

    Runs *fn* in a thread-pool thread without lifecycle feedback.
    Use :class:`WorkerLong` when progress / error signals are needed.
    """

    def __init__(self, fn, *args, **kwargs) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    @pyqtSlot()
    def run(self) -> None:
        self._fn(*self._args, **self._kwargs)


class WorkerLong(QRunnable):
    """
    Thread wrapper with full lifecycle signals.

    Injects a ``progress_callback`` keyword argument into *fn* so the
    worker can emit integer progress values (0–100).

    Signals
    -------
    signals.finished  — emitted once the function returns or raises
    signals.error     — emitted with ``(exc_type, value, traceback_str)``
    signals.result    — emitted with the function's return value
    signals.progress  — emitted by the function via ``progress_callback.emit(n)``
    """

    def __init__(self, fn, *args, **kwargs) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self.signals = WorkerSignals()
        self._kwargs["progress_callback"] = self.signals.progress

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception:
            traceback.print_exc()
            exc_type, value = sys.exc_info()[:2]
            self.signals.error.emit((exc_type, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_screen_size() -> Tuple[int, int]:
    """
    Return the primary screen dimensions as ``(width, height)``.

    Falls back to a sensible default on non-Windows platforms.
    """
    try:
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except AttributeError:
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.geometry()
            return geo.width(), geo.height()
        return 1920, 1080


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class ImageViewer(QMainWindow):
    """
    Main application window for the Bayesian Analyzer.

    Workflow
    --------
    1. User opens an ``.svs`` file → thumbnail is generated and displayed.
    2. Tiles are created in background threads.
    3. User starts the analysis → classification runs in a background thread.
    4. Results and uncertainty maps can be viewed via the View menu / toolbar.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        super().__init__()

        # --- State ---
        self._work_dir: str = ""
        self._result_dir: str = ""
        self._svs_path: str = ""

        self._analysis_type: str = "fast"
        self._model_name: str = DEFAULT_MODEL
        self._monte_carlo_samples: int = DEFAULT_MONTE_CARLO

        # Tile-generation metadata (populated by StartAnalysis.tile_gen)
        self._tile_x_start: List[int] = []
        self._tile_x_stop: List[int] = []
        self._process_names: List[str] = []
        self._tile_start_idx: List[int] = []
        self._tile_stop_idx: List[int] = []
        self._tile_rows: int = 0
        self._svs_level: int = 1

        self._screen_size: Tuple[int, int] = _get_screen_size()
        self._thread_pool = QThreadPool()

        # --- UI ---
        self._printer = QPrinter()
        self._scale_factor: float = 0.0
        self._progress_dialog: Optional[QDialog] = None
        self._progress_ui: Optional[Actions] = None

        self._image_label = QLabel(WELCOME_MESSAGE)
        self._image_label.setFont(QFont("Helvetica", 15, QFont.Black))
        self._image_label.setBackgroundRole(QPalette.Dark)
        self._image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._image_label.setScaledContents(True)

        self._scroll_area = QScrollArea()
        self._scroll_area.setBackgroundRole(QPalette.Dark)
        self._scroll_area.setWidget(self._image_label)
        self._scroll_area.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self._scroll_area)

        self._toolbar = QToolBar("Main toolbar")
        self._toolbar.setStyleSheet("QToolBar { spacing: 15px; }")
        self.addToolBar(self._toolbar)

        self._create_actions()
        self._create_menus()

        self.setWindowTitle(APP_TITLE)
        self.setWindowIcon(QIcon(APP_ICON))
        self.showMaximized()

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def _open_file(self) -> None:
        """
        Prompt the user to select an ``.svs`` file, generate a thumbnail,
        and kick off tile-creation threads.
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Open SVS File", "", "SVS Files (*.svs)")
        if not file_path:
            return

        self._svs_path = file_path
        self._work_dir = self._initialise_analysis(file_path)
        self._result_dir = os.path.join(self._work_dir, "result", "")

        thumbnail_path = os.path.join(self._work_dir, "thumbnail", "th.png")
        image = QImage(thumbnail_path)

        if image.isNull():
            QMessageBox.warning(
                self,
                APP_TITLE,
                f"Cannot load image from:\n{thumbnail_path}",
            )
            return

        self._display_image(image)
        self._scale_factor = 1.0

        self._print_act.setEnabled(True)
        self._fit_to_window_act.setEnabled(True)
        self._start_analysis_act.setEnabled(True)
        self._deep_zoom_act.setEnabled(True)
        self._update_zoom_actions()

        self._start_tile_threads()

    def _initialise_analysis(self, file_path: str) -> str:
        """
        Open the SVS file with :class:`StartAnalysis`, generate the
        thumbnail and cache tile metadata.

        Returns
        -------
        str
            The working-directory path returned by ``StartAnalysis``.
        """
        analysis = (
            StartAnalysis(lev_sec=self._svs_level)
            if self._analysis_type == "slow"
            else StartAnalysis()
        )
        analysis.openSvs(file_path)
        work_dir: str = analysis.get_thumb()

        (
            self._tile_x_start,
            self._tile_x_stop,
            self._process_names,
            self._tile_start_idx,
            self._tile_stop_idx,
            self._tile_rows,
            self._svs_level,
        ) = analysis.tile_gen(state=0)

        logger.debug(
            "Tile metadata — x_start=%s x_stop=%s names=%s "
            "start_idx=%s stop_idx=%s rows=%d level=%d",
            self._tile_x_start,
            self._tile_x_stop,
            self._process_names,
            self._tile_start_idx,
            self._tile_stop_idx,
            self._tile_rows,
            self._svs_level,
        )
        return work_dir

    # ------------------------------------------------------------------
    # Image display
    # ------------------------------------------------------------------

    def _display_image(self, image: QImage) -> None:
        """Render *image* in the central label, scaling to fit the screen if needed."""
        self._image_label.setPixmap(QPixmap.fromImage(image))

        w, h = image.width(), image.height()
        longer_axis = 0 if w >= h else 1
        img_long = w if longer_axis == 0 else h
        screen_long = self._screen_size[longer_axis]

        if not self._fit_to_window_act.isChecked():
            self._image_label.adjustSize()

        if screen_long < img_long:
            ratio = screen_long / img_long - 0.04
            self._image_label.resize(ratio * self._image_label.pixmap().size())

    def _view_result(self, name: str, folder: str) -> None:
        """
        Load and display a result image.

        Parameters
        ----------
        name:
            Base file name (without extension).
        folder:
            One of ``"result"``, ``"th"`` (thumbnail), or ``"uncertainty"``.
        """
        path_map = {
            "result": os.path.join(self._result_dir, f"{name}.png"),
            "th": os.path.join(self._work_dir, "thumbnail", "th.png"),
            "uncertainty": os.path.join(self._result_dir, "uncertainty", f"{name}.png"),
        }
        image_path = path_map.get(folder, "")
        logger.debug("Viewing image: %s", image_path)

        image = QImage(image_path)
        if image.isNull():
            QMessageBox.warning(self, APP_TITLE, f"Could not load:\n{image_path}")
            return
        self._display_image(image)

    # ------------------------------------------------------------------
    # Progress dialog
    # ------------------------------------------------------------------

    def _show_progress(self, title: str) -> None:
        """Open the progress dialog with the given *title*."""
        self._progress_dialog = QDialog(self)
        self._progress_ui = Actions()
        self._progress_ui.initUI(self._progress_dialog, title)
        self._progress_dialog.show()

    def _hide_progress(self) -> None:
        """Safely close the progress dialog if it is open."""
        if self._progress_dialog:
            self._progress_dialog.hide()

    # ------------------------------------------------------------------
    # Tile-creation threads
    # ------------------------------------------------------------------

    def _folder_exists(self, name: str) -> bool:
        """Return ``True`` if *name* already exists inside the working directory."""
        return name in os.listdir(self._work_dir)

    def _create_tiles(
        self,
        tile_args: List,
        progress_callback,  # injected by WorkerLong
    ) -> str:
        """
        Worker function: create PNG tiles for one process partition.

        Parameters
        ----------
        tile_args:
            ``[x_start, x_stop, process_name, tile_start, tile_stop, n_rows, level]``
        progress_callback:
            Injected by :class:`WorkerLong`; call ``.emit(pct)`` to report progress.
        """
        x_start, x_stop, process_name, tile_start, tile_stop, n_rows, level = tile_args
        logger.debug(
            "Creating tiles — process=%s x=[%d,%d) start=%d stop=%d",
            process_name,
            x_start,
            x_stop,
            tile_start,
            tile_stop,
        )

        if self._folder_exists(process_name):
            time.sleep(1)
            progress_callback.emit(100)
            self._hide_progress()
            return f"Tile folder '{process_name}' already exists — skipping."

        analysis = (
            StartAnalysis(lev_sec=self._svs_level)
            if self._analysis_type == "slow"
            else StartAnalysis()
        )
        analysis.openSvs(self._svs_path)
        tile_source = analysis.tile_gen(state=1)

        folder_path = os.path.join(self._work_dir, process_name)
        os.mkdir(folder_path)

        is_primary = tile_start == 1
        current_index = tile_start

        for x in range(x_start, x_stop):
            for y in range(n_rows):
                tile = tile_source.get_tile(level, (x, y))
                tile_path = os.path.join(
                    folder_path,
                    f"tile_{tile_start}_{x}_{y}.png",
                )
                tile.save(tile_path, "PNG")
                current_index += 1

                if is_primary:
                    pct = int(100 * (current_index - 1) / tile_stop)
                    progress_callback.emit(pct)
                    if current_index - 1 == tile_stop:
                        time.sleep(1)
                        self._hide_progress()

        return "Tile creation complete."

    def _start_tile_threads(self) -> None:
        """Launch one :class:`WorkerLong` per process partition to create tiles."""
        if os.listdir(self._work_dir) and os.listdir(self._work_dir)[0] == self._process_names[0]:
            logger.debug("Tiles already present — skipping thread launch.")
            return

        self._show_progress(title="Tile creation")

        for idx in range(len(self._process_names)):
            tile_args = [
                self._tile_x_start[idx],
                self._tile_x_stop[idx],
                self._process_names[idx],
                self._tile_start_idx[idx],
                self._tile_stop_idx[idx],
                self._tile_rows,
                self._svs_level,
            ]
            worker = WorkerLong(self._create_tiles, tile_args)
            worker.signals.result.connect(lambda msg: logger.info("Tile worker result: %s", msg))
            worker.signals.progress.connect(lambda pct: logger.debug("Tile progress: %d%%", pct))
            if self._progress_ui:
                worker.signals.progress.connect(self._progress_ui.onCountChanged)
            worker.signals.finished.connect(lambda: logger.debug("Tile worker finished."))
            worker.signals.error.connect(self._on_worker_error)
            self._thread_pool.start(worker)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _start_analysis(self) -> None:
        """Run the Bayesian classification in a background thread (if not already done)."""
        if os.path.exists(self._result_dir):
            logger.info("Analysis results already present — loading existing results.")
            self._view_result("Pred_class", "result")
        else:
            # Import here to surface the ImportError clearly if the module is missing.
            from Classification import Classification  # noqa: PLC0415

            cls = Classification(self._work_dir, ty="analysis")
            self._show_progress(title="Analysis")
            worker = WorkerLong(
                cls.classify,
                self._analysis_type,
                self._monte_carlo_samples,
                self._model_name,
            )
            worker.signals.progress.connect(lambda pct: logger.debug("Analysis: %d%%", pct))
            if self._progress_ui:
                worker.signals.progress.connect(self._progress_ui.onCountChanged)
            worker.signals.finished.connect(self._on_analysis_complete)
            worker.signals.error.connect(self._on_worker_error)
            self._thread_pool.start(worker)

        self._enable_view_actions(True)

    def _on_analysis_complete(self) -> None:
        """Called when the analysis thread finishes successfully."""
        self._hide_progress()
        logger.info("Analysis thread complete.")
        self._view_result("Pred_class", "result")

    def _on_worker_error(self, error_tuple: tuple) -> None:
        """Display a critical dialog when a background worker raises an exception."""
        exc_type, value, tb_str = error_tuple
        logger.error("Worker error: %s\n%s", value, tb_str)
        QMessageBox.critical(
            self,
            "Background thread error",
            f"{exc_type.__name__}: {value}\n\nSee the console for the full traceback.",
        )

    # ------------------------------------------------------------------
    # Deep-zoom viewer
    # ------------------------------------------------------------------

    def _open_deep_zoom(self) -> None:
        """
        Start the deepzoom Flask server in a separate thread and open the
        browser once the server is ready.

        Spaces in the SVS path are unsupported by the deepzoom server and
        will cause an error; the user is warned explicitly.
        """
        if " " in self._svs_path:
            QMessageBox.critical(
                self,
                APP_TITLE,
                "The file path contains spaces:\n\n"
                f"{self._svs_path}\n\n"
                "The deepzoom viewer requires a path without spaces.\n"
                "Please move or rename the file and try again.",
            )
            return

        command = f"cmd /k python {DEEPZOOM_SERVER_SCRIPT} {self._svs_path}"
        self._thread_pool.start(Worker(lambda: os.system(command)))

        QMessageBox.information(
            self,
            APP_TITLE,
            "Your browser will open with the deepzoom viewer.\n\nPress OK to continue.",
        )
        self._thread_pool.start(Worker(self._open_browser))

    def _open_browser(self) -> None:
        """Open the deepzoom URL in the default browser after a short delay."""
        time.sleep(0.5)
        webbrowser.open_new_tab(DEEPZOOM_URL)

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def _print_image(self) -> None:
        """Open the print dialog and print the currently displayed pixmap."""
        dialog = QPrintDialog(self._printer, self)
        if dialog.exec_():
            painter = QPainter(self._printer)
            rect = painter.viewport()
            size = self._image_label.pixmap().size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self._image_label.pixmap().rect())
            painter.drawPixmap(0, 0, self._image_label.pixmap())

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def _zoom_in(self) -> None:
        self._scale_image(ZOOM_IN_FACTOR)

    def _zoom_out(self) -> None:
        self._scale_image(ZOOM_OUT_FACTOR)

    def _normal_size(self) -> None:
        self._image_label.adjustSize()
        self._scale_factor = 1.0

    def _fit_to_window(self) -> None:
        fit = self._fit_to_window_act.isChecked()
        self._scroll_area.setWidgetResizable(fit)
        if not fit:
            self._normal_size()
        self._update_zoom_actions()

    def _scale_image(self, factor: float) -> None:
        self._scale_factor *= factor
        self._image_label.resize(self._scale_factor * self._image_label.pixmap().size())
        self._adjust_scroll_bar(self._scroll_area.horizontalScrollBar(), factor)
        self._adjust_scroll_bar(self._scroll_area.verticalScrollBar(), factor)
        self._zoom_in_act.setEnabled(self._scale_factor < ZOOM_MAX)
        self._zoom_out_act.setEnabled(self._scale_factor > ZOOM_MIN)

    @staticmethod
    def _adjust_scroll_bar(scroll_bar, factor: float) -> None:
        scroll_bar.setValue(
            int(factor * scroll_bar.value() + (factor - 1) * scroll_bar.pageStep() / 2)
        )

    def _update_zoom_actions(self) -> None:
        enabled = not self._fit_to_window_act.isChecked()
        self._zoom_in_act.setEnabled(enabled)
        self._zoom_out_act.setEnabled(enabled)
        self._normal_size_act.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Analysis mode / Monte Carlo settings
    # ------------------------------------------------------------------

    def _set_fast_mode(self) -> None:
        self._analysis_type = "fast"
        self._slow_act.setChecked(False)

    def _set_slow_mode(self) -> None:
        self._analysis_type = "slow"
        self._fast_act.setChecked(False)
        self._open_file()

    def _set_monte_carlo(self, value: int) -> None:
        """Set the Monte Carlo sample count and uncheck the other options."""
        self._monte_carlo_samples = value
        for action, mc_value in zip(
            (self._mc5_act, self._mc25_act, self._mc50_act),
            MONTE_CARLO_OPTIONS,
        ):
            action.setChecked(mc_value == value)
        logger.debug("Monte Carlo samples set to %d", value)

    def _select_model(self) -> None:
        QMessageBox.information(
            self,
            APP_TITLE,
            "The selected model must produce three output classes: AC, AD, H.",
        )
        model_path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "H5 Files (*.h5)")
        if model_path:
            self._model_name = model_path
            logger.info("Model changed to: %s", model_path)

    # ------------------------------------------------------------------
    # View shortcuts
    # ------------------------------------------------------------------

    def _enable_view_actions(self, enabled: bool) -> None:
        for act in (
            self._v_no_overlay_act,
            self._v_all_classes_act,
            self._v_ac_act,
            self._v_ad_act,
            self._v_h_act,
            self._v_total_uncertainty_act,
            self._v_aleatoric_act,
            self._v_epistemic_act,
        ):
            act.setEnabled(enabled)

    # ------------------------------------------------------------------
    # About dialogs
    # ------------------------------------------------------------------

    def _about(self) -> None:
        QMessageBox.about(
            self,
            f"About {APP_TITLE}",
            "<p>The <b>Bayesian Analyzer</b> analyzes SVS pathology slides "
            "(typically ~1 GB) using a Bayesian neural network.</p>"
            "<p>It also supports deep-zoom visualization by opening a "
            "browser tab at maximum resolution.</p>",
        )

    def _about_deep_zoom(self) -> None:
        QMessageBox.about(
            self,
            "Deepzoom Viewer",
            "<p>The <b>Deepzoom</b> viewer lets you zoom into SVS files at "
            "maximum resolution inside your browser.</p>"
            f"<p>The server opens at <tt>{DEEPZOOM_URL}</tt>.</p>"
            "<p>The right panel shows metadata about the SVS file; additional "
            "images may appear on the left depending on the selected file.</p>",
        )

    # ------------------------------------------------------------------
    # Action / menu construction
    # ------------------------------------------------------------------

    def _create_actions(self) -> None:
        # File
        self._open_act = QAction(
            QIcon("icons/folder.png"), "Select SVS", self,
            shortcut="Ctrl+O", triggered=self._open_file,
        )
        self._print_act = QAction(
            "&Print…", self,
            shortcut="Ctrl+P", enabled=False, triggered=self._print_image,
        )
        self._exit_act = QAction(
            QIcon("icons/exit.ico"), "E&xit", self,
            shortcut="Ctrl+Q", triggered=self.close,
        )

        # Zoom
        self._zoom_in_act = QAction(
            QIcon("icons/zoomin.ico"), "Zoom &In (25%)", self,
            shortcut="Ctrl++", enabled=False, triggered=self._zoom_in,
        )
        self._zoom_out_act = QAction(
            QIcon("icons/zoomout.ico"), "Zoom &Out (25%)", self,
            shortcut="Ctrl+-", enabled=False, triggered=self._zoom_out,
        )
        self._normal_size_act = QAction(
            "&Normal Size", self,
            shortcut="Ctrl+N", enabled=False, triggered=self._normal_size,
        )
        self._fit_to_window_act = QAction(
            "&Fit to Window", self,
            shortcut="Ctrl+F", enabled=False,
            checkable=True, triggered=self._fit_to_window,
        )

        # Analysis
        self._start_analysis_act = QAction(
            QIcon("icons/start.ico"), "Start Analysis", self,
            shortcut="Ctrl+R", enabled=False, triggered=self._start_analysis,
        )
        self._fast_act = QAction(
            "Fast mode", self,
            checkable=True, checked=True, enabled=True, triggered=self._set_fast_mode,
        )
        self._slow_act = QAction(
            "Slow mode", self,
            checkable=True, enabled=True, triggered=self._set_slow_mode,
        )

        # Model / Monte Carlo
        self._select_model_act = QAction(
            "Change model", self, enabled=True, triggered=self._select_model,
        )
        self._mc5_act = QAction(
            "5", self, checkable=True, checked=True,
            triggered=lambda: self._set_monte_carlo(5),
        )
        self._mc25_act = QAction(
            "25", self, checkable=True,
            triggered=lambda: self._set_monte_carlo(25),
        )
        self._mc50_act = QAction(
            "50", self, checkable=True,
            triggered=lambda: self._set_monte_carlo(50),
        )

        # View results
        self._v_no_overlay_act = QAction(
            "No overlay", self, enabled=False,
            triggered=lambda: self._view_result("no_ov", "th"),
        )
        self._v_all_classes_act = QAction(
            "All classes", self, enabled=False,
            triggered=lambda: self._view_result("Pred_class", "result"),
        )
        self._v_ac_act = QAction(
            QIcon("icons/AC.ico"), "AC only", self, enabled=False,
            triggered=lambda: self._view_result("AC", "result"),
        )
        self._v_ad_act = QAction(
            QIcon("icons/AD.ico"), "AD only", self, enabled=False,
            triggered=lambda: self._view_result("AD", "result"),
        )
        self._v_h_act = QAction(
            QIcon("icons/H.ico"), "H only", self, enabled=False,
            triggered=lambda: self._view_result("H", "result"),
        )
        self._v_total_uncertainty_act = QAction(
            "Total uncertainty", self, enabled=False,
            triggered=lambda: self._view_result("tot", "uncertainty"),
        )
        self._v_aleatoric_act = QAction(
            "Aleatoric uncertainty", self, enabled=False,
            triggered=lambda: self._view_result("ale", "uncertainty"),
        )
        self._v_epistemic_act = QAction(
            "Epistemic uncertainty", self, enabled=False,
            triggered=lambda: self._view_result("epi", "uncertainty"),
        )

        # Deep zoom / help
        self._deep_zoom_act = QAction(
            QIcon("icons/binocul.ico"), "Deep Zoom Viewer", self,
            shortcut="Ctrl+D", enabled=False, triggered=self._open_deep_zoom,
        )
        self._about_act = QAction("&About", self, triggered=self._about)
        self._about_qt_act = QAction(
            "About &Qt", self, triggered=QApplication.instance().aboutQt,
        )
        self._info_deep_act = QAction(
            "&Deepzoom info", self, triggered=self._about_deep_zoom,
        )

    def _create_menus(self) -> None:
        # File menu
        file_menu = QMenu("&File", self)
        file_menu.addAction(self._open_act)
        file_menu.addAction(self._print_act)
        file_menu.addSeparator()
        file_menu.addAction(self._exit_act)

        # Analysis menu
        analysis_menu = QMenu("&Analysis", self)
        analysis_menu.addAction(self._fast_act)
        analysis_menu.addAction(self._slow_act)
        analysis_menu.addSeparator()

        settings_menu = QMenu("&Settings", self)
        settings_menu.addAction(self._select_model_act)

        mc_menu = QMenu("&Monte Carlo Samples", self)
        mc_menu.addAction(self._mc5_act)
        mc_menu.addAction(self._mc25_act)
        mc_menu.addAction(self._mc50_act)
        settings_menu.addMenu(mc_menu)

        analysis_menu.addMenu(settings_menu)
        analysis_menu.addSeparator()
        analysis_menu.addAction(self._start_analysis_act)

        # View menu
        view_menu = QMenu("&View", self)
        view_menu.addAction(self._v_no_overlay_act)
        view_menu.addAction(self._v_all_classes_act)
        view_menu.addSeparator()
        view_menu.addAction(self._v_ac_act)
        view_menu.addAction(self._v_ad_act)
        view_menu.addAction(self._v_h_act)
        view_menu.addSeparator()
        view_menu.addAction(self._v_total_uncertainty_act)
        view_menu.addAction(self._v_aleatoric_act)
        view_menu.addAction(self._v_epistemic_act)
        view_menu.addSeparator()
        view_menu.addAction(self._zoom_in_act)
        view_menu.addAction(self._zoom_out_act)
        view_menu.addAction(self._normal_size_act)
        view_menu.addSeparator()
        view_menu.addAction(self._fit_to_window_act)

        # Options menu
        options_menu = QMenu("&Options", self)
        options_menu.addAction(self._deep_zoom_act)
        options_menu.addAction(self._info_deep_act)

        # Help menu
        help_menu = QMenu("&Help", self)
        help_menu.addAction(self._about_act)
        help_menu.addAction(self._about_qt_act)

        for menu in (file_menu, analysis_menu, view_menu, options_menu, help_menu):
            self.menuBar().addMenu(menu)

        # Toolbar
        for item in (
            self._open_act,
            self._start_analysis_act,
            None,  # separator
            self._zoom_in_act,
            self._zoom_out_act,
            None,
            self._deep_zoom_act,
            None,
            self._v_ac_act,
            self._v_ad_act,
            self._v_h_act,
            None,
            self._exit_act,
        ):
            if item is None:
                self._toolbar.addSeparator()
            else:
                self._toolbar.addAction(item)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())