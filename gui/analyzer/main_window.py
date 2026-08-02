"""Main window of the Bayesian Analyzer.

Thin :class:`QMainWindow` that wires together the helpers in
``gui.analyzer.<actions|menus|image_display|tile_worker|analysis_worker|
deepzoom|about_dialogs>``. Each slot on this class is a 1-3 line
forwarder to one of those modules; the heavy logic lives there.
"""

from __future__ import annotations

import logging
import os

from PyQt5.QtCore import QSize, Qt, QThreadPool
from PyQt5.QtGui import QFont, QIcon, QPalette
from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QToolBar,
)

from src.multi_processing_analysis import StartAnalysis
from src.progress_bar import Actions

from .about_dialogs import about
from .actions import create_actions, set_monte_carlo
from .analysis_worker import on_analysis_complete, select_model, start_analysis
from .constants import APP_ICON, APP_TITLE, WELCOME_MESSAGE
from .deepzoom import about_deep_zoom, open_deep_zoom
from .image_display import (
    display_image,
    enable_view_actions,
    fit_to_window,
    normal_size,
    print_image,
    scale_image,
    update_zoom_actions,
    view_result,
    zoom_in,
    zoom_out,
)
from .menus import create_menus, populate_toolbar
from .state import AnalyzerState
from .tile_worker import (
    on_tile_worker_finished,
    show_worker_error,
    start_tile_threads,
)

logger = logging.getLogger(__name__)


def _get_screen_size() -> tuple[int, int]:
    """Return the primary screen dimensions as ``(width, height)``.

    Falls back to a sensible default on non-Windows platforms.
    """
    try:
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except AttributeError:
        from PyQt5.QtWidgets import QApplication
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.geometry()
            return geo.width(), geo.height()
        return 1920, 1080


class ImageViewer(QMainWindow):
    """Main application window for the Bayesian Analyzer."""

    def __init__(self) -> None:
        super().__init__()

        # --- Shared state ---
        self.state = AnalyzerState()
        self._screen_size: tuple[int, int] = _get_screen_size()
        self._thread_pool = QThreadPool()
        self._printer = QPrinter()
        self._progress_dialog: QDialog | None = None
        self._progress_ui: Actions | None = None
        self._deepzoom_proc = None  # set on _open_deep_zoom

        # --- UI skeleton ---
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
        self._toolbar.setIconSize(QSize(64, 64))
        self._toolbar.setStyleSheet("QToolBar { spacing: 15px; }")
        self.addToolBar(self._toolbar)

        # --- Actions / menus / toolbar ---
        # These write the 25 `_*_act` attributes onto self — see
        # tests/gui/test_actions_factory.py for the contract.
        create_actions(self)
        create_menus(self)
        populate_toolbar(self._toolbar, self)

        self.setWindowTitle(APP_TITLE)
        self.setWindowIcon(QIcon(APP_ICON))
        self.showMaximized()

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def _open_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Open SVS File", "", "SVS Files (*.svs)")
        if not file_path:
            return

        self.state.svs_path = file_path
        self._initialise_analysis(file_path)
        self.state.result_dir = os.path.join(self.state.work_dir, "result", "")

        thumbnail_path = os.path.join(self.state.work_dir, "thumbnail", "th.png")
        from PyQt5.QtGui import QImage
        image = QImage(thumbnail_path)

        if image.isNull():
            QMessageBox.warning(self, APP_TITLE, f"Cannot load image from:\n{thumbnail_path}")
            return

        display_image(self, image)
        self.state.scale_factor = 1.0

        self._print_act.setEnabled(True)
        self._fit_to_window_act.setEnabled(True)
        self._start_analysis_act.setEnabled(True)
        self._deep_zoom_act.setEnabled(True)
        update_zoom_actions(self)

        start_tile_threads(
            self, self.state, self._thread_pool, self._progress_ui, self._on_worker_error,
        )

    def _initialise_analysis(self, file_path: str) -> None:
        """Open the SVS file, generate the thumbnail, cache tile metadata."""
        state = self.state
        # ``lev_sec`` must index ``slide.level_dimensions`` (typically ≤ ~8
        # entries), NOT the DeepZoom level index ``levi`` returned by
        # ``tile_gen`` (which can exceed ``level_count`` and caused IndexError
        # in ``get_thumb``).
        analysis = (
            StartAnalysis(lev_sec=1)
            if state.analysis_type == "slow"
            else StartAnalysis()
        )
        analysis.openSvs(file_path)
        state.work_dir = analysis.get_thumb()

        (
            state.tile_x_start,
            state.tile_x_stop,
            state.process_names,
            state.tile_start_idx,
            state.tile_stop_idx,
            state.tile_rows,
            deepzoom_level,
        ) = analysis.tile_gen(state=0)
        # Cache the two distinct level namespaces so they are never conflated.
        state.svs_level = analysis.lev_sec
        state.svs_deepzoom_level = deepzoom_level

        logger.debug(
            "Tile metadata — x_start=%s x_stop=%s names=%s "
            "start_idx=%s stop_idx=%s rows=%d openslide_level=%d deepzoom_level=%d",
            state.tile_x_start, state.tile_x_stop, state.process_names,
            state.tile_start_idx, state.tile_stop_idx, state.tile_rows,
            state.svs_level, state.svs_deepzoom_level,
        )

    # ------------------------------------------------------------------
    # View / printing / zoom — 1-line forwarders to image_display
    # ------------------------------------------------------------------

    def _view_result(self, name: str, folder: str) -> None:
        view_result(self, self.state, name, folder)

    def _print_image(self) -> None:
        print_image(self)

    def _zoom_in(self) -> None:
        zoom_in(self, self.state)

    def _zoom_out(self) -> None:
        zoom_out(self, self.state)

    def _normal_size(self) -> None:
        normal_size(self, self.state)

    def _fit_to_window(self) -> None:
        fit_to_window(self, self.state)

    def _scale_image(self, factor: float) -> None:
        scale_image(self, self.state, factor)

    def _enable_view_actions(self, enabled: bool) -> None:
        enable_view_actions(self, enabled)

    # ------------------------------------------------------------------
    # Progress dialog
    # ------------------------------------------------------------------

    def _show_progress(self, title: str) -> None:
        self._progress_dialog = QDialog(self)
        self._progress_ui = Actions()
        self._progress_ui.initUI(self._progress_dialog, title)
        self._progress_dialog.show()

    def _hide_progress(self) -> None:
        if self._progress_dialog:
            self._progress_dialog.hide()

    # ------------------------------------------------------------------
    # Tile worker / analysis worker / DeepZoom — forwarders
    # ------------------------------------------------------------------

    def _on_tile_worker_finished(self) -> None:
        on_tile_worker_finished(self, self.state)

    def _on_worker_error(self, error_tuple: tuple) -> None:
        show_worker_error(self, error_tuple)

    def _start_analysis(self) -> None:
        start_analysis(self, self.state, self._thread_pool, self._progress_ui)

    def _on_analysis_complete(self) -> None:
        on_analysis_complete(self, self.state)

    def _select_model(self) -> None:
        # Slot signature for the "Change model" menu action — invoke select_model
        # but the dialog flow is owned by analysis_worker.select_model which
        # returns the path. We don't pass the user to a "did they cancel" gate
        # here; the actual abort logic lives in start_analysis when the user
        # later clicks Run.
        select_model(self, self.state)

    def _open_deep_zoom(self) -> None:
        open_deep_zoom(self, self.state, self._thread_pool)

    # ------------------------------------------------------------------
    # About
    # ------------------------------------------------------------------

    def _about(self) -> None:
        about(self)

    def _about_deep_zoom(self) -> None:
        about_deep_zoom(self)

    # ------------------------------------------------------------------
    # Analysis mode / Monte Carlo settings
    # ------------------------------------------------------------------

    def _set_fast_mode(self) -> None:
        self.state.analysis_type = "fast"
        self._slow_act.setChecked(False)

    def _set_slow_mode(self) -> None:
        self.state.analysis_type = "slow"
        self._fast_act.setChecked(False)
        self._open_file()

    def _set_monte_carlo(self, value: int) -> None:
        self.state.monte_carlo_samples = value
        set_monte_carlo(self, value)
        logger.debug("Monte Carlo samples set to %d", value)
