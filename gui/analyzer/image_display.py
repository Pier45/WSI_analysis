"""Image display, printing, and zoom helpers for the Bayesian Analyzer.

These were instance methods on ``ImageViewer``; they are extracted here
so each one is small, easy to read, and free of the window's plumbing.
Each helper takes either the ``ImageViewer`` itself (when it needs to
update Qt widgets like ``_image_label`` / ``_scroll_area``) plus the
shared :class:`~gui.analyzer.state.AnalyzerState`, so it doesn't read
``self.*`` for shared state.
"""

from __future__ import annotations

import logging
import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter, QPixmap
from PyQt5.QtPrintSupport import QPrintDialog
from PyQt5.QtWidgets import QMainWindow, QMessageBox

from .constants import APP_TITLE, ZOOM_IN_FACTOR, ZOOM_MAX, ZOOM_MIN, ZOOM_OUT_FACTOR
from .state import AnalyzerState

logger = logging.getLogger(__name__)


def display_image(parent: QMainWindow, image: QImage) -> None:
    """Render *image* in the central label, scaling to fit the screen if needed."""
    parent._image_label.setPixmap(QPixmap.fromImage(image))

    w, h = image.width(), image.height()
    longer_axis = 0 if w >= h else 1
    img_long = w if longer_axis == 0 else h
    screen_long = parent._screen_size[longer_axis]

    if not parent._fit_to_window_act.isChecked():
        parent._image_label.adjustSize()

    if screen_long < img_long:
        ratio = screen_long / img_long - 0.04
        parent._image_label.resize(ratio * parent._image_label.pixmap().size())


def view_result(parent: QMainWindow, state: AnalyzerState, name: str, folder: str) -> None:
    """Load and display a result image.

    ``folder`` is one of ``"result"``, ``"th"`` (thumbnail), or
    ``"uncertainty"``.
    """
    path_map = {
        "result": os.path.join(state.result_dir, f"{name}.png"),
        "th": os.path.join(state.work_dir, "thumbnail", "th.png"),
        "uncertainty": os.path.join(state.result_dir, "uncertainty", f"{name}.png"),
    }
    image_path = path_map.get(folder, "")
    logger.debug("Viewing image: %s", image_path)

    image = QImage(image_path)
    if image.isNull():
        QMessageBox.warning(parent, APP_TITLE, f"Could not load:\n{image_path}")
        return
    display_image(parent, image)


def print_image(parent: QMainWindow) -> None:
    """Open the print dialog and print the currently displayed pixmap."""
    dialog = QPrintDialog(parent._printer, parent)
    if dialog.exec_():
        painter = QPainter(parent._printer)
        rect = painter.viewport()
        size = parent._image_label.pixmap().size()
        size.scale(rect.size(), Qt.KeepAspectRatio)
        painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
        painter.setWindow(parent._image_label.pixmap().rect())
        painter.drawPixmap(0, 0, parent._image_label.pixmap())


# ---------------------------------------------------------------------------
# Zoom
# ---------------------------------------------------------------------------


def zoom_in(parent: QMainWindow, state: AnalyzerState) -> None:
    scale_image(parent, state, ZOOM_IN_FACTOR)


def zoom_out(parent: QMainWindow, state: AnalyzerState) -> None:
    scale_image(parent, state, ZOOM_OUT_FACTOR)


def normal_size(parent: QMainWindow, state: AnalyzerState) -> None:
    parent._image_label.adjustSize()
    state.scale_factor = 1.0


def fit_to_window(parent: QMainWindow, state: AnalyzerState) -> None:
    fit = parent._fit_to_window_act.isChecked()
    parent._scroll_area.setWidgetResizable(fit)
    if not fit:
        normal_size(parent, state)
    update_zoom_actions(parent)


def scale_image(parent: QMainWindow, state: AnalyzerState, factor: float) -> None:
    state.scale_factor *= factor
    parent._image_label.resize(state.scale_factor * parent._image_label.pixmap().size())
    _adjust_scroll_bar(parent._scroll_area.horizontalScrollBar(), factor)
    _adjust_scroll_bar(parent._scroll_area.verticalScrollBar(), factor)
    parent._zoom_in_act.setEnabled(state.scale_factor < ZOOM_MAX)
    parent._zoom_out_act.setEnabled(state.scale_factor > ZOOM_MIN)


def _adjust_scroll_bar(scroll_bar, factor: float) -> None:
    scroll_bar.setValue(
        int(factor * scroll_bar.value() + (factor - 1) * scroll_bar.pageStep() / 2)
    )


def update_zoom_actions(parent: QMainWindow) -> None:
    enabled = not parent._fit_to_window_act.isChecked()
    parent._zoom_in_act.setEnabled(enabled)
    parent._zoom_out_act.setEnabled(enabled)
    parent._normal_size_act.setEnabled(enabled)


def enable_view_actions(parent: QMainWindow, enabled: bool) -> None:
    """Toggle the eight View-menu overlay actions on/off."""
    for act in (
        parent._v_no_overlay_act,
        parent._v_all_classes_act,
        parent._v_ac_act,
        parent._v_ad_act,
        parent._v_h_act,
        parent._v_total_uncertainty_act,
        parent._v_aleatoric_act,
        parent._v_epistemic_act,
    ):
        act.setEnabled(enabled)
