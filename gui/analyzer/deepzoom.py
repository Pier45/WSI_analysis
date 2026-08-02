"""DeepZoom viewer launcher + info / about dialogs for the Bayesian Analyzer.

The viewer launches a Flask subprocess (``python -m
src.deepzoom.deepzoom_server <slide>``) and opens the browser at the
configured URL once the server is ready. Spaces in the SVS path are
unsupported by Flask on Windows — the user is warned explicitly.

The original ``_open_deep_zoom`` mistakenly kept a dead ``os.system("cmd
/k ...")`` block next to the working ``subprocess.Popen`` call; that dead
block is removed here — the subprocess path is the only one that ships.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
import webbrowser

from PyQt5.QtCore import QThreadPool
from PyQt5.QtWidgets import QMainWindow, QMessageBox

from src.qt_workers import Worker

from .constants import APP_TITLE, DEEPZOOM_URL
from .state import AnalyzerState

logger = logging.getLogger(__name__)


def open_deep_zoom(
    parent: QMainWindow,
    state: AnalyzerState,
    pool: QThreadPool,
) -> None:
    """Start the DeepZoom Flask server in a separate process and open the
    browser once the server is ready.

    Spaces in the SVS path are unsupported and will cause a Flask error
    on Windows; the user is warned explicitly.
    """
    if " " in state.svs_path:
        QMessageBox.critical(
            parent, APP_TITLE,
            "The file path contains spaces:\n\n"
            f"{state.svs_path}\n\n"
            "The deepzoom viewer requires a path without spaces.\n"
            "Please move or rename the file and try again.",
        )
        return

    # Portable launch: `python -m src.deepzoom.deepzoom_server <slide>` works
    # on Windows, WSL2, and inside the Docker container. `subprocess.Popen`
    # inherits os.environ so DEEPZOOM_HOST/PORT are honored automatically
    # (the server defaults to 127.0.0.1:5000, correct for the GUI's
    # in-process use).
    cmd = [sys.executable, "-m", "src.deepzoom.deepzoom_server", state.svs_path]
    try:
        parent._deepzoom_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        QMessageBox.critical(
            parent, APP_TITLE,
            "Failed to start the DeepZoom server.\n\n"
            f"{exc}",
        )
        return

    QMessageBox.information(
        parent, APP_TITLE,
        "Your browser will open with the deepzoom viewer.\n\nPress OK to continue.",
    )
    pool.start(Worker(open_browser))


def open_browser() -> None:
    """Open the DeepZoom URL in the default browser after a short delay."""
    time.sleep(0.5)
    webbrowser.open_new_tab(DEEPZOOM_URL)


def about_deep_zoom(parent: QMainWindow) -> None:
    QMessageBox.about(
        parent,
        "Deepzoom Viewer",
        "<p>The <b>Deepzoom</b> viewer lets you zoom into SVS files at "
        "maximum resolution inside your browser.</p>"
        f"<p>The server opens at <tt>{DEEPZOOM_URL}</tt>.</p>"
        "<p>The right panel shows metadata about the SVS file; additional "
        "images may appear on the left depending on the selected file.</p>",
    )
