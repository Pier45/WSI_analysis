"""About dialog for the Bayesian Analyzer."""

from __future__ import annotations

from PyQt5.QtWidgets import QMainWindow, QMessageBox

from .constants import APP_TITLE


def about(parent: QMainWindow) -> None:
    QMessageBox.about(
        parent,
        f"About {APP_TITLE}",
        "<p>The <b>Bayesian Analyzer</b> analyzes SVS pathology slides "
        "(typically ~1 GB) using a Bayesian neural network.</p>"
        "<p>It also supports deep-zoom visualization by opening a "
        "browser tab at maximum resolution.</p>",
    )
