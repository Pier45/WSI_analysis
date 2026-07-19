"""Main window of the Datacleaning application.

Owns the menu bar (File / About) and the Tutorial dialog, applies the
external Qt stylesheet if present, and hosts the :class:`MainTabWidget`
as its central widget.
"""

from __future__ import annotations

import os

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QAction,
    QMainWindow,
    QMenu,
    QMessageBox,
)

from .constants import APP_ICON_PATH, APP_STYLE_PATH, TUTORIAL_MESSAGE
from .main_tab_widget import MainTabWidget


class MainWindow(QMainWindow):
    """Main application window for Bayesian Datacleaning."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Bayesian Datacleaning")
        self.setWindowIcon(QIcon(APP_ICON_PATH))

        if os.path.exists(APP_STYLE_PATH):
            with open(APP_STYLE_PATH) as f:
                self.setStyleSheet(f.read())
        else:
            print(f"Warning: style file {APP_STYLE_PATH} not found. Using default style.")

        self.central_widget = MainTabWidget(self)
        self.setCentralWidget(self.central_widget)

        self._create_actions()
        self._create_menus()
        self.showMaximized()

    def _create_actions(self) -> None:
        self.tutorial_action = QAction("&Tutorial", self, triggered=self._show_tutorial)
        self.exit_action = QAction("Exit", self, triggered=self.close)

    def _create_menus(self) -> None:
        file_menu = QMenu("&File", self)
        file_menu.addAction(self.exit_action)

        about_menu = QMenu("About", self)
        about_menu.addAction(self.tutorial_action)

        self.menuBar().addMenu(file_menu)
        self.menuBar().addMenu(about_menu)

    def _show_tutorial(self) -> None:
        QMessageBox.information(self, "Bayesian Datacleaner", TUTORIAL_MESSAGE)
