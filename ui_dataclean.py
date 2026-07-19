"""
Bayesian Datacleaning Application — thin launcher.

The real application lives under :mod:`gui.dataclean`; this module just
applies the Linux-specific Qt/Cursor fixes, builds :class:`QApplication`
and :class:`~gui.dataclean.MainWindow`, and runs the event loop.

Run from the repo root so the ``gui.`` / ``src.`` / ``models.`` imports
resolve:

    python ui_dataclean.py
"""

from __future__ import annotations

import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from gui.dataclean import MainWindow


def main() -> int:
    # On Linux desktops (GNOME, Unity, etc.) Qt hides the in-window menu bar
    # by default and routes it to a global/top bar that is often not visible.
    # Disable the native menu bar so the menus render inside the window on
    # every platform, matching the Windows/Docker behavior.
    # Force the xcb QPA platform under Linux/Wayland: native Wayland makes
    # the mouse cursor fall back to a generic icon unrelated to the system
    # theme, while xcb (XWayland) honors the user's cursor theme.
    if sys.platform.startswith("linux"):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
        os.environ.setdefault("QT_QPA_PLATFORMPLUGIN_PATH", "")
        os.environ["QT_LINUX_IN_WINDOW_MENUBAR"] = "1"

    app = QApplication(sys.argv)
    if sys.platform.startswith("linux"):
        app.setAttribute(Qt.AA_DontUseNativeMenuBar, True)

    window = MainWindow()
    window.menuBar().setNativeMenuBar(False)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
