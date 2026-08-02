"""
Bayesian Analyzer — thin launcher.

The real application lives under :mod:`gui.analyzer`; this module just
applies the Linux-specific Qt/Cursor fixes, builds :class:`QApplication`
and :class:`~gui.analyzer.ImageViewer`, and runs the event loop.

Run from the repo root so the ``gui.`` / ``src.`` / ``models.`` imports
resolve:

    python ui_pyqt5.py
"""

from __future__ import annotations

import logging
import os
import sys

from PyQt5.QtWidgets import QApplication


def main() -> int:
    # On Linux desktops (GNOME, Unity, etc.) Qt hides the in-window menu bar
    # by default and routes it to a global/top bar that is often not visible.
    # Force the xcb QPA platform under Linux/Wayland: native Wayland makes the
    # mouse cursor fall back to a generic icon unrelated to the system theme,
    # while xcb (XWayland) honors the user's cursor theme.
    if sys.platform.startswith("linux"):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
        os.environ.setdefault("QT_QPA_PLATFORMPLUGIN_PATH", "")
        os.environ["QT_LINUX_IN_WINDOW_MENUBAR"] = "1"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)

    app = QApplication(sys.argv)
    from gui.analyzer import ImageViewer

    viewer = ImageViewer()
    viewer.menuBar().setNativeMenuBar(False)
    viewer.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
