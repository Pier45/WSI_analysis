"""User-tunable constants for the Bayesian Analyzer GUI.

These were originally module-level constants in ``ui_pyqt5.py``; they
have been hoisted into this module so that any helper (``actions.py``,
``menus.py``, ``deepzoom.py``, …) can import them without recreating a
circular import through ``ImageViewer``.
"""

from __future__ import annotations

import os
import sys

APP_TITLE = "Bayesian Analyzer"
# .ico is poorly supported by Qt on Linux/Wayland; prefer a PNG there so the
# window icon renders correctly instead of the Wayland fallback icon.
APP_ICON = "icons/target.png" if sys.platform.startswith("linux") else "icons/target.ico"

# Default filename used to seed the model-selection file dialog on first
# run. The user must still pick a real ``.h5`` file before any analysis
# runs (see :data:`DEFAULT_MODEL`).
DEFAULT_MODEL_FILENAME = "Model_1_85aug.h5"
# Sentinel meaning "no model selected yet" — ``start_analysis`` interprets
# this empty string as "prompt the user to pick a model before running".
# Keeping it explicit (rather than initialising state.model_name to the
# filename directly) avoids silently running with a stale path that may
# no longer exist on disk.
DEFAULT_MODEL = ""

# When the GUI runs natively on Windows, the browser connects to 127.0.0.1:5000.
# When the GUI runs inside Docker (WSL2), the host browser reaches the published
# port as localhost — set DEEPZOOM_BROWSER_HOST / DEEPZOOM_BROWSER_PORT env vars
# in the compose service to point at the published port if you remap it.
_browser_host = os.environ.get("DEEPZOOM_BROWSER_HOST", "127.0.0.1")
_browser_port = os.environ.get("DEEPZOOM_BROWSER_PORT", "5000")
DEEPZOOM_URL = f"http://{_browser_host}:{_browser_port}/"
DEEPZOOM_SERVER_SCRIPT = "src/deepzoom/deepzoom_server.py"

MONTE_CARLO_OPTIONS: tuple[int, ...] = (5, 25, 50)
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
