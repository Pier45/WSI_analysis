"""Start the Bayesian classification in a background thread.

Contains the first-run gating that prompts the user to pick a ``.h5``
model the first time they click Run, and the worker-lifecycle wiring
for the classification pass.
"""

from __future__ import annotations

import logging
import os

from PyQt5.QtCore import QThreadPool
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QMessageBox

from src.progress_bar import Actions
from src.qt_workers import WorkerLong

from .constants import APP_TITLE, DEFAULT_MODEL_FILENAME
from .state import AnalyzerState

logger = logging.getLogger(__name__)


def start_analysis(
    parent: QMainWindow,
    state: AnalyzerState,
    pool: QThreadPool,
    progress_ui: Actions | None,
) -> None:
    """Run the Bayesian classification in a background thread (or load existing results)."""
    # First-run gating: the user must select a model before any analysis
    # can run. ``DEFAULT_MODEL`` is the empty string on purpose so a brand-new
    # install doesn't silently try to load a non-existent .h5 and crash deep
    # inside the worker thread.
    if not state.model_name:
        QMessageBox.information(
            parent, APP_TITLE,
            "No model has been selected yet. Choose a model file (.h5) before running the analysis.",
        )
        chosen = select_model(parent, state)
        if not chosen:
            # User cancelled the file dialog — abort the run silently.
            logger.info("Analysis aborted — no model selected.")
            return

    if os.path.exists(state.result_dir):
        logger.info("Analysis results already present — loading existing results.")
        parent._view_result("Pred_class", "result")
    else:
        # Import here so a missing TF wheel surfaces a clear ImportError at call
        # time instead of breaking every test that imports the Analyzer GUI.
        from src.classification import Classification  # noqa: PLC0415

        cls = Classification(state.work_dir, ty="analysis")
        parent._show_progress(title="Analysis")
        worker = WorkerLong(
            cls.classify,
            state.analysis_type,
            state.monte_carlo_samples,
            state.model_name,
        )
        worker.signals.progress.connect(lambda pct: logger.debug("Analysis: %d%%", pct))
        if progress_ui:
            worker.signals.progress.connect(progress_ui.onCountChanged)
        worker.signals.finished.connect(lambda: on_analysis_complete(parent, state))
        worker.signals.error.connect(parent._on_worker_error)
        pool.start(worker)

    parent._enable_view_actions(True)


def on_analysis_complete(parent: QMainWindow, state: AnalyzerState) -> None:
    """Called when the analysis thread finishes successfully."""
    parent._hide_progress()
    logger.info("Analysis thread complete.")
    parent._view_result("Pred_class", "result")


def select_model(parent: QMainWindow, state: AnalyzerState) -> str | None:
    """Prompt the user to pick a ``.h5`` model file and cache the path.

    Returns the chosen path (also stored in ``state.model_name``), or
    ``None`` if the user cancelled. Returning the value lets callers
    like :func:`start_analysis` decide whether to abort the run instead
    of failing later inside the worker thread.
    """
    QMessageBox.information(
        parent, APP_TITLE,
        "The selected model must produce three output classes: AC, AD, H.",
    )
    # Seed the dialog with the default filename (and the directory of any
    # previously selected model) so the user lands on a sensible path
    # rather than the FS root on every first run.
    start_dir = os.path.dirname(state.model_name) if state.model_name else ""
    suggested = os.path.join(start_dir, DEFAULT_MODEL_FILENAME) if start_dir else DEFAULT_MODEL_FILENAME
    model_path, _ = QFileDialog.getOpenFileName(parent, "Select Model", suggested, "H5 Files (*.h5)")
    if model_path:
        state.model_name = model_path
        logger.info("Model changed to: %s", model_path)
        return model_path
    logger.info("Model selection cancelled by user.")
    return None
