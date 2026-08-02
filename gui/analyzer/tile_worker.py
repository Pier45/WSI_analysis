"""Tile-creation background workers for the Bayesian Analyzer.

One ``WorkerLong`` per process partition is launched to write PNG tiles
into ``<work_dir>/<process_name>``. The progress dialog is hidden from
the main thread (via the worker's ``finished`` signal) once the last
worker reports done.
"""

from __future__ import annotations

import logging
import os
import time

from PyQt5.QtCore import QThreadPool
from PyQt5.QtWidgets import QMessageBox

from src.multi_processing_analysis import StartAnalysis
from src.progress_bar import Actions
from src.qt_workers import WorkerLong

from .constants import APP_TITLE
from .state import AnalyzerState

logger = logging.getLogger(__name__)


def folder_exists(state: AnalyzerState, name: str) -> bool:
    """Return ``True`` if *name* already exists inside the working directory."""
    return name in os.listdir(state.work_dir)


def create_tiles(
    state: AnalyzerState,
    analysis_type: str,
    svs_path: str,
    svs_level: int,
    work_dir: str,
    tile_args: list,
    progress_callback,  # injected by WorkerLong
) -> str:
    """Worker function: create PNG tiles for one process partition.

    ``tile_args``: ``[x_start, x_stop, process_name, tile_start, tile_stop,
    n_rows, level]`` where ``level`` is the **DeepZoom** level index
    (returned as ``levi`` by ``tile_gen(state=0)``), NOT the OpenSlide
    level.
    """
    x_start, x_stop, process_name, tile_start, tile_stop, n_rows, level = tile_args
    logger.debug(
        "Creating tiles — process=%s x=[%d,%d) start=%d stop=%d level=%d",
        process_name, x_start, x_stop, tile_start, tile_stop, level,
    )

    if folder_exists(state, process_name):
        # Existing folder: emit 100 % and exit without touching the GUI
        # (no _hide_progress() here — that runs on the main thread in
        # on_tile_worker_finished once every worker has reported done).
        time.sleep(0.1)
        progress_callback.emit(100)
        return f"Tile folder '{process_name}' already exists — skipping."

    analysis = (
        StartAnalysis(lev_sec=svs_level)
        if analysis_type == "slow"
        else StartAnalysis()
    )
    analysis.openSvs(svs_path)
    tile_source = analysis.tile_gen(state=1)

    # Guard against stale/corrupted DeepZoom level indices to avoid the
    # openslide "Invalid address" ValueError. ``level_tiles`` is a tuple
    # of (nx, ny) per level — clamp ``level`` to the valid range.
    max_level = len(tile_source.level_tiles) - 1
    if not (0 <= level <= max_level):
        logger.warning(
            "DeepZoom level=%d out of range [0, %d]; clamping to %d.",
            level, max_level, max_level,
        )
        level = max_level
    tiles_x, tiles_y = tile_source.level_tiles[level]
    # Clamp per-axis ranges to the generator's actual tile grid for this
    # level — out-of-range (x, y) is what raises "Invalid address".
    x_hi = min(x_stop, tiles_x)
    y_hi = min(n_rows, tiles_y)

    folder_path = os.path.join(work_dir, process_name)
    os.mkdir(folder_path)

    is_primary = tile_start == 1
    current_index = tile_start

    for x in range(x_start, x_hi):
        for y in range(y_hi):
            tile = tile_source.get_tile(level, (x, y))
            # Use the per-tile running counter (current_index), not the
            # constant per-partition tile_start, so each PNG gets a unique
            # filename. Reusing tile_start made every file in a partition
            # share the same index, which then collided in
            # Classification.select_folder()'s dict key and left only one
            # tile per partition in the analysis dictionary.
            tile_path = os.path.join(folder_path, f"tile_{current_index}_{x}_{y}.png")
            tile.save(tile_path, "PNG")
            current_index += 1

            if is_primary:
                pct = int(100 * (current_index - 1) / tile_stop)
                progress_callback.emit(pct)

    return "Tile creation complete."


def start_tile_threads(
    parent,
    state: AnalyzerState,
    pool: QThreadPool,
    progress_ui: Actions | None,
    on_worker_error,
) -> None:
    """Launch one :class:`WorkerLong` per process partition to create tiles.

    If every expected partition folder already exists on disk (e.g. the
    user re-opens the same SVS file), tile generation is skipped entirely
    — no workers are queued and no progress dialog is shown.
    """
    existing = set(os.listdir(state.work_dir)) if os.path.isdir(state.work_dir) else set()
    missing = [name for name in state.process_names if name not in existing]
    if not missing:
        logger.info(
            "All %d tile partitions already present — skipping tile creation.",
            len(state.process_names),
        )
        return

    parent._show_progress(title="Tile creation")
    # Counter drained by on_tile_worker_finished; the dialog is hidden
    # on the main thread when the last worker reports done.
    state.pending_tile_workers = len(missing)

    for idx in range(len(state.process_names)):
        tile_args = [
            state.tile_x_start[idx],
            state.tile_x_stop[idx],
            state.process_names[idx],
            state.tile_start_idx[idx],
            state.tile_stop_idx[idx],
            state.tile_rows,
            state.svs_deepzoom_level,
        ]
        worker = WorkerLong(
            create_tiles,
            state,
            state.analysis_type,
            state.svs_path,
            state.svs_level,
            state.work_dir,
            tile_args,
        )
        worker.signals.result.connect(lambda msg: logger.info("Tile worker result: %s", msg))
        worker.signals.progress.connect(lambda pct: logger.debug("Tile progress: %d%%", pct))
        if progress_ui:
            worker.signals.progress.connect(progress_ui.onCountChanged)
        worker.signals.finished.connect(lambda: on_tile_worker_finished(parent, state))
        worker.signals.error.connect(on_worker_error)
        pool.start(worker)


def on_tile_worker_finished(parent, state: AnalyzerState) -> None:
    """Slot connected to each tile worker's ``finished`` signal.

    Runs on the main thread (Qt marshals cross-thread signals through the
    event loop), so it is safe to touch the progress dialog here. Hides
    the dialog once the last pending worker has reported done.
    """
    if not state.pending_tile_workers:
        return
    state.pending_tile_workers -= 1
    logger.debug("Tile worker finished — %d remaining.", state.pending_tile_workers)
    if state.pending_tile_workers <= 0:
        parent._hide_progress()
        state.pending_tile_workers = 0


def show_worker_error(parent, error_tuple: tuple) -> None:
    """Display a critical dialog when a background worker raises an exception."""
    exc_type, value, tb_str = error_tuple
    logger.error("Worker error: %s\n%s", value, tb_str)
    QMessageBox.critical(
        parent,
        APP_TITLE,
        f"{exc_type.__name__}: {value}\n\nSee the console for the full traceback.",
    )
