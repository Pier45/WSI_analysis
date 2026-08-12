"""Tile-creation background workers for the Bayesian Analyzer.

Historical design: one ``WorkerLong`` per process partition, each running
on the QThreadPool. Every partition worker reopened the SVS
(``openslide.OpenSlide`` + ``DeepZoomGenerator`` build) and ran its tile
loop serially inside a thread — but ``openslide-python`` is a ``ctypes``
binding, so ``openslide_read_region`` does NOT release the GIL during the
decode. Threads could only overlap I/O waits, not the CPU-bound decode,
which is the dominant cost on real SVS files.

Current design: a single ``WorkerLong`` owns a
``concurrent.futures.ProcessPoolExecutor`` and submits every missing
partition to it. Each process opens the SVS read-only once via a
process-local LRU cache (``_get_cached_generator`` in
:mod:`src.multi_processing_analysis`) so the parse + DeepZoom index build
happens once per worker *process*, not once per partition (Option B).
Per-tile progress is marshalled back through a
``multiprocessing.Queue`` and re-emitted on the Qt progress signal — keeps
the UI dialog working without coupling the worker pool to Qt.
"""
from __future__ import annotations

import datetime
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor

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
    """Worker function: create PNG tiles for *every* missing process partition
    using a :class:`~concurrent.futures.ProcessPoolExecutor`.

    ``tile_args`` is the legacy 7-element per-partition tuple, kept for
    backward compatibility with ``start_tile_threads``'s call shape:
    ``[x_start, x_stop, process_name, tile_start, tile_stop, n_rows, level]``
    — but since the refactor collapses N per-partition workers into one
    host, the *unused* elements are tolerated; only the partition
    metadata that the host needs is read out via ``state``.

    Progress is streamed back through a ``(partition_index, current_index,
    tile_count)`` triple per tile written, polled out of a
    ``multiprocessing.Queue`` and emitted on ``progress_callback`` so the
    Qt progress dialog keeps updating.
    """
    _tile_start_wall = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _tile_start_perf = time.perf_counter()
    logger.info(
        "Tile creation started at %s (svs=%s, work_dir=%s).",
        _tile_start_wall, svs_path, work_dir,
    )
    # Open the SVS once in the host process purely to read the DeepZoom
    # metadata (level_tiles / level_dimensions) needed to clamp the
    # per-partition x/y ranges. The actual tile extraction happens in the
    # child processes — this open is cheap metadata-only work, not a
    # decode, and gives us the bounds to validate each partition against
    # before paying the cost of dispatching it.
    analysis = (
        StartAnalysis(lev_sec=svs_level)
        if analysis_type == "slow"
        else StartAnalysis()
    )
    analysis.openSvs(svs_path)
    tile_source = analysis.tile_gen(state=1)

    deepzoom_level = state.svs_deepzoom_level
    # Clamp the DeepZoom level to the generator's valid range — the same
    # guard the per-partition worker used to apply on its own, retained
    # here so an out-of-range level never reaches a child process.
    max_level = len(tile_source.level_tiles) - 1
    if not (0 <= deepzoom_level <= max_level):
        logger.warning(
            "DeepZoom level=%d out of range [0, %d]; clamping to %d.",
            deepzoom_level, max_level, max_level,
        )
        deepzoom_level = max_level
    tiles_x, tiles_y = tile_source.level_tiles[deepzoom_level]

    partitions = []
    # Read the tile_size/overlap/limit_bounds defaults off the ``analysis``
    # instance we already opened for metadata — constructing extra
    # ``StartAnalysis`` objects here would re-default every iteration and
    # obscure the fact that these are just the class constants forwarded
    # to the child-process DeepZoomGenerator build.
    cfg_tile_size = analysis.tile_size
    cfg_overlap = analysis.overlap
    cfg_limit_bounds = analysis.limit_bounds
    # Global tile-counter ceiling — the maximum per-partition ``tile_stop``
    # across the partitions we will actually extract. Each child reports
    # progress against this shared total so the UI bar climbs monotonically
    # 0→100% as the parallel workers finish their rows, instead of
    # resetting to 0% at each partition boundary (which happened when each
    # child used its own per-partition stop as the denominator).
    _existing = (
        set(os.listdir(state.work_dir)) if os.path.isdir(state.work_dir) else set()
    )
    missing_indices = [
        i for i in range(len(state.process_names))
        if state.process_names[i] not in _existing
    ]
    global_total = max(
        (state.tile_stop_idx[i] for i in missing_indices), default=1
    )
    for idx in missing_indices:
        name = state.process_names[idx]
        x_start = state.tile_x_start[idx]
        x_stop = min(state.tile_x_stop[idx], tiles_x)
        partitions.append(
            (
                x_start,
                x_stop,
                name,
                state.tile_start_idx[idx],
                svs_path,
                svs_level,
                cfg_tile_size,
                cfg_overlap,
                cfg_limit_bounds,
                work_dir,
                min(state.tile_rows, tiles_y),
                deepzoom_level,
                idx,
                global_total,
            )
        )

    if not partitions:
        progress_callback.emit(100)
        logger.info(
            "No tile partitions to create — all already present (checked in %.4fs).",
            time.perf_counter() - _tile_start_perf,
        )
        return "All tile partitions already present."

    # ``multiprocessing.Queue()`` cannot be passed as an argument to
    # ``ProcessPoolExecutor.submit`` — its ``__getstate__`` raises
    # ``RuntimeError: Queue objects should only be shared between processes
    # through inheritance`` because the executor pickles the args under
    # the spawn protocol. A ``multiprocessing.Manager().Queue()`` is a
    # *picklable* proxy: the underlying state lives in a manager process
    # started here in the host, and the proxy serialises to a fd handle
    # that's valid across both ``fork`` (Linux) and ``spawn``
    # (Windows / macOS) start methods. The manager is owned by this
    # ``with`` block and torn down when the partitions finish.
    mgr = multiprocessing.Manager()
    progress_queue = mgr.Queue()

    # Per-partition count of tiles written so far — kept on the host so the
    # progress percentage is the SUM across all parallel workers, not the
    # last-emitted single worker's value. Each child pushes its own
    # ``tiles_written = current_index - tile_start`` (always relative to that
    # partition's own start, so it's safe to sum across workers), plus the
    # shared ``global_total`` so the host doesn't need it in scope.
    partition_progress: dict[int, int] = {}

    def _drain_progress():
        """Pull every queued progress triple and emit the global completion
        percentage on the Qt progress signal. Non-blocking; returns when
        the queue is empty. Called from the host thread between executor
        submissions and again during the final join.

        Uses the enclosing-scope ``global_total`` as the denominator rather
        than the per-event ``total`` field, so the percentage is correct
        even on the first call (when the queue is empty and no ``total``
        variable would otherwise be in scope — the UnboundLocalError that
        crashed the first real SVS run).
        """
        while not progress_queue.empty():
            try:
                pi, tiles_written, _total = progress_queue.get_nowait()
            except Exception:
                break
            partition_progress[pi] = tiles_written
        if not partition_progress or global_total <= 0:
            return
        completed = sum(partition_progress.values())
        pct = min(int(100 * completed / global_total), 100)
        progress_callback.emit(pct)

    try:
        n_workers = min(len(partitions), multiprocessing.cpu_count())
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [
                ex.submit(_analysis_tile_worker, p, progress_queue) for p in partitions
            ]
            # Poll the progress queue while the children run. Each call drains
            # everything currently queued; small sleep so we don't busy-spin.
            while not all(f.done() for f in futures):
                _drain_progress()
                time.sleep(0.05)
            # Final drain after the last future completes — guarantees the
            # 100 % progress event reaches the UI.
            _drain_progress()

        # Surface any child-process exception via the worker's caller (the
        # ``finished`` signal). Throwing here lets ``worker.signals.error``
        # fire instead of silently swallowing a decode failure in a child.
        for f in futures:
            exc = f.exception()
            if exc is not None:
                raise RuntimeError(f"Tile worker partition failed: {exc!r}") from exc
    finally:
        # Tear down the manager process explicitly so a long-lived Qt host
        # (the GUI may stay open for hours after tiling) doesn't leak a
        # stray manager process per analysis run.
        mgr.shutdown()

    progress_callback.emit(100)
    _elapsed = time.perf_counter() - _tile_start_perf
    logger.info(
        "Tile creation finished at %s — elapsed %.2fs.",
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        _elapsed,
    )
    return "Tile creation complete."


def start_tile_threads(
    parent,
    state: AnalyzerState,
    pool: QThreadPool,
    progress_ui: Actions | None,
    on_worker_error,
) -> None:
    """Launch one :class:`WorkerLong` that drives a :class:`ProcessPoolExecutor`
    over every missing partition.

    The previous design started one ``WorkerLong`` per partition; that
    made each partition re-open the SVS and run its tile loop serially
    inside a thread (no parallelism on the GIL-bound decode). The new
    design starts a *single* ``WorkerLong`` that owns the process pool —
    partitions run in parallel processes, each opening the SVS once via a
    per-process LRU cache.
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
    # ``_show_progress`` just (re)created ``parent._progress_ui`` — bind
    # signals to that fresh instance, NOT the ``progress_ui`` parameter
    # which is stale (the previous run's dialog, or ``None`` on first
    # call). Binding the stale ref is why the bar opened/closed empty:
    # emissions went to a dead/None slot.
    live_progress_ui = parent._progress_ui
    # Single host worker now (was: one per partition). The progress
    # counter still drains via ``on_tile_worker_finished`` once the host
    # reports done.
    state.pending_tile_workers = 1

    worker = WorkerLong(
        create_tiles,
        state,
        state.analysis_type,
        state.svs_path,
        state.svs_level,
        state.work_dir,
        # ``tile_args`` kept for backward-compat with the per-partition
        # signature — the host no longer reads it; partition metadata comes
        # from ``state`` directly.
        [
            state.tile_x_start[0] if state.tile_x_start else 0,
            state.tile_x_stop[0] if state.tile_x_stop else 0,
            state.process_names[0] if state.process_names else "",
            state.tile_start_idx[0] if state.tile_start_idx else 1,
            state.tile_stop_idx[0] if state.tile_stop_idx else 1,
            state.tile_rows,
            state.svs_deepzoom_level,
        ],
    )
    worker.signals.result.connect(lambda msg: logger.info("Tile worker result: %s", msg))
    worker.signals.progress.connect(lambda pct: logger.debug("Tile progress: %d%%", pct))
    if live_progress_ui:
        worker.signals.progress.connect(live_progress_ui.onCountChanged)
    worker.signals.finished.connect(lambda: on_tile_worker_finished(parent, state))
    worker.signals.error.connect(on_worker_error)
    pool.start(worker)


def on_tile_worker_finished(parent, state: AnalyzerState) -> None:
    """Slot connected to the host tile worker's ``finished`` signal.

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


# ---------------------------------------------------------------------------
# Module-level worker for the ProcessPoolExecutor
# ---------------------------------------------------------------------------


def _analysis_tile_worker(partition, progress_queue):
    """Module-level (picklable) tile-extraction worker for the analysis path.

    Mirrors the training-path ``_tile_partition_worker`` but pushes a
    per-row ``(partition_index, tiles_written, global_total)`` triple into
    ``progress_queue`` so the host can sum across all parallel workers and
    emit a single monotonic Qt progress percentage without coupling Qt
    objects to a child process.

    ``tiles_written`` is the count of tiles this *partition* has written so
    far (relative to its own ``tile_start``), so the host can safely sum it
    across workers without double-counting. ``global_total`` is the shared
    ceiling (max per-partition ``tile_stop`` over the whole run) — every
    worker emits the same value so the host doesn't need it in scope.

    Returns the total tiles written; raises on failure so the host's
    ``future.exception()`` triggers the worker error signal.
    """
    (
        x_start, x_stop, process_name, tile_start,
        svs_path, lev_sec, tile_size, overlap, limit_bounds,
        work_dir, n_rows, deepzoom_level,
        partition_index, global_total,
    ) = partition

    create_fold = os.path.join(work_dir, process_name)
    if os.path.isdir(create_fold):
        return 0
    os.mkdir(create_fold)

    # Per-process cached open (Option B): reuse the same DeepZoomGenerator
    # if this worker process happens to handle multiple partitions (which
    # happens when ``n_workers < len(partitions)``).
    from src.multi_processing_analysis import _get_cached_generator

    _slide, generator = _get_cached_generator(
        svs_path, lev_sec, tile_size, overlap, limit_bounds
    )

    max_level = len(generator.level_tiles) - 1
    if not (0 <= deepzoom_level <= max_level):
        deepzoom_level = max_level
    tiles_x, tiles_y = generator.level_tiles[deepzoom_level]
    x_hi = min(x_stop, tiles_x)
    y_hi = min(n_rows, tiles_y)

    tiles_written = 0
    # Push progress once per *row* (not per tile): row-based batching
    # cuts the Manager().Queue IPC calls by ~TILE_X (≈32× here),
    # which keeps the manager-proxy overhead from eroding the
    # ProcessPool speedup. UI progress still updates smoothly because
    # a row is ~1-2 ms of decode on a real SVS — well below the human
    # perception threshold for the progress bar.
    for x in range(x_start, x_hi):
        for y in range(y_hi):
            tile = generator.get_tile(deepzoom_level, (x, y))
            current_index = tile_start + tiles_written
            tile_path = os.path.join(create_fold, f"tile_{current_index}_{x}_{y}.png")
            tile.save(tile_path, "PNG")
            tiles_written += 1
        try:
            progress_queue.put_nowait((partition_index, tiles_written, global_total))
        except Exception:
            # Progress reporting is best-effort; a backpressured or
            # broken queue must never abort the tile extraction itself.
            pass

    return tiles_written
