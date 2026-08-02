"""Mutable shared state for the Bayesian Analyzer helpers.

Mirrors :class:`gui.dataclean.state.DataCleanState`. The original
``ImageViewer`` kept every piece of cross-slot state as a plain
attribute on ``self``. Once the slots were extracted into free
functions in :mod:`gui.analyzer`, those helpers needed somewhere
common to read/write the same paths, model name, tiling metadata and
zoom factor.

``ImageViewer`` owns a single instance (``self.state``) and passes it
to every helper call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .constants import DEFAULT_MODEL, DEFAULT_MONTE_CARLO


@dataclass
class AnalyzerState:
    """Mutable shared state across the Analyzer's helper modules."""

    # --- Paths ---
    svs_path: str = ""
    work_dir: str = ""
    result_dir: str = ""

    # --- Analysis parameters ---
    analysis_type: str = "fast"          # "fast" | "slow"
    model_name: str = DEFAULT_MODEL      # empty = "prompt user to pick .h5"
    monte_carlo_samples: int = DEFAULT_MONTE_CARLO

    # --- Tile-generation metadata (populated by StartAnalysis.tile_gen) ---
    tile_x_start: list[int] = field(default_factory=list)
    tile_x_stop: list[int] = field(default_factory=list)
    process_names: list[str] = field(default_factory=list)
    tile_start_idx: list[int] = field(default_factory=list)
    tile_stop_idx: list[int] = field(default_factory=list)
    tile_rows: int = 0
    # OpenSlide-level index (indexes ``slide.level_dimensions``, used as
    # ``StartAnalysis(lev_sec=...)``). Must stay < ``slide.level_count``.
    svs_level: int = 1
    # DeepZoom level index (``levi`` returned by ``tile_gen(state=0)``),
    # a separate namespace that grows much larger than ``level_count``.
    # Passed to ``DeepZoomGenerator.get_tile(level, ...)`` in workers.
    svs_deepzoom_level: int = 0

    # Counts outstanding tile-creation workers; drained by
    # ``on_tile_worker_finished`` so the progress dialog is hidden on the
    # main thread (never from a worker thread) once the last one is done.
    pending_tile_workers: int = 0

    # --- UI ---
    scale_factor: float = 0.0
