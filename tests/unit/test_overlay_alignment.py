"""Regression tests for ``Classification.overlay`` tile geometry.

Two bugs lived in ``overlay()`` until Aug 2026:

1. **X/Y axis swap** — the row slice ``image_base[r0:r0+shape_x, ...]`` used
   ``shape_x`` (derived from ``a.shape[1]`` = width) instead of the row-axis
   (height) size, and symmetrically the col slice used ``shape_y`` (height).
   Cells became non-square and shifted by a few pixels per step — the
   characteristic "tiles not perfectly aligned, spaced/overlapping by a few
   pixels" artifact in the classified overlays.

2. **Rounding drift** — ``c0 = int(round(column * step_x))`` with a float
   ``step_x = W / n_cols`` accumulated ±1 px drift across the grid, leaving
   thin gaps/overlaps between adjacent cells even for the square case.

The fix uses the integer partition ``floor(k*size/N) .. floor((k+1)*size/N)``
and assigns the row-axis (height) to the row slice and the col-axis (width)
to the col slice. These tests pin the corrected geometry by directly
checking the coverage mask produced by ``overlay()`` for every cell.

The tests are marked ``tf`` because ``src.classification`` imports TensorFlow
at module top — they belong with the `not-openslide` tier but require TF,
so they are skipped when TF is unavailable.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

# ``src.classification`` imports ``matplotlib.pyplot`` at module top, and the
# installed matplotlib/pyparsing pair emits a third-party
# ``PyparsingDeprecationWarning`` during pyplot's first import. The repo
# turns warnings into errors (``filterwarnings = ["error", ...]`` in
# pyproject.toml) — that fires *during collection*, before pytest can apply
# the per-module ``filterwarnings`` mark, so we suppress it explicitly here,
# at import time, before pulling classification in. It's third-party noise
# from matplotlib internals, not from our code.
warnings.filterwarnings(
    "ignore",
    category=PendingDeprecationWarning,
    module="pyparsing.*",
)
try:
    from pyparsing.warnings import PyparsingDeprecationWarning as _PypDep
except Exception:  # pragma: no cover — older pyparsing paths
    _PypDep = None
if _PypDep is not None:
    warnings.filterwarnings("ignore", category=_PypDep)

pytestmark = pytest.mark.tf
pytest.importorskip("tensorflow")  # src.classification pulls tf at import time

from src.classification import Classification  # noqa: E402


def _make_classification_with_thumbnail(monkeypatch, thumb_h, thumb_w):
    """Build a ``Classification`` whose ``overlay()`` runs against a synthetic
    thumbnail of shape ``(thumb_h, thumb_w, 3)`` without touching disk, TF,
    or openslide.

    - ``plt.imread`` is stubbed to return the synthetic thumbnail.
    - ``os.makedirs`` / ``os.path.exists`` are stubbed so the fake ``/fake``
      path never touches the real filesystem.
    - ``Image.open`` inside ``new_save`` is stubbed to a black RGB array, and
      ``new_save`` is replaced with a recorder that stores the per-call RGBA
      buffer so the test can inspect the painted cells.
    """
    thumb = np.ones((thumb_h, thumb_w, 3), dtype=np.uint8) * 128

    cls = Classification(path="/fake", ty="analysis")
    cls.dictionary = {}

    import matplotlib.pyplot as plt  # local import to patch the same object

    monkeypatch.setattr(plt, "imread", lambda _p: thumb.copy())
    # overlay() does ``os.makedirs(.../uncertainty)`` when result dir is
    # missing; short-circuit both calls so the fake path never hits disk.
    monkeypatch.setattr("os.path.exists", lambda _p: True, raising=False)
    monkeypatch.setattr("os.makedirs", lambda _p, **_kw: None, raising=False)

    saved: dict[str, np.ndarray] = {}

    def fake_new_save(self, image_base, res_name):
        saved[res_name] = image_base.copy()

    monkeypatch.setattr(Classification, "new_save", fake_new_save)
    return cls, saved


def _grid_dictionary(n_cols, n_rows, pred_class="AC"):
    """Build a ``Classification.dictionary``-shaped dict with a contiguous
    ``n_cols × n_rows`` grid. Every tile is classified ``pred_class`` with
    low uncertainty so the ``Pred_class`` branch paints every cell."""
    d = {}
    i = 0
    for row in range(n_rows):
        for col in range(n_cols):
            d[i] = {
                "col": col,
                "row": row,
                "pred_class": pred_class,
                "epi": 0.0,
                "ale": 0.0,
            }
            i += 1
    return d


# ---------------------------------------------------------------------------
# Bug #1 regression: X/Y axis swap
# ---------------------------------------------------------------------------


class TestOverlayAxisAlignment:
    """Cells must use row-axis (height) for the row slice and col-axis (width)
    for the col slice. The previous code swapped them, so on a non-square
    thumbnail a fully-populated grid produced cells whose height ~ width/W
    and width ~ height/H — visibly non-square and misaligned."""

    def test_non_square_thumbnail_cells_keep_aspect(self, monkeypatch):
        H, W = 60, 120  # 2:1 thumbnail
        n_cols, n_rows = 3, 2  # 3 across, 2 down → 6 cells
        cls, saved = _make_classification_with_thumbnail(monkeypatch, H, W)
        cls.dictionary = _grid_dictionary(n_cols, n_rows, pred_class="AC")

        cls.overlay(typean="analysis", unc="Pred_class")

        # The combined-class overlay (res_name index 0) paints every AC cell.
        combined = saved["/fake/result/Pred_class.png"]
        # Channel 0 (red) was += 1 for every AC cell → exactly H*W cell coverage.
        red = combined[:, :, 0]
        painted = np.count_nonzero(red > 0.5)
        assert painted == H * W, (
            f"Expected full {H}x{W}={H * W} px coverage, got {painted} — "
            "cells are not tiling the canvas (axis swap or wrong step)."
        )

    def test_uncertainty_modes_cover_full_canvas(self, monkeypatch):
        """Same geometry invariant for the epi/ale/tot branches, which use a
        different code path (only channel 2/3 is painted)."""
        H, W = 80, 100
        n_cols, n_rows = 4, 2
        for unc in ("epi", "ale", "tot"):
            cls, saved = _make_classification_with_thumbnail(monkeypatch, H, W)
            cls.dictionary = _grid_dictionary(n_cols, n_rows, pred_class="AC")
            # Give each tile a non-zero uncertainty so something is painted.
            for k in cls.dictionary:
                cls.dictionary[k]["epi"] = 0.5
                cls.dictionary[k]["ale"] = 0.5

            cls.overlay(typean="analysis", unc=unc)

            img = saved[f"/fake/result/uncertainty/{unc}.png"]
            # Channel 2 was += |u|; threshold low to capture the whole cell.
            painted = np.count_nonzero(img[:, :, 2] > 0.1)
            assert painted == H * W, (
                f"unc={unc}: expected {H * W} painted px, got {painted}"
            )


# ---------------------------------------------------------------------------
# Bug #2 regression: rounding drift → gaps/overlaps between adjacent cells
# ---------------------------------------------------------------------------


class TestOverlayNoGapsNoOverlaps:
    """Adjacent cells must share an exact integer boundary. The previous
    ``round(column * step_x)`` could leave 1-px gaps or 1-px overlaps between
    neighbouring cells (rounding drift). With the integer partition
    ``floor(k*W/N)`` / ``floor((k+1)*W/N)``, cell boundaries coincide."""

    def test_no_gap_no_overlap_on_awkward_size(self, monkeypatch):
        # 100 px wide / 3 cols → step = 33.33; rounding drift would surface.
        # 70 px tall / 2 rows → step = 35; integer partition is exact.
        H, W = 70, 100
        n_cols, n_rows = 3, 2
        cls, saved = _make_classification_with_thumbnail(monkeypatch, H, W)
        cls.dictionary = _grid_dictionary(n_cols, n_rows, pred_class="AC")

        cls.overlay(typean="analysis", unc="Pred_class")
        combined = saved["/fake/result/Pred_class.png"]

        # Full coverage (no gaps): every pixel painted exactly once.
        red = combined[:, :, 0]
        assert np.count_nonzero(red > 0.5) == H * W

        # No overlaps: a cell painted twice (overlapping neighbour) would
        # push channel 0 to 2.0 in the overlap region. Clamp-guard: assert
        # no pixel exceeds 1.0 + tiny epsilon (^ the += bug pattern).
        assert red.max() <= 1.0 + 1e-9, (
            "Channel 0 exceeded 1.0 → adjacent cells overlapped (the rounding-"
            "drift bug). max=" + str(red.max())
        )

    def test_cell_boundaries_are_integer_exact(self, monkeypatch):
        """The forward edge of cell k must equal the back edge of cell k+1.
        Concretely: column c covers x in [c0, c1) where c0 = c*W//n_cols and
        c1 = (c+1)*W//n_cols, so c1 of one cell == c0 of the next."""
        H, W = 30, 100
        n_cols, n_rows = 3, 1  # single row → isolate the column axis
        cls, saved = _make_classification_with_thumbnail(monkeypatch, H, W)
        cls.dictionary = _grid_dictionary(n_cols, n_rows, pred_class="AC")

        cls.overlay(typean="analysis", unc="Pred_class")
        combined = saved["/fake/result/Pred_class.png"]
        red = combined[:, :, 0]

        # Expected column splits for W=100, n_cols=3: [0,33), [33,66), [66,100).
        expected_splits = [(0, 33), (33, 66), (66, 100)]
        cols_in_order = sorted(
            {v["col"] for v in cls.dictionary.values()}
        )
        for (c_start, c_end), _col in zip(expected_splits, cols_in_order, strict=True):
            block = red[:, c_start:c_end]
            # The block must be fully painted (step_x portion of the cell).
            assert np.count_nonzero(block > 0.5) == H * (c_end - c_start), (
                f"x=[{c_start},{c_end}) not fully painted"
            )
        # And the gaps between blocks must be zero-width (boundaries coincide).
        for c in range(n_cols - 1):
            boundary = (c + 1) * W // n_cols
            # No unpainted column at the boundary.
            assert np.count_nonzero(red[:, boundary] <= 0.5) == 0
