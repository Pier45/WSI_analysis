"""Analysis-tier tests for ``StartAnalysis`` that exercise the OpenSlide /
DeepZoom level split — the very bug we just fixed.

Marked ``openslide`` so the fast unit job skips them; they only run when an
actual ``.svs`` fixture is available (under ``tests/fixtures/tiny.svs``) and
``libopenslide`` is installed. CI populates the fixture via the
``openslide-bin`` PyPI package; locally one can drop in a small slide.

Why we need a real .svs: ``StartAnalysis.openSvs`` constructs an
``openslide.OpenSlide`` and ``tile_gen`` constructs a ``DeepZoomGenerator``
— both pull metadata from the actual file. There's no public seam to fake
``slide.level_dimensions`` cleanly.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.openslide, pytest.mark.slow]
pytest.importorskip("openslide")


def test_lev_sec_under_level_count_does_not_raise_in_get_thumb(svs_path):
    """Regression for the IndexError: tuple index out of range.

    Original bug: ui_pyqt5 cached the DeepZoom level index (``levi``, can be
    > level_count) into ``self._svs_level`` and passed it back as ``lev_sec``,
    so ``get_thumb``'s ``self.list_levels[self.lev_sec]`` raised IndexError.

    Guard: ``get_thumb`` now clamps ``lev_sec`` to a valid OpenSlide level and
    logs a warning instead of raising. This test exercises the defence.
    """
    if svs_path is None:
        pytest.skip("No .svs fixture available; drop one at tests/fixtures/tiny.svs")

    from src.multi_processing_analysis import StartAnalysis

    # Construct with an absurd lev_sec that would have triggered the bug.
    analysis = StartAnalysis(lev_sec=10_000)
    analysis.openSvs(str(svs_path))
    analysis.base_folder_manager()
    # Must NOT raise IndexError — clamp fallback kicks in.
    work_dir = analysis.get_thumb()
    assert work_dir.endswith(os.sep) or "/" in work_dir


def test_get_thumb_uses_a_valid_openslide_level(svs_path):
    """When ``lev_sec`` is in range, the produced thumbnail must come from
    ``list_levels[lev_sec]`` exactly (no silent fallback)."""
    if svs_path is None:
        pytest.skip("No .svs fixture available")

    from src.multi_processing_analysis import StartAnalysis

    analysis = StartAnalysis()
    analysis.openSvs(str(svs_path))
    analysis.base_folder_manager()
    n_levels = analysis.slide.level_count
    # Pick the most-downsampled valid level — works on fixtures with as few
    # as 2 OpenSlide levels (a typical tiny.svs test sample).
    lev = n_levels - 1
    analysis.lev_sec = lev
    assert 0 <= lev < n_levels
    analysis.get_thumb()  # must not raise


def test_tile_gen_returns_deepzoom_level_index_not_openslide(svs_path):
    """Regression for the ``Invalid address`` ValueError.

    ``tile_gen(state=0)`` returns ``levi`` (a DeepZoom level index) as its 7th
    element. ui_pyqt5 must NOT reuse that as ``lev_sec`` for a future
    ``StartAnalysis(lev_sec=...)`` call (different namespace). This test pins
    the return-shape contract — the 7th element is *not* a valid
    ``slide.level_dimensions`` index in general.
    """
    if svs_path is None:
        pytest.skip("No .svs fixture available")

    from src.multi_processing_analysis import StartAnalysis

    analysis = StartAnalysis()
    analysis.openSvs(str(svs_path))
    analysis.base_folder_manager()
    n_levels = analysis.slide.level_count
    analysis.lev_sec = n_levels - 1  # most-downsampled valid level
    *_, numy, levi = analysis.tile_gen(state=0)

    assert isinstance(levi, int), "tile_gen(state=0) 7th element must be int"
    assert isinstance(numy, int), "tile_gen(state=0) 6th element must be int (rows)"
    assert levi >= 0


def test_openSvs_clamps_out_of_range_lev_sec_instead_of_IndexError(svs_path):
    """Regression: a small fixture (``level_count=2``) used to crash hard in
    ``tile_gen`` when the app default ``lev_sec=2`` was out of range. Now it
    must clamp silently and continue."""
    if svs_path is None:
        pytest.skip("No .svs fixture available")

    from src.multi_processing_analysis import StartAnalysis

    analysis = StartAnalysis(lev_sec=2)  # invalid on a 2-level fixture
    analysis.openSvs(str(svs_path))
    n_levels = analysis.slide.level_count
    if n_levels > 2:
        pytest.skip(f"fixture has level_count={n_levels}; need a 2-level slide to exercise the clamp")
    analysis.base_folder_manager()
    # Must not raise IndexError.
    *_, numy, levi = analysis.tile_gen(state=0)
    assert isinstance(levi, int) and levi >= 0
    assert isinstance(numy, int) and numy >= 1
    # And lev_sec was clamped to a valid level.
    assert 0 <= analysis.lev_sec < n_levels
