"""Unit tests for ``StartAnalysis.manage_process()``.

This method computes the x-axis tile partition for parallel tile generation.
It's pure Python (uses only ``math.ceil`` and ``multiprocessing.cpu_count()``)
so it qualifies for the fast unit-test tier — no openslide, no Qt, no TF.

Regression motivation: a bug in the start/stop / index math would silently
slice the wrong tiles out of every WSI and produce off-by-one artifact folders;
this is the cheapest place to lock the contract.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from src.multi_processing_analysis import StartAnalysis


@pytest.fixture
def analysis():
    """A StartAnalysis with the minimum attributes manage_process touches.

    We don't open a slide — manage_process() only reads ``self.*`` for the
    parts explicitly used in the method body (none of the OpenSlide state).
    """
    return StartAnalysis()


def _expected_partition(numtotx, numtoty, n_core):
    """Reference implementation of manage_process() used as the oracle.

    Kept independent (no shared code) so we are validating the production
    logic, not asserting it against itself.
    """
    if n_core >= numtotx:
        n_core = 1
    step_x = math.ceil(numtotx / n_core)
    images_per_process = numtoty * step_x

    numx_start, numx_stop, list_proc, start_idx, stop_idx = [], [], [], [], []
    for num_pro in range(1, n_core + 1):
        if (num_pro - 1) * step_x < numtotx:
            numx_start.append((num_pro - 1) * step_x)
            numx_stop.append(numx_start[-1] + step_x)
            if numx_stop[-1] > numtotx:
                numx_stop[-1] = numtotx
            start_idx.append((num_pro - 1) * images_per_process + 1)
            stop_idx.append(num_pro * images_per_process)
            list_proc.append(
                f"p_{numx_start[-1]}_{numx_stop[-1]}_{numtoty}"
            )
        else:
            # Last partition absorbs the remainder — adjust the last stop.
            if stop_idx:
                stop_idx[-1] = numtotx * numtoty
            break
    return numx_start, numx_stop, list_proc, start_idx, stop_idx


class TestManageProcessShape:
    """Return-shape contract: always a 5-tuple of equal-length lists."""

    @pytest.mark.parametrize(
        "numtotx,numtoty",
        [(1, 1), (10, 5), (63, 71), (100, 1), (1, 100)],
    )
    def test_returns_five_lists_of_equal_length(self, analysis, numtotx, numtoty):
        with patch("multiprocessing.cpu_count", return_value=4):
            result = analysis.manage_process(numtotx, numtoty)
        assert len(result) == 5
        for lst in result:
            assert isinstance(lst, list)
        # All five lists must have the same length (one entry per partition).
        lengths = {len(lst) for lst in result}
        assert len(lengths) == 1


class TestManageProcessValues:
    """Cross-check the production method against the reference oracle."""

    @pytest.mark.parametrize("n_core", [1, 2, 4, 8, 16])
    @pytest.mark.parametrize("numtotx,numtoty", [(10, 5), (63, 71), (45, 28)])
    def test_matches_reference_implementation(
        self, analysis, numtotx, numtoty, n_core
    ):
        with patch("multiprocessing.cpu_count", return_value=n_core):
            got = analysis.manage_process(numtotx, numtoty)
        expected = _expected_partition(numtotx, numtoty, n_core)
        assert got == expected

    def test_single_core_when_numtotx_below_cpu_count(self, analysis):
        """The branch ``n_core >= numtotx`` collapses to 1 partition so we
        don't oversubscribe a tiny image across many cores."""
        with patch("multiprocessing.cpu_count", return_value=16):
            numx_start, numx_stop, list_proc, *_ = analysis.manage_process(4, 71)
        assert len(numx_start) == 1
        assert numx_start == [0] and numx_stop == [4]
        assert list_proc == ["p_0_4_71"]

    def test_last_partition_clamped_to_numtotx(self, analysis):
        """``numx_stop[-1]`` must never exceed ``numtotx`` — otherwise the
        ``range(x_start, x_stop)`` loop in _create_tiles runs off the grid and
        raises ``ValueError: Invalid address`` from DeepZoomGenerator."""
        with patch("multiprocessing.cpu_count", return_value=4):
            _, numx_stop, _, _, _ = analysis.manage_process(63, 71)
        assert all(s <= 63 for s in numx_stop)
        # Total covered: from the first start (0) to the last stop (== numtotx).
        assert numx_stop[-1] == 63

    def test_process_name_format_matches_consumer(self, analysis):
        """``_folder_exists`` + ``Classification.select_folder`` parse the
        ``p_<xs>_<xe>_<y>`` format — lock the slash format."""
        with patch("multiprocessing.cpu_count", return_value=4):
            _, _, list_proc, _, _ = analysis.manage_process(63, 71)
        for name in list_proc:
            parts = name.split("_")
            assert parts[0] == "p"
            assert len(parts) == 4
            assert parts[-1] == "71"  # numtoty

    def test_start_index_is_one_based(self, analysis):
        """``manage_process`` uses 1-based tile indices (``start_idx`` starts
        at 1, not 0). The PNG filename counter depends on this — see
        ui_pyqt5.py ``is_primary = tile_start == 1``."""
        with patch("multiprocessing.cpu_count", return_value=4):
            _, _, _, start_idx, _ = analysis.manage_process(63, 71)
        assert start_idx[0] == 1

    def test_partitions_are_contiguous_in_x(self, analysis):
        """Adjacent partitions must touch: stop[i] == start[i+1]. Otherwise a
        column of tiles is silently dropped or duplicated."""
        with patch("multiprocessing.cpu_count", return_value=8):
            numx_start, numx_stop, _, _, _ = analysis.manage_process(63, 71)
        for i in range(len(numx_start) - 1):
            assert numx_stop[i] == numx_start[i + 1]
        # And the first partition starts at 0, last ends at numtotx.
        assert numx_start[0] == 0
        assert numx_stop[-1] == 63
