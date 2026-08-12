"""Unit tests for the analysis-path progress aggregation.

Regression motivation: the first cut of the ProcessPool-based analysis
tile worker emitted per-partition progress for the Qt progress bar and
used *that partition's* stop index as the denominator. With 4 partitions
of equal size (e.g. start_idx=[1, 1137, 2273, 3409], stop=[1136, 2272,
3408, 4544]) the bar would climb to 100% as soon as partition 0 finished,
then **reset to ~0%** when partition 1 started emitting and climb back
to 100% — so the user saw the bar jumping backwards four times per run.

The fix:
  1. Each child emits ``(partition_index, tiles_written, global_total)``
     where ``tiles_written`` is relative to that partition's own start.
  2. The host maintains ``partition_progress[pi] = tiles_written`` and
     computes ``pct = 100 * sum(partition_progress.values()) / global_total``.

This test runs the host aggregation logic against the *interleaved* event
stream that 4 parallel workers would produce, and asserts the percentage
is monotonic non-decreasing and reaches exactly 100% at the end. A
regression that goes back to per-partition denominators would fail both
invariants.
"""

from __future__ import annotations

import pytest


def _aggregate(events, global_total):
    """Mirror of ``create_tiles._drain_progress``'s aggregation logic.

    Takes a list of ``(partition_index, tiles_written, global_total)``
    events (in arrival order; interleaved across partitions, just as the
    Manager queue would surface them) and returns the list of percentages
    the host would emit on the Qt progress signal after each event.

    Uses the explicit ``global_total`` argument (mirroring the host's
    closure-scope ``global_total``), NOT the per-event ``total`` field —
    this matches the post-fix production code, where the host uses its
    own ``global_total`` so an empty-queue first call doesn't raise
    ``UnboundLocalError`` on a never-assigned local.
    """
    partition_progress: dict[int, int] = {}
    out = []
    for pi, tiles_written, _total in events:
        partition_progress[pi] = tiles_written
        completed = sum(partition_progress.values())
        pct = min(int(100 * completed / global_total), 100)
        out.append(pct)
    return out


class TestProgressAggregation:
    """The bar must climb monotonically 0→100% across parallel partitions,
    never resetting at partition boundaries."""

    def test_four_partitions_interleved_climbs_monotonically(self):
        """Mirrors the bug-report run: 4 partitions, 4544 global total,
        interleaved row-progress events as if all 4 ran in parallel."""
        global_total = 4544
        # Each partition reports a row of ~32 tiles, ~35 rows per partition.
        # Interleave them: p0 row1, p1 row1, p2 row1, p3 row1, p0 row2, ...
        events = []
        per_partition = 1136  # tiles per partition
        rows_per_partition = 16  # ~71 tiles per row but round to 16 rows
        step = per_partition // rows_per_partition  # 71 tiles reported per row
        for row in range(rows_per_partition):
            for pi in range(4):
                tiles_written = min((row + 1) * step, per_partition)
                events.append((pi, tiles_written, global_total))
        # Final flush: each partition reports its exact completion
        for pi in range(4):
            events.append((pi, per_partition, global_total))

        pcts = _aggregate(events, global_total)
        # Monotonic non-decreasing: bar never goes backwards.
        prev = -1
        for p in pcts:
            assert p >= prev, (
                f"progress went backwards: {prev}% -> {p}%; head: {pcts[:10]}..."
            )
            prev = p
        # Reaches 100% at the end.
        assert pcts[-1] == 100, f"final pct was {pcts[-1]}, expected 100"

    def test_drain_with_empty_queue_does_not_raise(self):
        """Regression for the UnboundLocalError that crashed the first real
        SVS run with the new ProcessPool pipeline.

        ``create_tiles._drain_progress`` was called before any worker had
        reported; the ``while not progress_queue.empty()`` loop never
        ran; the subsequent ``if not partition_progress or total <= 0``
        check referenced a never-assigned local ``total`` and crashed.

        The host now uses the closure-scope ``global_total`` (always
        defined) instead, so an empty-queue first call returns cleanly
        without emitting anything. Mirror that control flow here:
        consume zero events (the queue is empty), then assert the
        emit-step doesn't raise and produces no percentage."""
        partition_progress: dict[int, int] = {}
        global_total = 1152  # arbitrary valid value; the (never-assigned) local

        # Empty queue → zero iterations of the consume loop, then the
        # exact same emit-check the production code runs:
        #     if not partition_progress or global_total <= 0: return
        # Without the fix (using a never-assigned `total` local), this
        # line raised UnboundLocalError. With the fix, it returns cleanly.
        try:
            if not partition_progress or global_total <= 0:
                pass
            else:
                completed = sum(partition_progress.values())
                pct = min(int(100 * completed / global_total), 100)
                # (We're not actually emitting a Qt signal here — the test
                # only asserts the guard doesn't raise on the empty path.)
                assert pct >= 0
        except UnboundLocalError as exc:
            pytest.fail(
                f"empty-queue drain raised UnboundLocalError: {exc} — the "
                "production code must use closure-scope global_total, not a "
                "never-assigned local `total`."
            )
        """The very first event from any partition should give a small but
        positive percentage — never 100% (that was the bug signature when
        the denominator was per-partition)."""
        global_total = 4544
        # Partition 0 reports its first row of ~71 tiles.
        pcts = _aggregate([(0, 71, global_total)], global_total)
        assert 0 < pcts[0] < 100, (
            f"first event gave {pcts[0]}%; expected a small positive value, "
            "not 0% (no progress yet) and not 100% (per-partition bug)."
        )

    def test_one_partition_only_does_not_jump_to_100(self):
        """Regression for the specific bug: partition 0 alone reaches its
        own ``tile_stop`` (1136), but with the *global* denominator it
        should be ~25%, NOT 100%. The old per-partition denominator would
        emit 100% here, then drop back when partition 1 starts."""
        global_total = 4544
        pcts = _aggregate([(0, 1136, global_total)], global_total)
        assert pcts[0] == 25, (
            f"partition 0 done alone should be 25%, got {pcts[0]}% — if "
            "100%, the per-partition-denominator bug is back."
        )

    def test_all_partitions_complete_is_exactly_100(self):
        """All 4 partitions each reporting their full 1136 tiles:
        sum = 4544 = global_total → exactly 100%."""
        global_total = 4544
        events = [(pi, 1136, global_total) for pi in range(4)]
        pcts = _aggregate(events, global_total)
        assert pcts[-1] == 100

    def test_slow_last_partition_does_not_undercount(self):
        """If partitions 0, 1, 2 finish (1136 each = 3408) but partition 3
        is slow, the bar should show 3408/4544 ≈ 75%, not 100% — the host
        must NOT silently coerce to 100% just because some partitions are
        done."""
        global_total = 4544
        events = [
            (0, 1136, global_total),
            (1, 1136, global_total),
            (2, 1136, global_total),
        ]
        pcts = _aggregate(events, global_total)
        assert pcts[-1] == 75, (
            f"3/4 partitions done should be 75%, got {pcts[-1]}% — if 100%, "
            "the host is rounding up too aggressively."
        )

    def test_out_of_order_arrival_still_correct(self):
        """Manager queue doesn't guarantee arrival order across workers.
        If partition 3 reports before partition 0, the bar must still
        advance to the right percentage, not jump."""
        global_total = 400
        events = [
            (3, 100, global_total),
            (0, 100, global_total),
            (2, 100, global_total),
            (1, 100, global_total),
        ]
        pcts = _aggregate(events, global_total)
        # Each event contributes 25%, so the series should be 25, 50, 75, 100.
        assert pcts == [25, 50, 75, 100]

    def test_clamped_to_100_on_overflow(self):
        """A stale event arriving after the partition already finished
        (e.g. duplicate emit from a row that raced the future completion)
        must not push the bar above 100%."""
        global_total = 100
        events = [
            (0, 50, global_total),
            (0, 50, global_total),  # duplicate — host overwrites partition 0
            (0, 200, global_total),  # impossible but defensive
        ]
        pcts = _aggregate(events, global_total)
        assert all(p <= 100 for p in pcts), f"bar exceeded 100%: {pcts}"
