"""Unit tests for ``StartAnalysis._build_partition_args`` — the pure helper
that assembles the per-partition tuples consumed by the
``ProcessPoolExecutor`` workers in the upgraded tile generator.

Regression motivation: the previous threading version
(:py:meth:`StartAnalysis.process_to_start`) read partition state directly
off ``self`` inside the worker; the new ProcessPool version instead
serialises that state into a picklable tuple per partition (so it can
cross the process boundary under the ``spawn`` start method on Windows /
macOS). If a field is mis-ordered or dropped, every worker silently opens
the wrong slide or writes to the wrong folder. Locking the tuple shape
here is the cheapest place to catch it.
"""

from __future__ import annotations

from src.multi_processing_analysis import _build_partition_args


def _fields():
    """Yield the canonical field names in the order the worker unpacks them.
    The worker does::

        (x_start, x_stop, process_name, tile_start, file_path, lev_sec,
         tile_size, overlap, limit_bounds, base_folder, n_rows, levi) = args

    If a field gets reordered in production code without updating this
    test, at least one assertion below fails — that's the regression we
    want to catch.
    """
    return [
        "x_start",
        "x_stop",
        "process_name",
        "tile_start",
        "file_path",
        "lev_sec",
        "tile_size",
        "overlap",
        "limit_bounds",
        "base_folder",
        "n_rows",
        "levi",
    ]


def _kwargs():
    """Canonical kwargs for a 2-partition run."""
    return dict(
        numx_start=[0, 4],
        numx_stop=[4, 8],
        list_proc=["p_0_4_71", "p_4_8_71"],
        start_indexs=[1, 285],
        file_path="/data/slide.svs",
        lev_sec=1,
        tile_size=64,
        overlap=0,
        limit_bounds=True,
        base_folder="/out/data/slide_1/",
        n_rows=71,
        levi=9,
        existing=set(),
    )


class TestBuildPartitionArgs:
    """Tuple shape, ordering, and existing-folder filtering."""

    def test_one_tuple_per_partition(self):
        args = _build_partition_args(**_kwargs())
        assert len(args) == 2

    def test_tuple_length_matches_worker_unpack_contract(self):
        args = _build_partition_args(**_kwargs())
        for tup in args:
            assert len(tup) == len(_fields())

    def test_tuple_field_order_matches_worker_unpack(self):
        """The worker unpacks ``args`` in a fixed order; this asserts each
        position carries the field we documented. Catches reordering
        bugs that would silently swap, e.g., ``lev_sec`` and ``levi``."""
        kw = _kwargs()
        args = _build_partition_args(**kw)
        # Partition 0 expected tuple:
        expected_p0 = (
            kw["numx_start"][0],
            kw["numx_stop"][0],
            kw["list_proc"][0],
            kw["start_indexs"][0],
            kw["file_path"],
            kw["lev_sec"],
            kw["tile_size"],
            kw["overlap"],
            kw["limit_bounds"],
            kw["base_folder"],
            kw["n_rows"],
            kw["levi"],
        )
        assert args[0] == expected_p0

    def test_partitions_preserve_input_order(self):
        """Partition order must match ``list_proc`` order so the start/stop
        range and the running ``tile_start`` counter stay aligned — the
        PNG filename counter depends on this."""
        kw = _kwargs()
        args = _build_partition_args(**kw)
        names = [a[2] for a in args]
        assert names == kw["list_proc"]

    def test_existing_partition_is_filtered_out(self):
        """If ``list_proc[i]`` is already on disk, that partition is skipped
        (matches the legacy ``folder_manage`` contract — never overwrite an
        existing partition folder, so a partial run resumes cleanly)."""
        kw = _kwargs()
        kw["existing"] = {"p_0_4_71"}  # first partition already done
        args = _build_partition_args(**kw)
        assert len(args) == 1
        assert args[0][2] == "p_4_8_71"  # only the second one remains

    def test_all_existing_returns_empty_list(self):
        kw = _kwargs()
        kw["existing"] = set(kw["list_proc"])
        assert _build_partition_args(**kw) == []

    def test_per_partition_fields_are_picklable_builtins(self):
        """``spawn`` start method pickles the tuples to children — every
        field must be a builtin (int, str, bool). A non-builtin (e.g. a
        ``Path`` or a Qt object) would break the ProcessPoolExecutor under
        ``spawn`` on Windows / macOS."""
        args = _build_partition_args(**_kwargs())
        for tup in args:
            for v in tup:
                assert isinstance(v, (int, str, bool)), (
                    f"non-picklable-builtin type {type(v).__name__} in partition tuple"
                )

    def test_per_partition_levi_inherited_from_caller(self):
        """All partitions share the same ``levi`` (DeepZoom level index)
        — the slide-wide configuration is the same across the x-axis
        split. Asserts the helper doesn't accidentally partition the level."""
        args = _build_partition_args(**_kwargs())
        levs = {a[11] for a in args}
        assert levs == {9}  # matches the kwarg we passed
