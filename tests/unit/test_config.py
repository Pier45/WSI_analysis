"""Unit tests for ``src.config`` — pure Python, no optional deps.

These run in any CI matrix (Windows, Linux) and constitute the fastest tier
of the test suite: import the module and assert its exported constants are
the canonical values that the trained ``.h5`` checkpoints encode.

Regression motivation: any change to ``CLASS_NAMES`` order silently
misaligns softmax indices in the saved model, so we lock the order here.
"""

from __future__ import annotations

import src
from src.config import CLASS_NAMES, INPUT_SHAPE, N_CLASSES, WSI_OUTPUT_DIR


class TestConfigConstants:
    """Lock the canonical class/shape contract — changing these is a breaking
    change that requires retraining and a migration note in CHANGES.md."""

    def test_class_names_order_is_alphabetical(self):
        """Canonical order is alphabetical, defining argmax index."""
        assert CLASS_NAMES == ("AC", "AD", "H")

    def test_class_names_is_a_tuple_not_list(self):
        """Tuples are immutable and hashable — guards against accidental
        in-place mutation that would silently corrupt downstream consumers."""
        assert isinstance(CLASS_NAMES, tuple)

    def test_n_classes_derived_from_class_names(self):
        """``N_CLASSES`` must stay in sync with ``len(CLASS_NAMES)``."""
        assert N_CLASSES == len(CLASS_NAMES) == 3

    def test_input_shape_is_64x64x3(self):
        """The Bayesian CNNs are trained on 64x64 RGB tiles — any other shape
        is an input-contract break."""
        assert INPUT_SHAPE == (64, 64, 3)
        assert len(INPUT_SHAPE) == 3

    def test_reexports_match_package_init(self):
        """``src/__init__.py`` re-exports the three constants — they must be
        the same objects (no shadow copies)."""
        assert src.CLASS_NAMES is CLASS_NAMES
        assert src.N_CLASSES is N_CLASSES
        assert src.INPUT_SHAPE is INPUT_SHAPE

    def test_wsi_output_dir_reads_env_at_import_time(self, monkeypatch):
        """``WSI_OUTPUT_DIR`` is read once at import time from the environment.
        Re-importing under a fresh env var value returns that value. (We can
        only verify the *behaviour at import* indirectly, by confirming it
        reflects whatever was set when ``src.config`` was imported — i.e. the
        current value, which on CI is unset -> None.)"""
        import os

        # On a clean CI env this is None; locally it may be a string. Either is
        # acceptable: we just assert it equals os.environ.get at first import.
        assert WSI_OUTPUT_DIR == os.environ.get("WSI_OUTPUT_DIR")
