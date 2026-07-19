"""Shared mutable state for the Datacleaning tabs.

The original ``MainTabWidget`` kept every piece of cross-tab state as a
plain attribute on ``self``. That worked while all 5 tabs lived in one
class, but once each tab became its own ``QWidget`` subclass, the tabs
needed somewhere common to read/write the same paths, parameters, and
analysis results.

``DataCleanState`` is that common place. The :class:`MainTabWidget`
owns a single instance and passes it to every tab constructor. Each tab
reads/writes the attributes it needs; the coordinator wires up the
worker callbacks that update ``training_log`` and the uncertainty
collections.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.uncertainty_analysis import Th

from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_MODEL_PATH,
    DEFAULT_MONTE_CARLO_SAMPLES,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VAL_PATH,
    DEFAULT_WORK_PATH,
)


@dataclass
class DataCleanState:
    """Mutable shared state across the five Datacleaning tabs.

    Attributes are intentionally left un-typed where the original code
    didn't constrain them (e.g. ``selected_threshold`` is ``str`` for
    auto/otsu/new modes but ``float`` for manual mode). Tabs just read
    and write what they need; the coordinator's job is to wire them
    together.
    """

    # --- Paths (set by Get-Tiles / Training / Cleaning) ---
    train_path: str = DEFAULT_TRAIN_PATH
    val_path: str = DEFAULT_VAL_PATH
    test_path: str = ""
    model_path: str = DEFAULT_MODEL_PATH
    work_path: str = DEFAULT_WORK_PATH
    tiles_train_path: str = ""
    tiles_val_path: str = ""
    tiles_test_path: str = ""
    clean_save_path: str = ""

    # --- Training parameters ---
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    monte_carlo_samples: int = DEFAULT_MONTE_CARLO_SAMPLES
    model_type: str = "drop"
    use_augmentation: bool = False

    # --- Uncertainty JSON results per dataset ---
    train_json: str = "new_train_js.txt"
    val_json: str = "new_val_js.txt"
    test_json: str = ""

    # --- Cleaning tab working data ---
    selected_threshold: object = ""
    threshold_flag: int = 0
    cleaning_obj: Th | None = None
    aleatoric_values: list[float] = field(default_factory=list)
    epistemic_values: list[float] = field(default_factory=list)
    total_uncertainty_values: list[float] = field(default_factory=list)

    # --- Training live log (streamed via the ``view`` worker kwarg) ---
    training_log: str = ""
