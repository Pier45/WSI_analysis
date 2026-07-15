"""
Project-wide configuration constants.

Single source of truth for class names, input shape and class count,
imported by ``models/``, ``src/`` and the GUI entry points.

Canonical class order is alphabetical: ``['AC', 'AD', 'H']``.
This order is used by:

* ``flow_from_directory(classes=CLASS_NAMES)`` → defines class-index
  assignment in the trained model's output softmax.
* ``Classification.cl`` → maps predicted argmax back to a class name.
* ``confusion_matrix(labels=CLASS_NAMES)`` in ``test_widget``.

Any change here must be reflected in the saved model's expected output
order; do not silently reorder — pre-trained ``.h5`` checkpoints encode
the previous order.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

# Histological-tile classes (colorectal). Order defines argmax index.
CLASS_NAMES: Tuple[str, ...] = ("AC", "AD", "H")

# Number of classes — derived from CLASS_NAMES so they cannot drift.
N_CLASSES: int = len(CLASS_NAMES)

# Input tile shape expected by the Bayesian CNNs.
INPUT_SHAPE: Tuple[int, int, int] = (64, 64, 3)

# Override directory for per-SVS analysis output (tiles, thumbnail, result,
# uncertainty overlays, dictionary JSON). When unset, output is written next
# to the input .svs file (``<svs_dir>/data/<svs_name>_<lev_sec>/``). When set
# (e.g. via the ``WSI_OUTPUT_DIR`` env var), output goes to
# ``<WSI_OUTPUT_DIR>/data/<svs_name>_<lev_sec>/`` instead — useful for
# read-only SVS sources and Docker containers where the host bind-mounts a
# dedicated output volume. ``None`` means "use the SVS file's directory".
WSI_OUTPUT_DIR: Optional[str] = os.environ.get("WSI_OUTPUT_DIR")
