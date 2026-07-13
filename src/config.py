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

from typing import Tuple

# Histological-tile classes (colorectal). Order defines argmax index.
CLASS_NAMES: Tuple[str, ...] = ("AC", "AD", "H")

# Number of classes — derived from CLASS_NAMES so they cannot drift.
N_CLASSES: int = len(CLASS_NAMES)

# Input tile shape expected by the Bayesian CNNs.
INPUT_SHAPE: Tuple[int, int, int] = (64, 64, 3)
