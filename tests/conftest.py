"""Top-level pytest configuration shared by every test subfolder.

This file:
* Bootstraps ``sys.path`` so the snake_case source packages (``src.``,
  ``models.``) and the root-level entry points (``ui_pyqt5``, ``ui_dataclean``)
  resolve — exactly the same CWD=repo-root invariant documented in README.md.
* Registers pytest markers (declared in ``pyproject.toml``'s
  ``[tool.pytest.ini_options].markers``) so ``--strict-markers`` stays happy
  and ``pytest --markers`` documents them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make the repo root importable from anywhere; tests are often structured as
# independent modules and pytest may be invoked from the repo root or from a
# subfolder (CI matrix). Resolve once at import time.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Allow tests/fixtures to find small SVS samples by absolute path without
# hard-coding them. ``tests/fixtures`` is the conventional place; absence of a
# fixture is signalled by the ``svs_path`` fixture returning None, and any
# test that needs an SVS must opt-in via ``pytestmark = pytest.mark.openslide``
# and skip itself when ``svs_path`` is None.
FIXTURES_DIR = _REPO_ROOT / "tests" / "fixtures"


def pytest_configure(config: pytest.Config) -> None:
    """Register markers (declared in pyproject.toml) and add custom ini-style
    options. Not strictly required with --strict-markers since they are already
    declared in [tool.pytest.ini_options], but kept as documentation."""
    for marker in [
        "openslide: tests requiring libopenslide + an .svs fixture",
        "tf: tests requiring TensorFlow",
        "gui: tests requiring a Qt offscreen platform (pytest-qt)",
        "slow: long-running / full-pipeline tests",
    ]:
        config.addinivalue_line("markers", marker)


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Absolute path to ``tests/fixtures`` — create it on demand so a missing
    folder never triggers FileNotFoundError, only the skip in dependent tests."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def svs_path(fixtures_dir: Path):
    """Return the path to a small ``.svs`` test fixture, or ``None``.

    Tests that need a real slide must (a) be marked ``openslide`` and
    (b) ``pytest.skip`` when this fixture is ``None`` — this is what keeps the
    fast CI unit job green even when no fixture is available locally/CI-side.
    """
    candidate = fixtures_dir / "tiny.svs"
    return candidate if candidate.exists() else None
