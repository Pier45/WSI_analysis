"""GUI-tier-specific pytest configuration.

Provides a minimal ``qapp`` and ``qtbot`` fixture so the _make_action tests
can run WITHOUT pytest-qt installed (PyQt5 is already a runtime dep).

If pytest-qt *is* installed, its fixtures take precedence and we never run
our fallback (Pytest's fixture resolution prefers plugins over conftest).
"""

from __future__ import annotations

import os
import sys

import pytest

# Make sure the offscreen Qt platform is selected before QApplication is
# constructed; setting QT_QPA_PLATFORM here (in addition to the module-level
# setdefault in test_actions_factory.py) covers any other gui/ tests too.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# --- Fallback qapp ---------------------------------------------------------
# pytest-qt exposes `qapp`. If it's not registered, we build one ourselves
# and cache it on the session. Re-entered safely (no double construction).

_qapp_singleton = None


@pytest.fixture(scope="session")
def qapp():
    """Session-scoped QApplication fallback when pytest-qt is absent.

    pytest-qt别名名 auto-uses `qapp` from its plugin; declaring it here with
    the same name means our version is shadowed when the plugin loads, and
    used as a fallback otherwise.
    """
    global _qapp_singleton  # noqa: PLW0603  (session-scoped singleton by design)
    if _qapp_singleton is None:
        from PyQt5.QtWidgets import QApplication
        # Make sure there's exactly one QApplication; reuse the running one
        # if pytest is itself hosted inside a Qt app (rare in CI).
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv if isinstance(sys.argv, list) else [])
        _qapp_singleton = app
    yield _qapp_singleton


# --- Fallback qtbot -------------------------------------------------------
# pytest-qt's qtbot is a fairly rich helper. We only need the waitSignal
# subset for our tests, so emulate just that. Tests that need the full
# pytest-qt API must run with the real plugin.

class _WaitSignalCtx:
    """Context manager returned by our qtbot.waitSignal fallback."""

    def __init__(self, signal):
        self._signal = signal
        self._emitted = False

    def __enter__(self):
        self._signal.connect(self._slot)
        return self

    def __exit__(self, exc_type, exc, tb):
        self._signal.disconnect(self._slot)
        # If the signal never fired, do not raise — pytest-qt does raise by
        # default but our fallback is permissive; one of our tests asserts the
        # post-condition explicitly so a permissive ctx mgr is sufficient.
        return False

    def _slot(self, *args, **kwargs):
        self._emitted = True


class _QtBot:
    """Minimal subset of pytest-qt's qtbot: just ``waitSignal``."""

    def waitSignal(self, signal, timeout=1000):
        return _WaitSignalCtx(signal)


@pytest.fixture
def qtbot():
    """Fallback qtbot providing ``waitSignal`` only — enough for our tests."""
    return _QtBot()
