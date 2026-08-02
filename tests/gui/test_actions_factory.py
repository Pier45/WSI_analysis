"""GUI tests for ``ui_pyqt5.ImageViewer._make_action`` (the Qt action factory
introduced to resolve a Pylance ``reportCallIssue`` from mixing
``QAction(QIcon, str, parent, ...)`` positional overloads).

These require:
* PyQt5 installed (it's a runtime dep — present on every dev box).
* An offscreen Qt platform — set ``QT_QPA_PLATFORM=offscreen`` in CI.

Imported indirectly via ui_pyqt5, which transitively imports the heavy
TF stack. We guard that import so a missing wheel skips the test instead of
erroring the collection.
"""

from __future__ import annotations

import os

import pytest

# Set the offscreen Qt platform *before* importing any PyQt5 module so the
# QApplication can be constructed in a headless CI runner. Setting it here
# (at module import) is the right place; doing it in a fixture would be too
# late since pytest-qt constructs the QApplication on first use of the
# ``qapp`` / ``qtbot`` fixtures.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.gui

# Skip the whole module if PyQt5 isn't importable (rare in this repo since
# it's a runtime dep, but cheap insurance against broken local envs).
pytest.importorskip("PyQt5")


def _import_viewer():
    """Import ImageViewer lazily so the heavy gui.analyzer import gate happens
    at test call time, not at collection time (so ``pytest --collect-only``
    works even if a transitive dep is missing)."""
    from gui.analyzer import ImageViewer  # noqa: WPS433  (local import on purpose)
    return ImageViewer


@pytest.fixture
def viewer(qapp):
    """Instantiate ImageViewer with an offscreen QApplication.

    The empty-arg constructor works because ``__init__`` only builds the UI
    pieces, doesn't open a file dialog (those are triggered by user action,
    not constructor side-effects). ``qapp`` is provided by pytest-qt.
    """
    ImageViewer = _import_viewer()
    v = ImageViewer()
    # Don't show() — keeps the test headless-friendly.
    yield v
    v.close()


class TestMakeActionContract:
    """``make_action`` must produce a ``QAction`` whose properties equal the
    keyword args we passed in. This is the entire point of the refactor; a
    regression here breaks every menu/toolbar item."""

    def test_text_is_set(self, viewer):
        from PyQt5.QtWidgets import QAction  # local import keeps module import light

        from gui.analyzer.actions import make_action
        act = make_action(viewer, "&Print…")
        assert isinstance(act, QAction)
        # QAction.text() returns the mnemonic stripped of '&' markers.
        assert "Print" in act.text()

    def test_default_enabled_is_true(self, viewer):
        from gui.analyzer.actions import make_action
        act = make_action(viewer, "Foo")
        assert act.isEnabled() is True

    def test_enabled_false_propagates(self, viewer):
        from gui.analyzer.actions import make_action
        act = make_action(viewer, "Foo", enabled=False)
        assert act.isEnabled() is False

    def test_checkable_and_checked(self, viewer):
        from gui.analyzer.actions import make_action
        act = make_action(viewer, "Foo", checkable=True, checked=True)
        assert act.isCheckable() is True
        assert act.isChecked() is True

    def test_checkable_false_with_checked_true_is_a_no_op(self, viewer):
        """If ``checkable=False`` but ``checked=True``, Qt silently ignores the
        ``checked`` flag (the action is not togglable). Lock that behaviour
        so a refactor doesn't accidentally raise instead."""
        from gui.analyzer.actions import make_action
        act = make_action(viewer, "Foo", checkable=False, checked=True)
        assert act.isCheckable() is False
        # isChecked() on a non-checkable action returns False.
        assert act.isChecked() is False

    def test_shortcut_is_set(self, viewer):
        from PyQt5.QtGui import QKeySequence

        from gui.analyzer.actions import make_action
        act = make_action(viewer, "Foo", shortcut="Ctrl+P")
        assert act.shortcut().toString() == "Ctrl+P"

    def test_no_shortcut_when_empty(self, viewer):
        from gui.analyzer.actions import make_action
        act = make_action(viewer, "Foo", shortcut="")
        assert act.shortcut().toString() == ""

    def test_icon_set_when_path_provided(self, viewer, tmp_path):
        """We need a real on-disk image for QIcon to load; PyQt5 silently
        produces an empty icon if the path is bad, so we write a tiny PNG."""
        from PyQt5.QtGui import QImage

        from gui.analyzer.actions import make_action

        img_path = tmp_path / "icon.png"
        # 16x16 magenta PNG.
        qi = QImage(16, 16, QImage.Format_RGB32)
        qi.fill(0xFFFF00FF)
        qi.save(str(img_path))

        act = make_action(viewer, "Foo", icon=str(img_path))
        assert not act.icon().isNull()

    def test_icon_omitted_is_null_icon(self, viewer):
        from gui.analyzer.actions import make_action
        act = make_action(viewer, "Foo")
        assert act.icon().isNull()

    def test_triggered_handler_is_connected(self, viewer, qtbot):
        """The ``triggered`` callback must actually fire when the action is
        triggered — otherwise the menu item silently does nothing. This is
        the single most important property of the factory and the part the
        Pylance overload bug actually broke if the connection was dropped."""
        from gui.analyzer.actions import make_action
        calls = []
        act = make_action(viewer, "Foo", triggered=lambda: calls.append(1))
        with qtbot.waitSignal(act.triggered):
            act.trigger()
        assert calls == [1]

    def test_triggered_omitted_does_not_raise_on_emit(self, viewer, qtbot):
        """If no ``triggered`` is passed we must NOT have connected anything,
        so emitting the signal must be a no-op, not a TypeError."""
        from gui.analyzer.actions import make_action
        act = make_action(viewer, "Foo")
        with qtbot.waitSignal(act.triggered):
            act.trigger()
        # Reaching here is the assertion.


class TestCreateActionsUsesFactory:
    """``_create_actions`` should populate every expected attribute via
    ``_make_action``. Locking the public attribute set guards against typos
    that would later AttributeError deep in a menu setup."""

    EXPECTED_ATTRS = (
        "_open_act", "_print_act", "_exit_act",
        "_zoom_in_act", "_zoom_out_act", "_normal_size_act", "_fit_to_window_act",
        "_start_analysis_act", "_fast_act", "_slow_act",
        "_select_model_act", "_mc5_act", "_mc25_act", "_mc50_act",
        "_v_no_overlay_act", "_v_all_classes_act",
        "_v_ac_act", "_v_ad_act", "_v_h_act",
        "_v_total_uncertainty_act", "_v_aleatoric_act", "_v_epistemic_act",
        "_deep_zoom_act", "_about_act", "_info_deep_act",
    )

    @pytest.mark.parametrize("attr", EXPECTED_ATTRS)
    def test_action_attribute_exists(self, viewer, attr):
        assert hasattr(viewer, attr), (
            f"ImageViewer.{attr} not set — _create_actions incomplete?"
        )
        from PyQt5.QtWidgets import QAction
        assert isinstance(getattr(viewer, attr), QAction)
