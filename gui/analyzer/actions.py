"""Qt action factory + bulk action construction for the Bayesian Analyzer.

The test contract pinned in ``tests/gui/test_actions_factory.py``:

- :func:`make_action` produces a ``QAction`` whose properties match the
  keyword args we passed in (text, enabled, checkable, checked,
  shortcut, icon, ``triggered``).
- :func:`create_actions` populates the 25 ``_*_act`` attributes on the
  ``ImageViewer`` instance. The test asserts each one exists and is a
  ``QAction`` — names must not be renamed without updating the test.

Hoisting these out of ``ImageViewer`` keeps the action-construction
contract test-importable without instantiating a ``QMainWindow`` for a
pure ``QAction``-shape test, and lets the menus module import the
actions without a circular ``ImageViewer`` import.
"""

from __future__ import annotations

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction, QMainWindow

from .constants import MONTE_CARLO_OPTIONS


def make_action(
    parent: QMainWindow,
    text: str,
    *,
    icon: str | None = None,
    shortcut: str = "",
    enabled: bool = True,
    checkable: bool = False,
    checked: bool = False,
    triggered=None,
) -> QAction:
    """Factory used by every menu/toolbar action in the Analyzer.

    Wrapping ``QAction`` construction in this helper resolves a Pylance
    ``reportCallIssue`` from mixing ``QAction(QIcon, str, parent, ...)``
    positional overloads; the test must verify the callback is actually
    connected so the menu/toolbar item isn't silently dead.
    """
    action = QAction(text, parent)
    if icon is not None:
        action.setIcon(QIcon(icon))
    if shortcut:
        action.setShortcut(shortcut)
    action.setEnabled(enabled)
    action.setCheckable(checkable)
    action.setChecked(checked)
    if triggered is not None:
        action.triggered.connect(triggered)
    return action


def create_actions(parent: QMainWindow) -> None:
    """Populate every ``_*_act`` attribute on *parent*.

    Called from ``ImageViewer.__init__``. The attribute names are part
    of the public contract with ``tests/gui/test_actions_factory.py`` —
    do not rename without updating the test's ``EXPECTED_ATTRS`` tuple.
    """
    # File
    parent._open_act = make_action(
        parent, "Select SVS", icon="icons/folder.png",
        shortcut="Ctrl+O", triggered=parent._open_file,
    )
    parent._print_act = make_action(
        parent, "&Print…", shortcut="Ctrl+P", enabled=False, triggered=parent._print_image,
    )
    parent._exit_act = make_action(
        parent, "E&xit", icon="icons/exit.ico",
        shortcut="Ctrl+Q", triggered=parent.close,
    )

    # Zoom
    parent._zoom_in_act = make_action(
        parent, "Zoom &In (25%)", icon="icons/zoomin.ico",
        shortcut="Ctrl++", enabled=False, triggered=parent._zoom_in,
    )
    parent._zoom_out_act = make_action(
        parent, "Zoom &Out (25%)", icon="icons/zoomout.ico",
        shortcut="Ctrl+-", enabled=False, triggered=parent._zoom_out,
    )
    parent._normal_size_act = make_action(
        parent, "&Normal Size", shortcut="Ctrl+N", enabled=False, triggered=parent._normal_size,
    )
    parent._fit_to_window_act = make_action(
        parent, "&Fit to Window", shortcut="Ctrl+F",
        enabled=False, checkable=True, triggered=parent._fit_to_window,
    )

    # Analysis
    parent._start_analysis_act = make_action(
        parent, "Start Analysis", icon="icons/start.ico",
        shortcut="Ctrl+R", enabled=False, triggered=parent._start_analysis,
    )
    parent._fast_act = make_action(
        parent, "Fast mode", checkable=True, checked=True, triggered=parent._set_fast_mode,
    )
    parent._slow_act = make_action(
        parent, "Slow mode", checkable=True, triggered=parent._set_slow_mode,
    )

    # Model / Monte Carlo
    parent._select_model_act = make_action(
        parent, "Change model", triggered=parent._select_model,
    )
    parent._mc5_act = make_action(
        parent, "5", checkable=True, checked=True,
        triggered=lambda: parent._set_monte_carlo(5),
    )
    parent._mc25_act = make_action(
        parent, "25", checkable=True, triggered=lambda: parent._set_monte_carlo(25),
    )
    parent._mc50_act = make_action(
        parent, "50", checkable=True, triggered=lambda: parent._set_monte_carlo(50),
    )

    # View results
    parent._v_no_overlay_act = make_action(
        parent, "No overlay", enabled=False,
        triggered=lambda: parent._view_result("no_ov", "th"),
    )
    parent._v_all_classes_act = make_action(
        parent, "All classes", enabled=False,
        triggered=lambda: parent._view_result("Pred_class", "result"),
    )
    parent._v_ac_act = make_action(
        parent, "AC only", icon="icons/AC.png", enabled=False,
        triggered=lambda: parent._view_result("AC", "result"),
    )
    parent._v_ad_act = make_action(
        parent, "AD only", icon="icons/AD.png", enabled=False,
        triggered=lambda: parent._view_result("AD", "result"),
    )
    parent._v_h_act = make_action(
        parent, "H only", icon="icons/H.png", enabled=False,
        triggered=lambda: parent._view_result("H", "result"),
    )
    parent._v_total_uncertainty_act = make_action(
        parent, "Total uncertainty", enabled=False,
        triggered=lambda: parent._view_result("tot", "uncertainty"),
    )
    parent._v_aleatoric_act = make_action(
        parent, "Aleatoric uncertainty", enabled=False,
        triggered=lambda: parent._view_result("ale", "uncertainty"),
    )
    parent._v_epistemic_act = make_action(
        parent, "Epistemic uncertainty", enabled=False,
        triggered=lambda: parent._view_result("epi", "uncertainty"),
    )

    # Deep zoom / help
    parent._deep_zoom_act = make_action(
        parent, "Deep Zoom Viewer", icon="icons/binocul.ico",
        shortcut="Ctrl+D", enabled=False, triggered=parent._open_deep_zoom,
    )
    parent._about_act = make_action(parent, "&About", triggered=parent._about)
    parent._info_deep_act = make_action(
        parent, "&Deepzoom info", triggered=parent._about_deep_zoom,
    )


def set_monte_carlo(parent: QMainWindow, value: int) -> None:
    """Check the right MC action; uncheck the rest."""
    for action, mc_value in zip(
        (parent._mc5_act, parent._mc25_act, parent._mc50_act),
        MONTE_CARLO_OPTIONS,
        strict=True,
    ):
        action.setChecked(mc_value == value)
