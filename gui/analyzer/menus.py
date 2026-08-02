"""Menu bar + toolbar construction for the Bayesian Analyzer.

Pure UI wiring — no slots defined here, only the actions set up by
:func:`gui.analyzer.actions.create_actions` are arranged into the
menu bar (File / Analysis / View / Options / Help) and the main
toolbar.
"""

from __future__ import annotations

from PyQt5.QtWidgets import QMainWindow, QMenu, QToolBar


def create_menus(parent: QMainWindow) -> None:
    """Build the five top-level menus and attach them to ``parent.menuBar()``."""
    # File menu
    file_menu = QMenu("&File", parent)
    file_menu.addAction(parent._open_act)
    file_menu.addAction(parent._print_act)
    file_menu.addSeparator()
    file_menu.addAction(parent._exit_act)

    # Analysis menu
    analysis_menu = QMenu("&Analysis", parent)
    analysis_menu.addAction(parent._fast_act)
    analysis_menu.addAction(parent._slow_act)
    analysis_menu.addSeparator()

    settings_menu = QMenu("&Settings", parent)
    settings_menu.addAction(parent._select_model_act)

    mc_menu = QMenu("&Monte Carlo Samples", parent)
    mc_menu.addAction(parent._mc5_act)
    mc_menu.addAction(parent._mc25_act)
    mc_menu.addAction(parent._mc50_act)
    settings_menu.addMenu(mc_menu)

    analysis_menu.addMenu(settings_menu)
    analysis_menu.addSeparator()
    analysis_menu.addAction(parent._start_analysis_act)

    # View menu
    view_menu = QMenu("&View", parent)
    view_menu.addAction(parent._v_no_overlay_act)
    view_menu.addAction(parent._v_all_classes_act)
    view_menu.addSeparator()
    view_menu.addAction(parent._v_ac_act)
    view_menu.addAction(parent._v_ad_act)
    view_menu.addAction(parent._v_h_act)
    view_menu.addSeparator()
    view_menu.addAction(parent._v_total_uncertainty_act)
    view_menu.addAction(parent._v_aleatoric_act)
    view_menu.addAction(parent._v_epistemic_act)
    view_menu.addSeparator()
    view_menu.addAction(parent._zoom_in_act)
    view_menu.addAction(parent._zoom_out_act)
    view_menu.addAction(parent._normal_size_act)
    view_menu.addSeparator()
    view_menu.addAction(parent._fit_to_window_act)

    # Options menu
    options_menu = QMenu("&Options", parent)
    options_menu.addAction(parent._deep_zoom_act)
    options_menu.addAction(parent._info_deep_act)

    # Help menu
    help_menu = QMenu("&Help", parent)
    help_menu.addAction(parent._about_act)

    for menu in (file_menu, analysis_menu, view_menu, options_menu, help_menu):
        parent.menuBar().addMenu(menu)


def populate_toolbar(toolbar: QToolBar, parent: QMainWindow) -> None:
    """Fill the main toolbar with the action/separator sequence used by the Analyzer."""
    for item in (
        parent._open_act,
        parent._start_analysis_act,
        None,  # separator
        parent._zoom_in_act,
        parent._zoom_out_act,
        None,
        parent._deep_zoom_act,
        None,
        parent._v_ac_act,
        parent._v_ad_act,
        parent._v_h_act,
    ):
        if item is None:
            toolbar.addSeparator()
        else:
            toolbar.addAction(item)
