"""
Shared Qt thread-pool plumbing for the two GUI entry points.

Both ``ui_pyqt5.py`` and ``ui_dataclean.py`` previously shipped their own
near-duplicate ``WorkerSignals`` / ``Worker`` / ``WorkerLong`` /
``LongRunningWorker`` classes. This module unifies them under one set:

    Worker            — fire-and-forget runnable; no signals injected.
    WorkerSignals     — Qt signal set emitted by ``WorkerLong``.
    WorkerLong        — long-running runnable; injects ``progress_callback``
                        (always) and ``view`` (only when the wrapped callable
                        accepts it) into the call.

``LongRunningWorker`` is kept as a backward-compatible alias for ``WorkerLong``
so existing call-sites in ``ui_dataclean.py`` keep working unchanged.

The reason injection of ``view`` is conditional: several callables wrapped by
``ui_pyqt5.py`` (e.g. ``_create_tiles``) only accept ``progress_callback``;
unconditionally injecting ``view=`` would raise ``TypeError`` at call time.
``ProgressInjector`` introspects the callable's signature with
``inspect.signature`` and only forwards kwargs the callable actually accepts.
"""

from __future__ import annotations

import inspect
import sys
import traceback
from typing import Any, Callable, Optional

from PyQt5.QtCore import QObject, QRunnable, pyqtSlot, pyqtSignal


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


class WorkerSignals(QObject):
    """Signals emitted by :class:`WorkerLong` during its lifecycle.

    ``intermediate_result`` is emitted via the ``view`` kwarg when the wrapped
    callable accepts one. UIs that only care about progress can ignore it.
    """

    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)
    intermediate_result = pyqtSignal(object)


# ---------------------------------------------------------------------------
# Callables & injection helpers
# ---------------------------------------------------------------------------


def _accepts_kwarg(fn: Callable, name: str) -> bool:
    """True iff *fn* accepts a keyword argument named *name*.

    Returns True when *fn*'s signature explicitly names *name*, when it has a
    ``**kwargs`` catch-all, or when *fn* is a bound builtin without an
    introspectable signature (in which case we err on the side of forwarding,
    matching the historical behaviour).
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return True

    params = sig.parameters
    if name in params:
        return True
    # VAR_KEYWORD means **kwargs — accept anything
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


class Worker(QRunnable):
    """
    Fire-and-forget thread wrapper.

    Runs *fn* in a thread-pool thread without lifecycle feedback. Use
    :class:`WorkerLong` when progress / error signals are needed.
    """

    def __init__(self, fn: Callable, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    @pyqtSlot()
    def run(self) -> None:
        self._fn(*self._args, **self._kwargs)


class WorkerLong(QRunnable):
    """
    Thread wrapper with full lifecycle signals.

    Injects a ``progress_callback`` keyword argument into *fn* so the worker
    can emit integer progress values (0-100). When *fn* accepts a ``view``
    parameter, the ``intermediate_result`` signal is also forwarded as That
    keyword so the worker can stream log/status strings to the UI.

    Signals
    -------
    signals.finished           — emitted once the function returns or raises
    signals.error               — emitted with (exc_type, value, traceback_str)
    signals.result              — emitted with the function's return value
    signals.progress           — emitted by the function via progress_callback.emit(n)
    signals.intermediate_result — emitted by the function via view.emit(msg), when supported
    """

    def __init__(self, fn: Callable, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = dict(kwargs)
        self.signals = WorkerSignals()

        # Always forward progress_callback; the dataclean model / classification
        # / tiling / thresholding callables all accept it.
        self._kwargs["progress_callback"] = self.signals.progress

        # Only forward view if the wrapped callable actually accepts it.
        # Without this guard, wrapping a fn like _create_tiles(self, tile_args,
        # progress_callback) would raise TypeError("got an unexpected keyword
        # argument 'view'") at call time.
        if _accepts_kwarg(fn, "view"):
            self._kwargs["view"] = self.signals.intermediate_result

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception:
            traceback.print_exc()
            exc_type, value = sys.exc_info()[:2]
            self.signals.error.emit((exc_type, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


# Backward-compatible alias so ``LongRunningWorker``-style call sites continue
# to work after the import-switch.
LongRunningWorker = WorkerLong


__all__ = ["WorkerSignals", "Worker", "WorkerLong", "LongRunningWorker"]
