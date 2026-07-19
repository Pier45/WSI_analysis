"""Reusable Qt widgets used across the Datacleaning tabs."""

from __future__ import annotations

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QFrame


class HorizontalLine(QFrame):
    """Decorative horizontal separator."""

    def __init__(self) -> None:
        super().__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)


class VerticalLine(QFrame):
    """Decorative vertical separator."""

    def __init__(self) -> None:
        super().__init__()
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)


class MatplotlibCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for rendering inline histograms."""

    def __init__(
        self,
        title: str,
        parent=None,
        width: int = 10,
        height: int = 8,
        dpi: int = 100,
    ) -> None:
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.set_facecolor("#323232")
        self.axes.set_title(title)
        self.axes.set_xlim(0, 1)
        super().__init__(fig)
