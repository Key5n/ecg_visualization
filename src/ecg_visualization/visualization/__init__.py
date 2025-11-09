"""Visualization utilities split into styles, plotters, layouts, and export helpers."""

from .styles import (
    apply_default_style,
    ABNORMAL_INTERVAL_COLOR,
    EXTREME_INTERVAL_COLOR,
)
from .plotters import (
    highlight_windows,
    plot_signal,
    plot_symbols,
)
from .layouts import create_page_layout
from .limits import compute_signal_ylim
from .export import pdf_exporter, PdfExporter

__all__ = [
    "apply_default_style",
    "ABNORMAL_INTERVAL_COLOR",
    "EXTREME_INTERVAL_COLOR",
    "plot_signal",
    "plot_symbols",
    "highlight_windows",
    "create_page_layout",
    "compute_signal_ylim",
    "pdf_exporter",
    "PdfExporter",
]
