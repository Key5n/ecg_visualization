from contextlib import contextmanager
from typing import Iterator

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure


class PdfExporter:
    """Helper that wraps PdfPages for multi-page export flows."""

    def __init__(self, pdf_pages: PdfPages) -> None:
        self._pdf_pages = pdf_pages

    def add_page(self, fig: Figure, *, pad_inches: float = 0.0) -> None:
        """Append a figure to the PDF."""
        self._pdf_pages.savefig(fig, pad_inches=pad_inches)


@contextmanager
def pdf_exporter(path: str) -> Iterator[PdfExporter]:
    """Context manager that yields a PdfExporter."""
    with PdfPages(path) as pdf_pages:
        yield PdfExporter(pdf_pages)


def save_png(fig: Figure, path: str, **kwargs: object) -> None:
    """Save a figure as PNG with sane defaults."""
    fig.savefig(path, format="png", **kwargs)


def save_svg(fig: Figure, path: str, **kwargs: object) -> None:
    """Save a figure as SVG with sane defaults."""
    fig.savefig(path, format="svg", **kwargs)
