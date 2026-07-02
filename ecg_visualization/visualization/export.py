import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator, Mapping

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from pypdf import PdfReader, PdfWriter


class PdfExporter:
    """Helper that wraps PdfPages for multi-page export flows."""

    def __init__(self, pdf_pages: PdfPages) -> None:
        self._pdf_pages = pdf_pages

    def add_page(self, fig: Figure, *, pad_inches: float = 0.0) -> None:
        """Append a figure to the PDF."""
        self._pdf_pages.savefig(fig, pad_inches=pad_inches)


@contextmanager
def pdf_exporter(
    path: str,
    *,
    metadata: Mapping[str, str] | None = None,
) -> Iterator[PdfExporter]:
    """Context manager that yields a PdfExporter."""
    with PdfPages(path) as pdf_pages:
        if metadata:
            info_dict = pdf_pages.infodict()
            for key, value in metadata.items():
                info_dict[key] = value
        yield PdfExporter(pdf_pages)
    if metadata:
        _embed_metadata(Path(path), metadata)


def save_png(fig: Figure, path: str, **kwargs: object) -> None:
    """Save a figure as PNG with sane defaults."""
    fig.savefig(path, format="png", **kwargs)


def save_svg(fig: Figure, path: str, **kwargs: object) -> None:
    """Save a figure as SVG with sane defaults."""
    fig.savefig(path, format="svg", **kwargs)


def _embed_metadata(path: Path, metadata: Mapping[str, str]) -> None:
    """Inject arbitrary metadata entries via pypdf."""
    reader = PdfReader(str(path))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)

    merged: dict[str, str] = {}
    if reader.metadata:
        for key, value in reader.metadata.items():
            if isinstance(key, str) and value is not None:
                merged[key] = str(value)
    for key, value in metadata.items():
        merged[f"/{key}"] = str(value)

    with NamedTemporaryFile(
        delete=False, suffix=path.suffix, dir=str(path.parent)
    ) as tmp_file:
        writer.add_metadata(merged)
        writer.write(tmp_file)
        tmp_path = Path(tmp_file.name)
    os.replace(tmp_path, path)
