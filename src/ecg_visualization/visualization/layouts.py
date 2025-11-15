import numpy as np
from numpy.typing import NDArray
import math
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def create_page_layout(
    n_rows: int,
    *,
    figsize: tuple[float, float] = (8.27, 11.69),
) -> tuple[Figure, NDArray]:
    """Create a grid layout for a single ECG page."""
    fig, axs = plt.subplots(
        nrows=n_rows,
        figsize=figsize,
    )
    return fig, axs


@dataclass(frozen=True)
class PaginationConfig:
    """Configuration for paging ECG signals into multi-row PDF pages."""

    seconds_per_row: int = 10
    rows_per_page: int = 6

    def compute(
        self,
        sampling_rate: int,
        total_samples: int,
    ) -> tuple[int, int, int]:
        """Return (steps_per_row, rows_per_page, total_pages)."""
        steps_per_row = sampling_rate * self.seconds_per_row
        samples_per_page = steps_per_row * self.rows_per_page
        n_pages = math.ceil(total_samples / samples_per_page) if samples_per_page else 0
        return steps_per_row, self.rows_per_page, n_pages


def paginate_signals(
    n_signals: int,
    sampling_rate: int,
    layout_config: PaginationConfig,
) -> NDArray[np.float64]:
    """Compute time axes shaped into (pages, rows, steps) for ECG pagination."""

    n_steps, n_rows, n_pages = layout_config.compute(sampling_rate, n_signals)
    shape = (n_pages, n_rows, n_steps)
    total_steps = math.prod(shape)

    if total_steps == 0:
        return np.zeros(shape, dtype=float)

    ts = np.arange(total_steps, dtype=float) / sampling_rate
    return ts.reshape(shape)
