import numpy as np
from numpy.typing import NDArray
from ecg_visualization.utils.utils import padding_reshape
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
    signals: NDArray[np.float64],
    sampling_rate: int,
    layout_config: PaginationConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int, int, int]:
    """Pad and reshape signals/time arrays into (pages, rows, steps)."""

    n_steps, n_rows, n_pages = layout_config.compute(sampling_rate, len(signals))
    signals_paged = padding_reshape(signals, (n_pages, n_rows, n_steps))

    length = n_pages * n_rows * n_steps
    ts_paged = np.linspace(0, length / sampling_rate, length).reshape(
        (n_pages, n_rows, n_steps)
    )

    return signals_paged, ts_paged, n_steps, n_rows, n_pages
