from typing import Iterable, Sequence

import numpy as np
from matplotlib.axes import Axes

from .styles import ABNORMAL_INTERVAL_COLOR


def plot_signal(
    ax: Axes,
    ts: np.ndarray,
    signal: np.ndarray,
    *,
    ylim_lower: float,
    ylim_upper: float,
    beat_times: Sequence[float],
    label: str = "mν",
) -> None:
    """Plot the ECG signal with baseline styles and beat annotations."""
    ax.plot(ts, signal, "-")
    ax.set_ylim(ylim_lower, ylim_upper)
    ax.set_ylabel(label)
    ax.set_xlim(ts[0], ts[-1])

    for beat_time in beat_times:
        ax.text(
            beat_time,
            ylim_lower,
            "N",
            fontsize=4,
            horizontalalignment="center",
        )


def plot_symbols(
    ax: Axes,
    symbol_events: Iterable[tuple[float, str]],
    *,
    window_start: float,
    window_end: float,
    ylim_lower: float,
) -> None:
    """Mark abnormal rhythm symbols within the current axis window."""
    for sample_time, symbol in symbol_events:
        if symbol == "N" or sample_time < window_start or sample_time > window_end:
            continue
        ax.axvline(sample_time, color="red", alpha=0.5)
        ax.text(
            sample_time,
            ylim_lower,
            symbol,
            fontsize=8,
            horizontalalignment="center",
            c="red",
        )


def highlight_windows(
    ax: Axes,
    abnormal_windows: Iterable[tuple[float, float]],
    *,
    window_start: float,
    window_end: float,
    ylim_upper: float,
    color: str = ABNORMAL_INTERVAL_COLOR,
) -> None:
    """Highlight abnormal RR windows that overlap the current axis limits."""
    for window_start_global, window_end_global in sorted(abnormal_windows):
        if window_end_global <= window_start or window_start_global >= window_end:
            continue
        highlight_start = max(window_start_global, window_start)
        highlight_end = min(window_end_global, window_end)

        if highlight_end > highlight_start:
            ax.axvspan(
                highlight_start,
                highlight_end,
                color=color,
                alpha=0.2,
            )

        window_half_point = (window_start_global + window_end_global) / 2
        if window_start <= window_half_point <= window_end:
            ax.text(
                (highlight_start + highlight_end) / 2,
                ylim_upper,
                f"From {window_start_global:.2f}s to {window_end_global:.2f}s",
                fontsize=6,
                horizontalalignment="center",
                verticalalignment="bottom",
                c=color,
            )
