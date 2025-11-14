from typing import Iterable, Sequence

import numpy as np
from matplotlib.axes import Axes

from .styles import ABNORMAL_INTERVAL_COLOR


def plot_normal_beats(
    ax: Axes,
    beat_times: Sequence[float],
    *,
    ylim_lower: float,
    label: str = "N",
    fontsize: float = 4,
) -> None:
    """Annotate normal beats with textual markers along the lower axis bound."""
    for beat_time in beat_times:
        ax.text(
            beat_time,
            ylim_lower,
            label,
            fontsize=fontsize,
            horizontalalignment="center",
        )


def plot_anomaly_score(
    ax: Axes,
    score_times: Sequence[float],
    scores: Sequence[float],
    *,
    ylim_lower: float,
    ylim_upper: float,
    color: str = "tab:red",
    label: str = "Anomaly score",
    linewidth: float = 1.0,
    alpha: float = 0.8,
) -> Axes | None:
    """Overlay anomaly scores on a twin axis that shares the signal timeline."""
    if not score_times or not scores:
        return None

    ax.plot(
        score_times,
        scores,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
    )
    ax.set_ylabel(label, color=color)
    ax.set_ylim(ylim_lower, ylim_upper)
    ax.tick_params(axis="y", colors=color)
    ax.spines["right"].set_color(color)
    return ax


def plot_signal(
    ax: Axes,
    ts: np.ndarray,
    signal: np.ndarray,
    *,
    ylim_lower: float,
    ylim_upper: float,
    label: str = "mν",
) -> None:
    """Plot the ECG signal with baseline styles."""
    ax.plot(ts, signal, "-")
    ax.set_ylim(ylim_lower, ylim_upper)
    ax.set_ylabel(label)
    ax.set_xlim(ts[0], ts[-1])


def plot_symbols(
    ax: Axes,
    symbol_events: Iterable[tuple[float, str]],
    *,
    ylim_lower: float,
) -> None:
    """Mark abnormal rhythm symbols within the current axis window."""
    for sample_time, symbol in symbol_events:
        if symbol == "N":
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


def plot_histogram(
    ax: Axes,
    values: np.ndarray,
    *,
    bins: int | Sequence[float] | str = "auto",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    color: str = "tab:blue",
    alpha: float = 0.7,
    percentile_lines: Sequence[float] | None = None,
    percentile_color: str = "tab:red",
    percentile_linestyle: str = "--",
) -> None:
    """Render a styled histogram of the provided values."""
    if values.size == 0:
        return

    ax.hist(values, bins=bins, color=color, alpha=alpha)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    if percentile_lines:
        percentile_values = np.percentile(values, percentile_lines)
        ylim_upper = ax.get_ylim()[1]
        y_pos = ylim_upper * 0.95
        for percentile, cutoff in zip(
            percentile_lines, percentile_values, strict=False
        ):
            ax.axvline(
                cutoff,
                color=percentile_color,
                linestyle=percentile_linestyle,
                alpha=0.8,
            )
            ax.text(
                cutoff,
                y_pos,
                f"{percentile:.0f}%: {cutoff:.2f}s",
                rotation=90,
                fontsize=7,
                color=percentile_color,
                horizontalalignment="right",
                verticalalignment="top",
            )
