from __future__ import annotations

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.visualization.plotters import (
    plot_aux_notes,
    plot_beats,
    plot_signal,
    plot_symbols,
)


def render_signal_row(
    *,
    ax: Axes,
    ts: npt.NDArray[np.float64],
    entity: ECGEntity,
    signal_ylim: tuple[float, float],
) -> None:
    window_start, window_end = float(ts[0]), float(ts[-1])
    sampling_rate_hz = float(entity.dataset.sampling_rate_hz)

    start_idx = int(np.floor(window_start * sampling_rate_hz))
    end_idx = min(int(np.floor(window_end * sampling_rate_hz)) + 1, entity.signals.size)
    signal_values = _align_signal_to_window(ts, entity.signals[start_idx:end_idx])

    plot_signal(
        ax,
        ts,
        signal_values,
        ylim_lower=signal_ylim[0],
        ylim_upper=signal_ylim[1],
        label="Voltage [mV]",
    )

    beat_times = np.asarray(entity.beats, dtype=np.float64) / sampling_rate_hz
    beat_times_in_window = beat_times[
        (beat_times >= window_start) & (beat_times <= window_end)
    ]
    plot_beats(
        ax,
        beat_times_in_window.tolist(),
        ylim_lower=signal_ylim[0],
    )

    symbol_times = (
        np.asarray(entity.annotation.sample, dtype=np.float64) / sampling_rate_hz
    )
    symbol_events = [
        (sample_time, symbol)
        for sample_time, symbol in zip(
            symbol_times,
            entity.annotation.symbol,
            strict=True,
        )
        if window_start <= sample_time <= window_end
    ]
    plot_symbols(
        ax,
        symbol_events,
        ylim_lower=signal_ylim[0],
    )

    aux_note_events = [
        (sample_time, note)
        for sample_time, note in zip(
            symbol_times,
            entity.aux_notes,
            strict=False,
        )
        if window_start <= sample_time <= window_end
    ]
    plot_aux_notes(
        ax,
        aux_note_events,
        ylim_upper=signal_ylim[1],
    )


def _align_signal_to_window(
    ts: npt.NDArray[np.float64],
    signal_values: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    if signal_values.size == ts.size:
        return signal_values

    padded = np.full(ts.shape, np.nan, dtype=float)
    limit = min(signal_values.size, ts.size)
    padded[:limit] = signal_values[:limit]
    return padded


def decorate_signal_page(
    *,
    fig: Figure,
    entity: ECGEntity,
    page_idx: int,
) -> None:
    if page_idx == 0:
        fig.suptitle(f"{entity.dataset.name}: {entity.entity_id}")
    fig.supxlabel("Time (sec)")
    fig.subplots_adjust(left=0.08, right=0.94, bottom=0.05, top=0.95)
