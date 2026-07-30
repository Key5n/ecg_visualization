from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.tasks.rhythm_event_sequences.config import (
    RhythmEventSequencesConfig,
)
from ecg_visualization.tasks.rhythm_event_sequences.utils import ConcatenatedSequence
from ecg_visualization.visualization.export import pdf_exporter
from ecg_visualization.visualization.layouts import paginate_signals
from ecg_visualization.visualization.plotters import plot_beats, plot_symbols

if TYPE_CHECKING:
    from ecg_visualization.tasks.rhythm_event_sequences.score.helpers import ScoreResult


def _plot_concat_scores(
    entity: ECGEntity,
    concat: ConcatenatedSequence,
    score_result: ScoreResult,
    *,
    config: RhythmEventSequencesConfig,
) -> plt.Figure:
    samples = np.asarray(concat.samples, dtype=float)
    sampling_rate_hz = float(concat.sampling_rate_hz)
    ts = np.arange(samples.size, dtype=float) / sampling_rate_hz

    signal_min = float(np.nanmin(samples))
    signal_max = float(np.nanmax(samples))
    signal_margin = (signal_max - signal_min) * 0.05 or 1.0
    signal_ylim = (signal_min - signal_margin, signal_max + signal_margin)

    scores = np.asarray(score_result.scores, dtype=float)

    fig, (signal_ax, score_ax) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(15, 6),
    )

    signal_ax.plot(ts, samples, "-", linewidth=0.8)
    signal_ax.set_ylabel("Voltage [mV]")
    signal_ax.set_ylim(*signal_ylim)

    score_ax.plot(
        score_result.times_sec,
        scores,
        color="black",
        linewidth=1.0,
    )
    score_ax.set_ylabel("Score")
    score_ax.set_xlabel("Time (sec)")
    score_ax.set_yscale("log")

    _highlight_concat_segments(
        signal_ax,
        concat,
        segment_colors=config.segment_colors,
        segment_labels=config.segment_labels,
    )
    _highlight_concat_segments(
        score_ax,
        concat,
        segment_colors=config.segment_colors,
        segment_labels=config.segment_labels,
    )

    signal_ax.set_title(f"ID: {entity.entity_id}")
    fig.tight_layout()
    return fig


def export_concat_scores_pdf(
    entity: ECGEntity,
    concat: ConcatenatedSequence,
    score_result: ScoreResult,
    *,
    output_path: str,
    config: RhythmEventSequencesConfig,
) -> None:
    """Export the score overview and paginated concatenated-sequence details."""
    with pdf_exporter(output_path) as exporter:
        overview = _plot_concat_scores(entity, concat, score_result, config=config)
        exporter.add_page(overview)
        plt.close(overview)

        ts_paged = paginate_signals(
            len(concat.samples),
            int(concat.sampling_rate_hz),
            config.pagination,
        )
        for page_idx, ts_rows in enumerate(ts_paged):
            fig = _plot_concat_score_page(
                entity,
                concat,
                score_result,
                ts_rows=ts_rows,
                page_idx=page_idx,
                config=config,
            )
            exporter.add_page(fig, pad_inches=0)
            plt.close(fig)


def _plot_concat_score_page(
    entity: ECGEntity,
    concat: ConcatenatedSequence,
    score_result: ScoreResult,
    *,
    ts_rows: np.ndarray,
    page_idx: int,
    config: RhythmEventSequencesConfig,
) -> Figure:
    samples = np.asarray(concat.samples, dtype=float)
    sampling_rate_hz = float(concat.sampling_rate_hz)
    scores = np.asarray(score_result.scores, dtype=float)
    beat_times_sec = np.asarray(concat.beats, dtype=float) / sampling_rate_hz
    symbol_times_sec = np.asarray(concat.symbol_samples, dtype=float) / sampling_rate_hz

    signal_min = float(np.nanmin(samples))
    signal_max = float(np.nanmax(samples))
    signal_margin = (signal_max - signal_min) * 0.05 or 1.0
    positive_scores = scores[np.isfinite(scores) & (scores > 0)]
    score_min = float(np.min(positive_scores))
    score_max = float(np.max(positive_scores))
    score_margin = (score_max - score_min) * 0.05 or score_min * 0.05
    score_ylim = (
        max(score_min - score_margin, score_min * 0.95),
        score_max + score_margin,
    )

    fig, axes = plt.subplots(
        nrows=len(ts_rows),
        sharey=True,
        figsize=(8.27, 11.69),
        squeeze=False,
    )
    signal_axes = axes[:, 0]
    score_axes = [signal_ax.twinx() for signal_ax in signal_axes]

    for row_idx, ts in enumerate(ts_rows):
        signal_ax = signal_axes[row_idx]
        score_ax = score_axes[row_idx]
        window_start = float(ts[0])
        window_end = float(ts[-1])
        start_idx = int(np.floor(window_start * sampling_rate_hz))
        end_idx = min(
            int(np.floor(window_end * sampling_rate_hz)) + 1,
            samples.size,
        )
        row_samples = samples[start_idx:end_idx]
        row_ts = ts[: row_samples.size]

        signal_ax.plot(row_ts, row_samples, "-", linewidth=0.7)
        signal_ax.set_ylabel("mV")
        signal_ax.set_ylim(
            signal_min - signal_margin,
            signal_max + signal_margin,
        )
        beat_times_in_window = beat_times_sec[
            (beat_times_sec >= window_start) & (beat_times_sec <= window_end)
        ]
        plot_beats(
            signal_ax,
            beat_times_in_window.tolist(),
            ylim_lower=signal_min - signal_margin,
        )
        symbol_events_in_window = [
            (symbol_time, symbol)
            for symbol_time, symbol in zip(
                symbol_times_sec,
                concat.symbol_values,
                strict=True,
            )
            if window_start <= symbol_time <= window_end
        ]
        plot_symbols(
            signal_ax,
            symbol_events_in_window,
            ylim_lower=signal_min - signal_margin,
        )

        score_mask = (score_result.times_sec >= window_start) & (
            score_result.times_sec <= window_end
        )
        score_ax.plot(
            score_result.times_sec[score_mask],
            scores[score_mask],
            color="black",
            linewidth=0.9,
        )
        score_ax.set_ylabel("Score")
        score_ax.set_yscale("log")
        score_ax.set_ylim(*score_ylim)

        signal_ax.set_xlim(window_start, window_end)
        signal_ax.set_xlabel("Time (sec)")
        _highlight_concat_segments(
            signal_ax,
            concat,
            segment_colors=config.segment_colors,
            segment_labels=config.segment_labels,
            window_start=window_start,
            window_end=window_end,
        )

    fig.suptitle(
        f"{entity.dataset.name}: {entity.entity_id} — concatenated sequence scores "
        f"(detail page {page_idx + 1})"
    )
    fig.subplots_adjust(left=0.1, right=0.88, bottom=0.05, top=0.95, hspace=0.45)
    return fig


def _highlight_segments(
    ax: Axes,
    segments: list[tuple[str, float, float]],
    *,
    window_start: float,
    window_end: float,
    segment_colors: dict[str, str],
    segment_labels: dict[str, str],
) -> None:
    ylim_upper = ax.get_ylim()[1]

    for name, start_sec, end_sec in segments:
        if end_sec <= window_start or start_sec >= window_end:
            continue

        highlight_start = max(start_sec, window_start)
        highlight_end = min(end_sec, window_end)
        color = segment_colors.get(name, "#adb5bd")
        ax.axvspan(highlight_start, highlight_end, color=color, alpha=0.15)

        midpoint = (start_sec + end_sec) / 2
        if window_start <= midpoint <= window_end:
            ax.text(
                midpoint,
                ylim_upper,
                segment_labels.get(name, name),
                fontsize=6,
                horizontalalignment="center",
                verticalalignment="bottom",
                color=color,
            )


def _highlight_concat_segments(
    ax: Axes,
    concat: ConcatenatedSequence,
    *,
    segment_colors: dict[str, str],
    segment_labels: dict[str, str],
    window_start: float | None = None,
    window_end: float | None = None,
) -> None:
    segments: list[tuple[str, float, float]] = []
    running_start = 0
    for name, window in (
        ("sinus_train", concat.segments_info.train),
        ("pre_ar", concat.segments_info.pre_ar),
        ("ar", concat.segments_info.ar),
        ("sinus_test", concat.segments_info.test),
    ):
        if window is None:
            continue
        segment_samples = int(
            np.round((window.end_sec - window.start_sec) * concat.sampling_rate_hz)
        )
        start_sec = running_start / concat.sampling_rate_hz
        end_sec = (running_start + segment_samples) / concat.sampling_rate_hz
        segments.append((name, start_sec, end_sec))
        running_start += segment_samples

    _highlight_segments(
        ax,
        segments,
        window_start=segments[0][1] if window_start is None else window_start,
        window_end=segments[-1][2] if window_end is None else window_end,
        segment_colors=segment_colors,
        segment_labels=segment_labels,
    )
