from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.tasks.rhythm_event_sequences.config import (
    RhythmEventSequencesConfig,
)
from ecg_visualization.tasks.rhythm_event_sequences.utils import ConcatenatedSequence

if TYPE_CHECKING:
    from ecg_visualization.tasks.rhythm_event_sequences.score.score import ScoreResult


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
    )
    _highlight_concat_segments(
        score_ax,
        concat,
        segment_colors=config.segment_colors,
    )

    signal_ax.set_title(f"ID: {entity.entity_id}")
    fig.tight_layout()
    return fig


def _highlight_segments(
    ax: Axes,
    segments: list[tuple[str, float, float]],
    *,
    window_start: float,
    window_end: float,
    segment_colors: dict[str, str],
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
                name,
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
) -> None:
    segments: list[tuple[str, float, float]] = []
    running_start = 0
    for name, window in (
        ("sinus_train", concat.segments_info.train),
        ("pre_vf", concat.segments_info.pre_vf),
        ("vf", concat.segments_info.vf),
        ("sinus_test", concat.segments_info.test),
    ):
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
        window_start=segments[0][1],
        window_end=segments[-1][2],
        segment_colors=segment_colors,
    )
