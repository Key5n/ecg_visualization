from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ecg_visualization.datasets.dataset import ECGEntity
from ecg_visualization.logging import configure_root_logging
from ecg_visualization.scripts.sddb_concat.constants import (
    SEGMENT_COLORS,
    VISUALIZE_OUTPUT_DIR,
)
from ecg_visualization.scripts.sddb_concat.utils import (
    ConcatenatedSequence,
    iter_concatenated_sequences,
)
from ecg_visualization.visualization.styles import apply_default_style

LOGGER = logging.getLogger(__name__)


def sddb_concat_visualize() -> None:
    configure_root_logging()
    apply_default_style()
    VISUALIZE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    processed = 0
    for entity, concat in iter_concatenated_sequences():
        fig = _plot_concatenated_signal(entity, concat)
        output_path = VISUALIZE_OUTPUT_DIR / f"{entity.entity_id}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        LOGGER.info("Saved concatenated signal to %s", output_path)
        processed += 1

    LOGGER.info(
        "Finished visualization. processed=%d output_dir=%s",
        processed,
        VISUALIZE_OUTPUT_DIR,
    )


def _plot_concatenated_signal(
    entity: ECGEntity,
    concat: ConcatenatedSequence,
) -> plt.Figure:
    samples = np.asarray(concat.samples, dtype=float)
    sr = float(concat.sampling_rate_hz)
    ts = np.arange(samples.size, dtype=float) / sr

    signal_min = float(np.nanmin(samples))
    signal_max = float(np.nanmax(samples))
    signal_margin = (signal_max - signal_min) * 0.05 or 1.0
    signal_ylim = (signal_min - signal_margin, signal_max + signal_margin)

    fig, ax = plt.subplots(1, 1, figsize=(8, 2.5))
    ax.plot(ts, samples, "-", linewidth=0.8)
    ax.set_title(f"ID: {entity.entity_id}")
    ax.set_ylabel("Voltage [mV]")
    ax.set_xlabel("Time (sec)")
    ax.set_ylim(*signal_ylim)
    _highlight_concat_segments(ax, concat, ylim_upper=signal_ylim[1])
    fig.tight_layout()
    return fig


def _highlight_segments(
    ax: Axes,
    segments: list[tuple[str, float, float]],
    *,
    window_start: float,
    window_end: float,
    ylim_upper: float,
) -> None:
    for name, start_sec, end_sec in segments:
        if end_sec <= window_start or start_sec >= window_end:
            continue

        highlight_start = max(start_sec, window_start)
        highlight_end = min(end_sec, window_end)
        color = SEGMENT_COLORS.get(name, "#adb5bd")
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
    ylim_upper: float,
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
        ylim_upper=ylim_upper,
    )
