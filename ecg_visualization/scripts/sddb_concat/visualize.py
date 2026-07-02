from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ecg_visualization.datasets.dataset import ECGEntity
from ecg_visualization.logging.config import configure_root_logging
from ecg_visualization.scripts.sddb_concat.constants import (
    SEGMENT_COLORS,
    SINUS_RR_MEDIAN_THRESHOLD_SEC,
    VISUALIZE_OUTPUT_DIR,
)
from ecg_visualization.scripts.sddb_concat.utils import (
    ConcatenatedSequence,
    iter_concatenated_sequences,
)
from ecg_visualization.visualization.export import PdfExporter, pdf_exporter
from ecg_visualization.visualization.layouts import (
    PaginationConfig,
    create_page_layout,
    paginate_signals,
)
from ecg_visualization.visualization.limits import compute_ylim
from ecg_visualization.visualization.plotters import (
    plot_histogram,
    plot_normal_beats,
    plot_signal,
    plot_symbols,
)
from ecg_visualization.visualization.styles import apply_default_style

LOGGER = logging.getLogger(__name__)

PAGINATION_CONFIG = PaginationConfig(seconds_per_row=10, rows_per_page=6)
RR_HISTOGRAM_XMIN_SEC = 0.0
RR_HISTOGRAM_XMAX_SEC = 2.0
RR_HISTOGRAM_BIN_WIDTH_SEC = 0.025
RR_HISTOGRAM_BINS = np.arange(
    RR_HISTOGRAM_XMIN_SEC,
    RR_HISTOGRAM_XMAX_SEC + RR_HISTOGRAM_BIN_WIDTH_SEC,
    RR_HISTOGRAM_BIN_WIDTH_SEC,
)
SEGMENT_LABELS = {
    "sinus_train": "train",
    "pre_vf": "pre-vf",
    "vf": "vf",
    "sinus_test": "test",
}


def sddb_concat_visualize() -> None:
    configure_root_logging()
    apply_default_style()
    VISUALIZE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    processed = 0
    for entity, concat in iter_concatenated_sequences():
        output_path = VISUALIZE_OUTPUT_DIR / f"{entity.entity_id}.pdf"
        _export_concatenated_pdf(entity, concat, output_path=output_path)
        LOGGER.info("Saved concatenated signal PDF to %s", output_path)
        processed += 1

    LOGGER.info(
        "Finished ecg_visualization.visualization. processed=%d output_dir=%s",
        processed,
        VISUALIZE_OUTPUT_DIR,
    )


def _export_concatenated_pdf(
    entity: ECGEntity,
    concat: ConcatenatedSequence,
    *,
    output_path: Path,
) -> None:
    ts_paged = paginate_signals(
        entity.signals.size,
        int(entity.sr),
        PAGINATION_CONFIG,
    )
    signal_ylim = compute_ylim(
        entity.signals,
        lower_bound=-5.0,
        upper_bound=5.0,
    )

    with pdf_exporter(str(output_path)) as exporter:
        _render_rr_interval_histogram_page(entity, concat, exporter)
        for page_idx, ts_row in enumerate(ts_paged):
            fig, axs = create_page_layout(PAGINATION_CONFIG.rows_per_page)
            for ts, ax in zip(ts_row, np.atleast_1d(axs), strict=True):
                _render_signal_row(
                    ax=ax,
                    ts=ts,
                    entity=entity,
                    concat=concat,
                    signal_ylim=signal_ylim,
                )
            _decorate_page(fig=fig, entity=entity, page_idx=page_idx)
            exporter.add_page(fig, pad_inches=0)
            plt.close(fig)


def _render_rr_interval_histogram_page(
    entity: ECGEntity,
    concat: ConcatenatedSequence,
    exporter: PdfExporter,
) -> None:
    rr_intervals = np.diff(np.asarray(entity.beats, dtype=np.float64))
    rr_intervals_sec = rr_intervals / float(entity.sr)

    fig, ax = plt.subplots(figsize=(8, 4))
    if rr_intervals_sec.size > 0:
        rr_intervals_in_range = rr_intervals_sec[
            (rr_intervals_sec >= RR_HISTOGRAM_XMIN_SEC)
            & (rr_intervals_sec <= RR_HISTOGRAM_XMAX_SEC)
        ]
        if rr_intervals_in_range.size > 0:
            plot_histogram(
                ax,
                rr_intervals_in_range,
                bins=RR_HISTOGRAM_BINS,
                title=f"{entity.dataset_name} / {entity.entity_id} RR intervals",
                xlabel="R-peak interval (sec)",
                ylabel="Count",
            )
        else:
            ax.set_title(f"{entity.dataset_name} / {entity.entity_id} RR intervals")
            ax.set_xlabel("R-peak interval (sec)")
            ax.set_ylabel("Count")
        ax.set_xlim(RR_HISTOGRAM_XMIN_SEC, RR_HISTOGRAM_XMAX_SEC)
        median_rr_interval = float(np.median(rr_intervals_sec))
        sinus_lower = median_rr_interval - SINUS_RR_MEDIAN_THRESHOLD_SEC
        sinus_upper = median_rr_interval + SINUS_RR_MEDIAN_THRESHOLD_SEC
        ax.axvspan(
            sinus_lower,
            sinus_upper,
            color="tab:green",
            alpha=0.15,
        )
        ax.axvline(
            median_rr_interval,
            color="tab:red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
        )
        ylim_upper = ax.get_ylim()[1]
        ax.text(
            median_rr_interval,
            ylim_upper * 0.95,
            f"Median: {median_rr_interval:.2f}s",
            rotation=90,
            fontsize=8,
            color="tab:red",
            horizontalalignment="right",
            verticalalignment="top",
        )
    else:
        ax.set_title(f"{entity.dataset_name} / {entity.entity_id} RR intervals")
        ax.set_xlabel("R-peak interval (sec)")
        ax.set_ylabel("Count")
        ax.text(
            0.5,
            0.5,
            "Not enough R-peaks to compute intervals.",
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
        )

    fig.tight_layout()
    exporter.add_page(fig, pad_inches=0)
    plt.close(fig)


def _render_signal_row(
    *,
    ax: Axes,
    ts: npt.NDArray[np.float64],
    entity: ECGEntity,
    concat: ConcatenatedSequence,
    signal_ylim: tuple[float, float],
) -> None:
    window_start, window_end = float(ts[0]), float(ts[-1])
    sr = float(entity.sr)

    start_idx = int(np.floor(window_start * sr))
    end_idx = min(int(np.floor(window_end * sr)) + 1, entity.signals.size)
    signal_values = _align_signal_to_window(ts, entity.signals[start_idx:end_idx])

    plot_signal(
        ax,
        ts,
        signal_values,
        ylim_lower=signal_ylim[0],
        ylim_upper=signal_ylim[1],
        label="Voltage [mV]",
    )

    beat_times = np.asarray(entity.beats, dtype=np.float64) / sr
    beat_times_in_window = beat_times[
        (beat_times >= window_start) & (beat_times <= window_end)
    ]
    plot_normal_beats(
        ax,
        beat_times_in_window.tolist(),
        ylim_lower=signal_ylim[0],
    )

    symbol_times = np.asarray(entity.annotation.sample, dtype=np.float64) / sr
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

    _highlight_concat_segments(
        ax,
        concat,
        window_start=window_start,
        window_end=window_end,
        ylim_lower=signal_ylim[0],
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


def _decorate_page(
    *,
    fig: Figure,
    entity: ECGEntity,
    page_idx: int,
) -> None:
    if page_idx == 0:
        fig.suptitle(f"{entity.dataset_name}: {entity.entity_id}")
    fig.supxlabel("Time (sec)")
    fig.subplots_adjust(left=0.08, right=0.94, bottom=0.05, top=0.95)


def _highlight_segments(
    ax: Axes,
    segments: list[tuple[str, float, float]],
    *,
    window_start: float,
    window_end: float,
    ylim_lower: float,
    ylim_upper: float,
) -> None:
    for name, start_sec, end_sec in segments:
        if end_sec <= window_start or start_sec >= window_end:
            continue

        highlight_start = max(start_sec, window_start)
        highlight_end = min(end_sec, window_end)
        color = SEGMENT_COLORS.get(name, "#adb5bd")
        ax.axvspan(highlight_start, highlight_end, color=color, alpha=0.15)
        ax.axvline(
            highlight_start,
            color=color,
            linestyle="--",
            linewidth=0.8,
            alpha=0.8,
        )
        ax.axvline(
            highlight_end,
            color=color,
            linestyle="--",
            linewidth=0.8,
            alpha=0.8,
        )

        midpoint = (start_sec + end_sec) / 2
        if window_start <= midpoint <= window_end:
            label = f"{SEGMENT_LABELS.get(name, name)}\n{start_sec:.1f}-{end_sec:.1f}s"
            ax.text(
                midpoint,
                ylim_upper - (ylim_upper - ylim_lower) * 0.02,
                label,
                fontsize=6,
                horizontalalignment="center",
                verticalalignment="top",
                color=color,
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": color,
                    "alpha": 0.8,
                },
            )


def _highlight_concat_segments(
    ax: Axes,
    concat: ConcatenatedSequence,
    *,
    window_start: float,
    window_end: float,
    ylim_lower: float,
    ylim_upper: float,
) -> None:
    segments: list[tuple[str, float, float]] = []
    for name, window in (
        ("sinus_train", concat.segments_info.train),
        ("pre_vf", concat.segments_info.pre_vf),
        ("vf", concat.segments_info.vf),
        ("sinus_test", concat.segments_info.test),
    ):
        segments.append((name, window.start_sec, window.end_sec))

    _highlight_segments(
        ax,
        segments,
        window_start=window_start,
        window_end=window_end,
        ylim_lower=ylim_lower,
        ylim_upper=ylim_upper,
    )
