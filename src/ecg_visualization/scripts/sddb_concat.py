from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ecg_visualization.datasets.dataset import (
    MAX_NORMAL_RR_INTERVAL_SEC,
    MIN_NORMAL_RR_INTERVAL_SEC,
    SDDB,
    ECG_Entity,
)
from ecg_visualization.visualization.export import pdf_exporter
from ecg_visualization.visualization.layouts import (
    PaginationConfig,
    create_page_layout,
    paginate_signals,
)
from ecg_visualization.visualization.limits import compute_ylim
from ecg_visualization.visualization.plotters import plot_signal
from ecg_visualization.visualization.styles import apply_default_style

LOGGER = logging.getLogger(__name__)

SEGMENT_DURATION_SEC = 10 * 60
OUTPUT_DIR = Path("result") / "sddb_concat"
VISUALIZATION_DIR = OUTPUT_DIR / "visualize"

VF_ONSET_SECONDS = SDDB.vf_onset_seconds
PAGINATION_CONFIG = PaginationConfig(seconds_per_row=10, rows_per_page=6)
SEGMENT_COLORS = {
    "sinus_train": "#2a9d8f",
    "pre_vf": "#f4a261",
    "vf": "#e63946",
    "sinus_test": "#264653",
}


@dataclass(frozen=True, slots=True)
class SinusWindow:
    start_sec: float
    end_sec: float


@dataclass(frozen=True, slots=True)
class SinusSegments:
    entity_id: str
    train: SinusWindow
    test: SinusWindow


# Define fixed sinus segments per entity here (seconds).
# Example:
# SINUS_SEGMENTS = (
#     SinusSegments(
#         entity_id="30",
#         train=SinusWindow(0.0, 600.0),
#         test=SinusWindow(72000.0, 72600.0),
#     ),
# )
SINUS_SEGMENTS: tuple[SinusSegments, ...] = (
    SinusSegments(
        entity_id="30",
        train=SinusWindow(2280.0, 2880.0),
        test=SinusWindow(5220.0, 5820.0),
    ),
    SinusSegments(
        entity_id="31",
        train=SinusWindow(600.0, 1200.0),
        test=SinusWindow(10080.0, 10680.0),
    ),
    SinusSegments(
        entity_id="32",
        train=SinusWindow(0.0, 600.0),
        test=SinusWindow(4980.0, 5580.0),
    ),
    SinusSegments(
        entity_id="33",
        train=SinusWindow(0.0, 600.0),
        test=SinusWindow(7500.0, 8100.0),
    ),
    SinusSegments(
        entity_id="34",
        train=SinusWindow(420.0, 1020.0),
        test=SinusWindow(72000.0, 72600.0),
    ),
    SinusSegments(
        entity_id="35",
        train=SinusWindow(0.0, 600.0),
        test=SinusWindow(72000.0, 72600.0),
    ),
    SinusSegments(
        entity_id="36",
        train=SinusWindow(0.0, 600.0),
        test=SinusWindow(72000.0, 72600.0),
    ),
    SinusSegments(
        entity_id="37",
        train=SinusWindow(0.0, 600.0),
        test=SinusWindow(72000.0, 72600.0),
    ),
    SinusSegments(
        entity_id="38",
        train=SinusWindow(0.0, 600.0),
        test=SinusWindow(72000.0, 72600.0),
    ),
    SinusSegments(
        entity_id="39",
        train=SinusWindow(0.0, 600.0),
        test=SinusWindow(72000.0, 72600.0),
    ),
    SinusSegments(
        entity_id="40",
        train=SinusWindow(0.0, 600.0),
        test=SinusWindow(72000.0, 72600.0),
    ),
    SinusSegments(
        entity_id="41",
        train=SinusWindow(0.0, 600.0),
        test=SinusWindow(72000.0, 72600.0),
    ),
    SinusSegments(
        entity_id="42",
        train=SinusWindow(0.0, 600.0),
        test=SinusWindow(72000.0, 72600.0),
    ),
    SinusSegments(
        entity_id="43",
        train=SinusWindow(0.0, 600.0),
        test=SinusWindow(72000.0, 72600.0),
    ),
    SinusSegments(
        entity_id="43",
        train=SinusWindow(0.0, 600.0),
        test=SinusWindow(72000.0, 72600.0),
    ),
    SinusSegments(
        entity_id="44",
        train=SinusWindow(0.0, 600.0),
        test=SinusWindow(72000.0, 72600.0),
    ),
    SinusSegments(
        entity_id="45",
        train=SinusWindow(0.0, 600.0),
        test=SinusWindow(72000.0, 72600.0),
    ),
    SinusSegments(
        entity_id="46",
        train=SinusWindow(0.0, 600.0),
        test=SinusWindow(72000.0, 72600.0),
    ),
    SinusSegments(
        entity_id="47",
        train=SinusWindow(0.0, 600.0),
        test=SinusWindow(72000.0, 72600.0),
    ),
    SinusSegments(
        entity_id="48",
        train=SinusWindow(0.0, 600.0),
        test=SinusWindow(72000.0, 72600.0),
    ),
    SinusSegments(
        entity_id="49",
        train=SinusWindow(0.0, 600.0),
        test=SinusWindow(72000.0, 72600.0),
    ),
    SinusSegments(
        entity_id="50",
        train=SinusWindow(0.0, 600.0),
        test=SinusWindow(72000.0, 72600.0),
    ),
    SinusSegments(
        entity_id="51",
        train=SinusWindow(1.0, 600.0),
        test=SinusWindow(72000.0, 72600.0),
    ),
    SinusSegments(
        entity_id="52",
        train=SinusWindow(1.0, 600.0),
        test=SinusWindow(72000.0, 72600.0),
    ),
)


def concat_sddb() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = SDDB()
    sinus_by_entity = _build_sinus_segments_by_entity(SINUS_SEGMENTS)
    for entity in dataset.data_entities:
        entity_id = entity.entity_id
        if entity_id not in SDDB.vf_onset_seconds:
            LOGGER.info(
                "Skipping %s: no VF onset time available or marked no-VF.",
                entity_id,
            )
            continue

        concat = _build_concatenated_sequence(
            entity,
            SDDB.vf_onset_seconds[entity_id],
            sinus_by_entity.get(entity_id),
        )
        if concat is None:
            continue

        output_path = OUTPUT_DIR / f"{entity_id}.npz"
        np.savez_compressed(output_path, **concat)
        LOGGER.info("Saved concatenated sequence to %s", output_path)


def visualize_sddb_concat() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    apply_default_style()
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

    dataset = SDDB()
    sinus_by_entity = _build_sinus_segments_by_entity(SINUS_SEGMENTS)
    for entity in dataset.data_entities:
        entity_id = entity.entity_id
        vf_onset = SDDB.vf_onset_seconds.get(entity_id)
        if vf_onset is None:
            LOGGER.info(
                "Skipping %s: no VF onset time available or marked no-VF.",
                entity_id,
            )
            continue

        concat = _build_concatenated_sequence(
            entity,
            vf_onset,
            sinus_by_entity.get(entity_id),
        )
        if concat is None:
            continue

        output_path = VISUALIZATION_DIR / f"{entity_id}.pdf"
        _export_concatenated_pdf(entity, concat, output_path)
        LOGGER.info("Saved concatenated visualization to %s", output_path)


def _build_concatenated_sequence(
    entity: ECG_Entity,
    vf_onset_sec: int,
    sinus_segments: SinusSegments | None,
) -> dict[str, np.ndarray] | None:
    signal = entity.signals
    sr = float(entity.sr)
    segment_samples = int(SEGMENT_DURATION_SEC * sr)
    total_duration_sec = signal.size / sr

    pre_vf_start_sec = vf_onset_sec - SEGMENT_DURATION_SEC
    pre_vf_end_sec = vf_onset_sec
    vf_start_sec = vf_onset_sec
    vf_end_sec = vf_onset_sec + SEGMENT_DURATION_SEC

    if pre_vf_start_sec < 0:
        LOGGER.warning(
            "Skipping %s: pre-VF start is before record start.",
            entity.entity_id,
        )
        return None

    if vf_end_sec > total_duration_sec:
        LOGGER.warning(
            "Skipping %s: VF window exceeds record length (%.1fs).",
            entity.entity_id,
            total_duration_sec,
        )
        return None

    if sinus_segments is None:
        train_start_sec = _find_normal_segment_start(
            entity,
            start_bound=0.0,
            end_bound=pre_vf_start_sec,
            duration_sec=SEGMENT_DURATION_SEC,
        )
        if train_start_sec is None:
            LOGGER.warning(
                "Skipping %s: no 10-min sinus segment before pre-VF window.",
                entity.entity_id,
            )
            return None

        test_start_sec = _find_normal_segment_start(
            entity,
            start_bound=vf_end_sec,
            end_bound=total_duration_sec,
            duration_sec=SEGMENT_DURATION_SEC,
        )
        if test_start_sec is None:
            LOGGER.warning(
                "Skipping %s: no 10-min sinus segment after VF window.",
                entity.entity_id,
            )
            return None
    else:
        train_start_sec = _validate_sinus_window(
            entity,
            sinus_segments.train,
            label="train",
            total_duration_sec=total_duration_sec,
        )
        if train_start_sec is None:
            return None
        test_start_sec = _validate_sinus_window(
            entity,
            sinus_segments.test,
            label="test",
            total_duration_sec=total_duration_sec,
        )
        if test_start_sec is None:
            return None

    segments = [
        ("sinus_train", train_start_sec),
        ("pre_vf", pre_vf_start_sec),
        ("vf", vf_start_sec),
        ("sinus_test", test_start_sec),
    ]

    concatenated_samples: list[np.ndarray] = []
    segment_names: list[str] = []
    segment_start_samples: list[int] = []
    segment_end_samples: list[int] = []
    source_start_seconds: list[float] = []
    source_end_seconds: list[float] = []

    running_start = 0
    for name, start_sec in segments:
        start_sample = int(np.round(start_sec * sr))
        end_sample = start_sample + segment_samples
        if end_sample > signal.size:
            LOGGER.warning(
                "Skipping %s: %s segment exceeds record length.",
                entity.entity_id,
                name,
            )
            return None

        concatenated_samples.append(signal[start_sample:end_sample])
        segment_names.append(name)
        segment_start_samples.append(running_start)
        running_start += segment_samples
        segment_end_samples.append(running_start)
        source_start_seconds.append(start_sample / sr)
        source_end_seconds.append(end_sample / sr)

    return {
        "samples": np.concatenate(concatenated_samples),
        "sampling_rate_hz": np.array([sr], dtype=np.float64),
        "segment_names": np.asarray(segment_names, dtype=object),
        "segment_start_samples": np.asarray(segment_start_samples, dtype=np.int_),
        "segment_end_samples": np.asarray(segment_end_samples, dtype=np.int_),
        "source_start_seconds": np.asarray(source_start_seconds, dtype=np.float64),
        "source_end_seconds": np.asarray(source_end_seconds, dtype=np.float64),
    }


def _export_concatenated_pdf(
    entity: ECG_Entity,
    concat: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    samples = np.asarray(concat["samples"], dtype=float)
    sr = float(np.asarray(concat["sampling_rate_hz"], dtype=float)[0])
    segment_names = concat["segment_names"].astype(str).tolist()
    segment_start_samples = concat["segment_start_samples"].astype(int).tolist()
    segment_end_samples = concat["segment_end_samples"].astype(int).tolist()

    segments = [
        (
            name,
            start / sr,
            end / sr,
        )
        for name, start, end in zip(
            segment_names,
            segment_start_samples,
            segment_end_samples,
            strict=True,
        )
    ]

    ts_paged = paginate_signals(
        samples.size,
        int(sr),
        PAGINATION_CONFIG,
    )
    if ts_paged.size == 0:
        LOGGER.warning("Skipping %s: no samples available.", entity.entity_id)
        return

    signal_ylim = compute_ylim(samples, lower_bound=-5.0, upper_bound=5.0)

    with pdf_exporter(str(output_path)) as exporter:
        for page_idx, ts_row in enumerate(ts_paged):
            fig, axs = create_page_layout(PAGINATION_CONFIG.rows_per_page)
            for row_idx, (ts, ax) in enumerate(
                zip(ts_row, np.atleast_1d(axs), strict=True)
            ):
                start_sample = int(round(ts[0] * sr))
                row_signal = _extract_window(samples, start_sample, ts.size)
                plot_signal(
                    ax,
                    ts,
                    row_signal,
                    ylim_lower=signal_ylim[0],
                    ylim_upper=signal_ylim[1],
                )
                _highlight_segments(
                    ax,
                    segments,
                    window_start=float(ts[0]),
                    window_end=float(ts[-1]),
                    ylim_upper=signal_ylim[1],
                )
                if page_idx == 0 and row_idx == 0:
                    ax.set_title("Concatenated SDDB segments", fontsize=9)

            fig.suptitle(f"{entity.dataset_name}: {entity.entity_id}")
            fig.supxlabel("Time (sec)")
            fig.subplots_adjust(left=0.08, right=0.95, bottom=0.05, top=0.93)
            exporter.add_page(fig, pad_inches=0)
            plt.close(fig)


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


def _extract_window(
    samples: np.ndarray,
    start_sample: int,
    length: int,
) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=float)

    window = np.full(length, np.nan, dtype=float)
    if start_sample >= samples.size:
        return window

    end_sample = min(start_sample + length, samples.size)
    window[: end_sample - start_sample] = samples[start_sample:end_sample]
    return window


def _build_sinus_segments_by_entity(
    segments: Iterable[SinusSegments],
) -> dict[str, SinusSegments]:
    return {segment.entity_id: segment for segment in segments}


def _validate_sinus_window(
    entity: ECG_Entity,
    window: SinusWindow,
    *,
    label: str,
    total_duration_sec: float,
) -> float | None:
    if window.end_sec <= window.start_sec:
        LOGGER.warning(
            "Skipping %s: sinus %s window has invalid bounds.",
            entity.entity_id,
            label,
        )
        return None

    duration = window.end_sec - window.start_sec
    if duration < SEGMENT_DURATION_SEC:
        LOGGER.warning(
            "Skipping %s: sinus %s window is shorter than %ds.",
            entity.entity_id,
            label,
            SEGMENT_DURATION_SEC,
        )
        return None

    if window.start_sec < 0 or window.end_sec > total_duration_sec:
        LOGGER.warning(
            "Skipping %s: sinus %s window exceeds record length (%.1fs).",
            entity.entity_id,
            label,
            total_duration_sec,
        )
        return None

    return window.start_sec


def _find_normal_segment_start(
    entity: ECG_Entity,
    *,
    start_bound: float,
    end_bound: float,
    duration_sec: float,
) -> float | None:
    beat_times = entity.beats.astype(np.float64) / entity.sr
    if beat_times.size < 2:
        return None

    rr_intervals = np.diff(beat_times)
    normal_mask = (rr_intervals >= MIN_NORMAL_RR_INTERVAL_SEC) & (
        rr_intervals <= MAX_NORMAL_RR_INTERVAL_SEC
    )
    abnormal_prefix = np.concatenate(([0], np.cumsum(~normal_mask, dtype=np.int_)))

    for start_idx in range(beat_times.size - 1):
        start_time = float(beat_times[start_idx])
        if start_time < start_bound:
            continue

        end_required = start_time + duration_sec
        if end_required > end_bound:
            break

        end_idx = int(np.searchsorted(beat_times, end_required, side="left"))
        if end_idx <= start_idx or end_idx >= beat_times.size:
            continue

        if abnormal_prefix[end_idx] - abnormal_prefix[start_idx] == 0:
            return start_time

    return None
