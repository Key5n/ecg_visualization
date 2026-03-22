from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Iterable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes

from ecg_visualization.datasets.dataset import SDDB, ECG_Entity
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
class SegmentWindow:
    start_sec: float
    end_sec: float


@dataclass(frozen=True, slots=True)
class SegmentsInfo:
    entity_id: str
    train: SegmentWindow
    test: SegmentWindow
    pre_vf: SegmentWindow
    vf: SegmentWindow


# Define fixed sinus segments per entity here (seconds).
# Example:
# SINUS_SEGMENTS = (
#     SegmentsInfo(
#         entity_id="30",
#         train=SegmentWindow(0.0, 600.0),
#         test=SegmentWindow(72000.0, 72600.0),
#         pre_vf=SegmentWindow(0.0, 600.0),
#         vf=SegmentWindow(600.0, 1200.0),
#     ),
# )
SINUS_SEGMENTS: tuple[SegmentsInfo, ...] = (
    SegmentsInfo(
        entity_id="30",
        train=SegmentWindow(2280.0, 2880.0),
        test=SegmentWindow(5220.0, 5820.0),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["30"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["30"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["30"],
            VF_ONSET_SECONDS["30"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="31",
        train=SegmentWindow(600.0, 1200.0),
        test=SegmentWindow(10080.0, 10680.0),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["31"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["31"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["31"],
            VF_ONSET_SECONDS["31"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="32",
        train=SegmentWindow(0.0, 600.0),
        test=SegmentWindow(4980.0, 5580.0),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["32"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["32"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["32"],
            VF_ONSET_SECONDS["32"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="33",
        train=SegmentWindow(0.0, 600.0),
        test=SegmentWindow(7500.0, 8100.0),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["33"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["33"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["33"],
            VF_ONSET_SECONDS["33"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="34",
        train=SegmentWindow(420.0, 1020.0),
        test=SegmentWindow(72000.0, 72600.0),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["34"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["34"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["34"],
            VF_ONSET_SECONDS["34"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="35",
        train=SegmentWindow(0.0, 600.0),
        test=SegmentWindow(72000.0, 72600.0),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["35"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["35"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["35"],
            VF_ONSET_SECONDS["35"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="36",
        train=SegmentWindow(0.0, 600.0),
        test=SegmentWindow(72000.0, 72600.0),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["36"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["36"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["36"],
            VF_ONSET_SECONDS["36"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="37",
        train=SegmentWindow(0.0, 600.0),
        test=SegmentWindow(72000.0, 72600.0),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["37"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["37"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["37"],
            VF_ONSET_SECONDS["37"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="38",
        train=SegmentWindow(0.0, 600.0),
        test=SegmentWindow(72000.0, 72600.0),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["38"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["38"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["38"],
            VF_ONSET_SECONDS["38"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="39",
        train=SegmentWindow(0.0, 600.0),
        test=SegmentWindow(72000.0, 72600.0),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["39"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["39"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["39"],
            VF_ONSET_SECONDS["39"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="41",
        train=SegmentWindow(0.0, 600.0),
        test=SegmentWindow(72000.0, 72600.0),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["41"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["41"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["41"],
            VF_ONSET_SECONDS["41"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="43",
        train=SegmentWindow(0.0, 600.0),
        test=SegmentWindow(72000.0, 72600.0),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["43"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["43"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["43"],
            VF_ONSET_SECONDS["43"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="43",
        train=SegmentWindow(0.0, 600.0),
        test=SegmentWindow(72000.0, 72600.0),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["43"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["43"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["43"],
            VF_ONSET_SECONDS["43"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="44",
        train=SegmentWindow(0.0, 600.0),
        test=SegmentWindow(72000.0, 72600.0),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["44"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["44"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["44"],
            VF_ONSET_SECONDS["44"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="45",
        train=SegmentWindow(0.0, 600.0),
        test=SegmentWindow(72000.0, 72600.0),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["45"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["45"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["45"],
            VF_ONSET_SECONDS["45"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="46",
        train=SegmentWindow(0.0, 600.0),
        test=SegmentWindow(72000.0, 72600.0),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["46"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["46"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["46"],
            VF_ONSET_SECONDS["46"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="47",
        train=SegmentWindow(0.0, 600.0),
        test=SegmentWindow(72000.0, 72600.0),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["47"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["47"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["47"],
            VF_ONSET_SECONDS["47"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="48",
        train=SegmentWindow(0.0, 600.0),
        test=SegmentWindow(72000.0, 72600.0),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["48"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["48"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["48"],
            VF_ONSET_SECONDS["48"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="50",
        train=SegmentWindow(0.0, 600.0),
        test=SegmentWindow(72000.0, 72600.0),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["50"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["50"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["50"],
            VF_ONSET_SECONDS["50"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="51",
        train=SegmentWindow(1.0, 600.0),
        test=SegmentWindow(72000.0, 72600.0),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["51"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["51"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["51"],
            VF_ONSET_SECONDS["51"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="52",
        train=SegmentWindow(1.0, 600.0),
        test=SegmentWindow(72000.0, 72600.0),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["52"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["52"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["52"],
            VF_ONSET_SECONDS["52"] + SEGMENT_DURATION_SEC,
        ),
    ),
)


def concat_sddb() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = SDDB()
    segments_info_by_entity = _build_segments_info_by_entity(SINUS_SEGMENTS)
    for entity in dataset.data_entities:
        entity_id = entity.entity_id
        segments_info = segments_info_by_entity.get(entity_id)
        if segments_info is None:
            LOGGER.info("Skipping %s: no sinus segments configured.", entity_id)
            continue

        concat = _build_concatenated_sequence(entity, segments_info)
        if concat is None:
            continue

        output_path = OUTPUT_DIR / f"{entity_id}.pkl"
        with output_path.open("wb") as handle:
            pickle.dump(concat, handle, protocol=pickle.HIGHEST_PROTOCOL)
        LOGGER.info("Saved concatenated sequence to %s", output_path)


def visualize_sddb_concat() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    apply_default_style()
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

    dataset = SDDB()
    segments_info_by_entity = _build_segments_info_by_entity(SINUS_SEGMENTS)
    for entity in dataset.data_entities:
        entity_id = entity.entity_id
        segments_info = segments_info_by_entity.get(entity_id)
        if segments_info is None:
            LOGGER.info("Skipping %s: no sinus segments configured.", entity_id)
            continue

        concat = _build_concatenated_sequence(entity, segments_info)
        if concat is None:
            continue

        output_path = VISUALIZATION_DIR / f"{entity_id}.pdf"
        _export_concatenated_pdf(entity, concat, output_path)
        LOGGER.info("Saved concatenated visualization to %s", output_path)


@dataclass(frozen=True, slots=True)
class ConcatenatedSequence:
    samples: npt.NDArray[np.float64]
    beats: npt.NDArray[np.int_]
    sampling_rate_hz: float
    segments_info: SegmentsInfo
    SEGMENT_ORDER: ClassVar[tuple[str, ...]] = (
        "sinus_train",
        "pre_vf",
        "vf",
        "sinus_test",
    )


def _segment_windows(segments_info: SegmentsInfo) -> list[tuple[str, SegmentWindow]]:
    return [
        ("sinus_train", segments_info.train),
        ("pre_vf", segments_info.pre_vf),
        ("vf", segments_info.vf),
        ("sinus_test", segments_info.test),
    ]


def _build_concatenated_sequence(
    entity: ECG_Entity,
    segments_info: SegmentsInfo,
) -> ConcatenatedSequence | None:
    signal = entity.signals
    sr = float(entity.sr)
    total_duration_sec = signal.size / sr

    if not _validate_segment_window(
        entity,
        segments_info.train,
        label="sinus train",
        total_duration_sec=total_duration_sec,
    ):
        return None
    if not _validate_segment_window(
        entity,
        segments_info.test,
        label="sinus test",
        total_duration_sec=total_duration_sec,
    ):
        return None
    if not _validate_segment_window(
        entity,
        segments_info.pre_vf,
        label="pre-VF",
        total_duration_sec=total_duration_sec,
    ):
        return None
    if not _validate_segment_window(
        entity,
        segments_info.vf,
        label="VF",
        total_duration_sec=total_duration_sec,
    ):
        return None

    concatenated_samples: list[np.ndarray] = []
    concatenated_beats: list[npt.NDArray[np.int_]] = []
    running_offset = 0
    for name, window in _segment_windows(segments_info):
        start_sample = int(np.round(window.start_sec * sr))
        end_sample = int(np.round(window.end_sec * sr))
        if end_sample > signal.size:
            LOGGER.warning(
                "Skipping %s: %s segment exceeds record length.",
                entity.entity_id,
                name,
            )
            return None

        concatenated_samples.append(signal[start_sample:end_sample])
        segment_beats = entity.beats[
            (entity.beats >= start_sample) & (entity.beats < end_sample)
        ]
        segment_length = end_sample - start_sample
        concatenated_beats.append(
            np.asarray(
                segment_beats - start_sample + running_offset,
                dtype=np.int_,
            )
        )
        running_offset += segment_length

    return ConcatenatedSequence(
        samples=np.asarray(np.concatenate(concatenated_samples), dtype=np.float64),
        beats=(
            np.asarray(np.concatenate(concatenated_beats), dtype=np.int_)
            if concatenated_beats
            else np.array([], dtype=np.int_)
        ),
        sampling_rate_hz=sr,
        segments_info=segments_info,
    )


def _export_concatenated_pdf(
    entity: ECG_Entity,
    concat: ConcatenatedSequence,
    output_path: Path,
) -> None:
    samples = np.asarray(concat.samples, dtype=float)
    sr = float(concat.sampling_rate_hz)
    segments: list[tuple[str, float, float]] = []
    running_start = 0
    for name, window in _segment_windows(concat.segments_info):
        segment_samples = int(
            np.round((window.end_sec - window.start_sec) * concat.sampling_rate_hz)
        )
        segments.append((name, running_start / sr, (running_start + segment_samples) / sr))
        running_start += segment_samples

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


def _build_segments_info_by_entity(
    segments: Iterable[SegmentsInfo],
) -> dict[str, SegmentsInfo]:
    return {segment.entity_id: segment for segment in segments}



def _validate_segment_window(
    entity: ECG_Entity,
    window: SegmentWindow,
    *,
    label: str,
    total_duration_sec: float,
) -> bool:
    if window.end_sec <= window.start_sec:
        LOGGER.warning(
            "Skipping %s: %s window has invalid bounds.",
            entity.entity_id,
            label,
        )
        return False

    if window.start_sec < 0 or window.end_sec > total_duration_sec:
        LOGGER.warning(
            "Skipping %s: %s window exceeds record length (%.1fs).",
            entity.entity_id,
            label,
            total_duration_sec,
        )
        return False

    return True
