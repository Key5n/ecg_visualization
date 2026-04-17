from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar, Iterable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from biosppy.signals.ecg import ecg as biosppy_ecg
from matplotlib.axes import Axes

from ecg_visualization.datasets.dataset import SDDB, ECGEntity
from ecg_visualization.models.md_rs.md_rs import MDRS
from ecg_visualization.scripts.sddb_concat_mdrs.constants import (
    DEFAULT_MD_RS_CONFIG,
    MAX_REASONABLE_RR_INTERVAL_SEC,
    OUTPUT_DIR,
    SEGMENT_COLORS,
    SINUS_SEGMENTS,
    SegmentWindow,
    SegmentsInfo,
    WINDOW_SIZE,
)
from ecg_visualization.utils.utils import prepare_sequences, sliding_window_sequences
from ecg_visualization.visualization.styles import apply_default_style

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScoreResult:
    times_sec: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True, slots=True)
class ConcatenatedSequence:
    samples: npt.NDArray[np.float64]
    beats: npt.NDArray[np.int_]
    symbol_samples: npt.NDArray[np.int_]
    symbol_values: tuple[str, ...]
    sampling_rate_hz: float
    segments_info: SegmentsInfo
    SEGMENT_ORDER: ClassVar[tuple[str, ...]] = (
        "sinus_train",
        "pre_vf",
        "vf",
        "sinus_test",
    )


def sddb_concat_mdrs_scores() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    apply_default_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = SDDB()
    segments_info_by_entity = _build_segments_info_by_entity(SINUS_SEGMENTS)

    processed = 0
    skipped = 0
    for entity in dataset.data_entities:
        segments_info = segments_info_by_entity.get(entity.entity_id)
        if segments_info is None:
            LOGGER.info("Skipping %s: no sinus segments configured.", entity.entity_id)
            skipped += 1
            continue

        concat = _build_concatenated_sequence(entity, segments_info)
        if concat is None:
            skipped += 1
            continue

        try:
            score_result = _score_concatenated_sequence(concat)
        except ValueError as exc:
            LOGGER.warning("Skipping %s: %s", entity.entity_id, exc)
            skipped += 1
            continue

        fig = _plot_concat_scores(entity, concat, score_result)
        output_path = OUTPUT_DIR / f"{entity.entity_id}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        LOGGER.info("Saved MD-RS scores to %s", output_path)
        processed += 1

    LOGGER.info(
        "Finished scoring. processed=%d skipped=%d output_dir=%s",
        processed,
        skipped,
        OUTPUT_DIR,
    )


def _segment_windows(segments_info: SegmentsInfo) -> list[tuple[str, SegmentWindow]]:
    return [
        ("sinus_train", segments_info.train),
        ("pre_vf", segments_info.pre_vf),
        ("vf", segments_info.vf),
        ("sinus_test", segments_info.test),
    ]


def _minimum_required_beats(segment_duration_sec: float) -> int:
    return max(2, int(np.ceil(segment_duration_sec / MAX_REASONABLE_RR_INTERVAL_SEC)))


def _resolve_segment_beats(
    entity: ECGEntity,
    name: str,
    segment_samples: npt.NDArray[np.float64],
    segment_beats: npt.NDArray[np.int_],
) -> npt.NDArray[np.int_]:
    segment_duration_sec = float(segment_samples.size) / float(entity.sr)
    minimum_required_beats = _minimum_required_beats(segment_duration_sec)
    if segment_beats.size >= minimum_required_beats:
        return np.asarray(segment_beats, dtype=np.int_)

    detected_beats = _detect_rpeaks(segment_samples, entity.sr)
    if detected_beats.size >= max(2, segment_beats.size):
        LOGGER.info(
            "Using detected R-peaks for %s:%s (annotated=%d detected=%d min_required=%d).",
            entity.entity_id,
            name,
            segment_beats.size,
            detected_beats.size,
            minimum_required_beats,
        )
        return detected_beats

    LOGGER.warning(
        "R-peak fallback for %s:%s was insufficient (annotated=%d detected=%d min_required=%d).",
        entity.entity_id,
        name,
        segment_beats.size,
        detected_beats.size,
        minimum_required_beats,
    )
    return np.asarray(segment_beats, dtype=np.int_)


def _detect_rpeaks(
    signal: npt.NDArray[np.float64],
    sampling_rate_hz: int,
) -> npt.NDArray[np.int_]:
    samples = np.asarray(signal, dtype=np.float64)
    if samples.size < 3:
        return np.array([], dtype=np.int_)

    try:
        result = biosppy_ecg(
            signal=samples,
            sampling_rate=float(sampling_rate_hz),
            show=False,
        )
    except Exception as exc:
        LOGGER.warning("biosppy R-peak detection failed: %s", exc)
        return np.array([], dtype=np.int_)

    return np.asarray(result["rpeaks"], dtype=np.int_)


def _build_concatenated_sequence(
    entity: ECGEntity,
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
    concatenated_symbol_samples: list[npt.NDArray[np.int_]] = []
    concatenated_symbol_values: list[str] = []
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
        annotated_segment_beats = entity.beats[
            (entity.beats >= start_sample) & (entity.beats < end_sample)
        ]
        segment_samples = np.asarray(signal[start_sample:end_sample], dtype=np.float64)
        segment_beats = _resolve_segment_beats(
            entity,
            name,
            segment_samples,
            np.asarray(annotated_segment_beats - start_sample, dtype=np.int_),
        )
        segment_length = end_sample - start_sample
        concatenated_beats.append(np.asarray(segment_beats + running_offset, dtype=np.int_))
        annotation_mask = (entity.annotation.sample >= start_sample) & (
            entity.annotation.sample < end_sample
        )
        segment_symbol_samples = np.asarray(
            entity.annotation.sample[annotation_mask] - start_sample + running_offset,
            dtype=np.int_,
        )
        concatenated_symbol_samples.append(segment_symbol_samples)
        concatenated_symbol_values.extend(
            entity.annotation.symbol[idx]
            for idx, in_segment in enumerate(annotation_mask)
            if in_segment
        )
        running_offset += segment_length

    return ConcatenatedSequence(
        samples=np.asarray(np.concatenate(concatenated_samples), dtype=np.float64),
        beats=(
            np.asarray(np.concatenate(concatenated_beats), dtype=np.int_)
            if concatenated_beats
            else np.array([], dtype=np.int_)
        ),
        symbol_samples=(
            np.asarray(np.concatenate(concatenated_symbol_samples), dtype=np.int_)
            if concatenated_symbol_samples
            else np.array([], dtype=np.int_)
        ),
        symbol_values=tuple(concatenated_symbol_values),
        sampling_rate_hz=sr,
        segments_info=segments_info,
    )


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


def _build_segments_info_by_entity(
    segments: Iterable[SegmentsInfo],
) -> dict[str, SegmentsInfo]:
    return {segment.entity_id: segment for segment in segments}


def _validate_segment_window(
    entity: ECGEntity,
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


def _score_concatenated_sequence(concat: ConcatenatedSequence) -> ScoreResult:
    beat_samples = np.asarray(concat.beats, dtype=np.int_)
    if beat_samples.size < WINDOW_SIZE + 1:
        raise ValueError("sequence does not contain enough beats")

    train_segment_samples = int(
        np.round(
            (concat.segments_info.train.end_sec - concat.segments_info.train.start_sec)
            * concat.sampling_rate_hz
        )
    )
    train_beats = beat_samples[beat_samples < train_segment_samples]
    if train_beats.size < WINDOW_SIZE + 1:
        raise ValueError("sinus_train segment does not contain enough beats")

    beat_times = beat_samples.astype(np.float64) / concat.sampling_rate_hz
    train_beat_times = train_beats.astype(np.float64) / concat.sampling_rate_hz
    rr_intervals = np.diff(beat_times)
    train_rr_intervals = np.diff(train_beat_times)

    train_windows = sliding_window_sequences(train_rr_intervals, WINDOW_SIZE)
    test_windows = sliding_window_sequences(rr_intervals, WINDOW_SIZE)

    train_sequence, test_sequence = prepare_sequences(train_windows, test_windows)

    config = {
        **DEFAULT_MD_RS_CONFIG,
        "N_u": train_sequence.shape[1],
    }
    model = MDRS(**config)
    model.train(train_sequence)
    model.reset_states()

    scores = model.predict(test_sequence)
    scores[: config["trans_length"]] = np.nan
    score_times = beat_times[WINDOW_SIZE:]
    return ScoreResult(times_sec=score_times, scores=scores)


def _plot_concat_scores(
    entity: ECGEntity,
    concat: ConcatenatedSequence,
    score_result: ScoreResult,
) -> plt.Figure:
    samples = np.asarray(concat.samples, dtype=float)
    sr = float(concat.sampling_rate_hz)
    ts = np.arange(samples.size, dtype=float) / sr

    signal_min = float(np.nanmin(samples))
    signal_max = float(np.nanmax(samples))
    signal_margin = (signal_max - signal_min) * 0.05 or 1.0
    signal_ylim = (signal_min - signal_margin, signal_max + signal_margin)

    scores = np.asarray(score_result.scores, dtype=float)
    positive_scores = scores[scores > 0]
    score_min = float(np.nanmin(positive_scores)) if positive_scores.size else 1e-6
    score_max = float(np.nanmax(scores)) if scores.size else 1.0
    score_margin = min(
        (score_max - score_min) * 0.1 or score_min * 0.1 or 1e-6,
        score_min * 0.99,
    )
    score_ylim = (score_min - score_margin, score_max + score_margin)

    fig, (signal_ax, score_ax) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(5, 2),
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
    score_ax.set_ylim(*score_ylim)

    _highlight_concat_segments(signal_ax, concat, ylim_upper=signal_ylim[1])
    _highlight_concat_segments(score_ax, concat, ylim_upper=score_ylim[1])

    signal_ax.set_title(f"ID: {entity.entity_id}")
    fig.tight_layout()
    return fig


def _highlight_concat_segments(
    ax: plt.Axes,
    concat: ConcatenatedSequence,
    *,
    ylim_upper: float,
) -> None:
    segments: list[tuple[str, float, float]] = []
    running_start = 0
    for name, window in _segment_windows(concat.segments_info):
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
