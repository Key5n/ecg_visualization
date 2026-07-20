from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar, Iterable

import numpy as np
import numpy.typing as npt

from ecg_visualization.core.dataset import ECGDataset
from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.physionet import load_data_sources
from ecg_visualization.tasks.rhythm_event_sequences.config import (
    RhythmEventSequencesConfig,
    SegmentsInfo,
    SegmentWindow,
)
from ecg_visualization.tasks.rhythm_event_sequences.event_windows import (
    resolve_event_windows,
)
from ecg_visualization.utils.signal_processing.rpeak_detection import detect_rpeaks
from ecg_visualization.utils.utils import find_true_runs

LOGGER = logging.getLogger(__name__)


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


def iter_concatenated_sequences(
    config: RhythmEventSequencesConfig,
) -> Iterable[tuple[ECGEntity, ConcatenatedSequence]]:
    for dataset in load_data_sources((config.dataset_id,)):
        for entity in dataset.get_entities():
            try:
                segments_info = _select_sinus_segments(
                    dataset,
                    entity,
                    segment_duration_sec=config.segment_duration_sec,
                    sinus_rr_median_threshold_sec=config.sinus_rr_median_threshold_sec,
                )
            except ValueError as exc:
                LOGGER.warning("Skipping %s: %s", entity.entity_id, exc)
                continue

            concat = _build_concatenated_sequence(
                entity,
                segments_info,
                max_reasonable_rr_interval_sec=config.max_reasonable_rr_interval_sec,
            )
            if concat is None:
                continue
            yield entity, concat


def _segment_windows(segments_info: SegmentsInfo) -> list[tuple[str, SegmentWindow]]:
    return [
        ("sinus_train", segments_info.train),
        ("pre_vf", segments_info.pre_vf),
        ("vf", segments_info.vf),
        ("sinus_test", segments_info.test),
    ]


def _build_sinus_segments(
    dataset: ECGDataset,
    entities: Iterable[ECGEntity],
    *,
    segment_duration_sec: float,
    sinus_rr_median_threshold_sec: float,
) -> tuple[SegmentsInfo, ...]:
    segments: list[SegmentsInfo] = []
    for entity in entities:
        segments_info = _select_sinus_segments(
            dataset,
            entity,
            segment_duration_sec=segment_duration_sec,
            sinus_rr_median_threshold_sec=sinus_rr_median_threshold_sec,
        )
        segments.append(segments_info)

    return tuple(segments)


def _select_sinus_segments(
    dataset: ECGDataset,
    entity: ECGEntity,
    *,
    segment_duration_sec: float,
    sinus_rr_median_threshold_sec: float,
) -> SegmentsInfo:
    rr_intervals = entity.rr_intervals

    if rr_intervals.size < 2:
        raise ValueError(f"{entity.entity_id} does not contain enough RR intervals")

    beat_times_sec = np.asarray(entity.beats, dtype=np.float64) / float(
        entity.dataset.sampling_rate_hz
    )
    median_rr_interval_sec = float(np.median(rr_intervals))
    near_median_mask = (
        np.abs(rr_intervals - median_rr_interval_sec) <= sinus_rr_median_threshold_sec
    )
    pre_vf_window, vf_window = resolve_event_windows(
        dataset,
        entity,
        segment_duration_sec=segment_duration_sec,
    )
    available_rr_mask = _build_available_rr_mask(
        beat_times_sec,
        pre_vf_window=pre_vf_window,
        vf_window=vf_window,
    )

    candidate_runs = find_true_runs(near_median_mask & available_rr_mask)
    if len(candidate_runs) < 2:
        raise ValueError(
            f"{entity.entity_id} expected 2 sinus runs, found {len(candidate_runs)}"
        )

    candidate_runs.sort(key=lambda run: (-(run[1] - run[0]), run[0]))
    train_window = _rr_run_to_segment_window(beat_times_sec, candidate_runs[0])
    test_window = _rr_run_to_segment_window(beat_times_sec, candidate_runs[1])
    return SegmentsInfo(
        entity_id=entity.entity_id,
        train=train_window,
        test=test_window,
        pre_vf=pre_vf_window,
        vf=vf_window,
    )


def _build_available_rr_mask(
    beat_times_sec: npt.NDArray[np.float64],
    *,
    pre_vf_window: SegmentWindow,
    vf_window: SegmentWindow,
) -> npt.NDArray[np.bool_]:
    rr_start_times_sec = beat_times_sec[:-1]
    rr_end_times_sec = beat_times_sec[1:]
    overlaps_pre_vf = (rr_start_times_sec < pre_vf_window.end_sec) & (
        rr_end_times_sec > pre_vf_window.start_sec
    )
    overlaps_vf = (rr_start_times_sec < vf_window.end_sec) & (
        rr_end_times_sec > vf_window.start_sec
    )
    return ~(overlaps_pre_vf | overlaps_vf)


def _rr_run_to_segment_window(
    beat_times_sec: npt.NDArray[np.float64],
    rr_run: tuple[int, int],
) -> SegmentWindow:
    start_idx, end_idx = rr_run
    return SegmentWindow(
        start_sec=float(beat_times_sec[start_idx]),
        end_sec=float(beat_times_sec[end_idx]),
    )


def _minimum_required_beats(
    segment_duration_sec: float,
    *,
    max_reasonable_rr_interval_sec: float,
) -> int:
    return max(2, int(np.ceil(segment_duration_sec / max_reasonable_rr_interval_sec)))


def _resolve_segment_beats(
    entity: ECGEntity,
    name: str,
    segment_samples: npt.NDArray[np.float64],
    segment_beats: npt.NDArray[np.int_],
    *,
    max_reasonable_rr_interval_sec: float,
) -> npt.NDArray[np.int_]:
    segment_duration_sec = float(segment_samples.size) / float(
        entity.dataset.sampling_rate_hz
    )
    minimum_required_beats = _minimum_required_beats(
        segment_duration_sec,
        max_reasonable_rr_interval_sec=max_reasonable_rr_interval_sec,
    )
    if segment_beats.size >= minimum_required_beats:
        return np.asarray(segment_beats, dtype=np.int_)

    detected_beats = detect_rpeaks(segment_samples, entity.dataset.sampling_rate_hz)
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


def _build_concatenated_sequence(
    entity: ECGEntity,
    segments_info: SegmentsInfo,
    *,
    max_reasonable_rr_interval_sec: float,
) -> ConcatenatedSequence | None:
    signal = entity.signals
    sampling_rate_hz = float(entity.dataset.sampling_rate_hz)
    total_duration_sec = signal.size / sampling_rate_hz

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
        start_sample = int(np.round(window.start_sec * sampling_rate_hz))
        end_sample = int(np.round(window.end_sec * sampling_rate_hz))
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
            max_reasonable_rr_interval_sec=max_reasonable_rr_interval_sec,
        )
        segment_length = end_sample - start_sample
        concatenated_beats.append(
            np.asarray(segment_beats + running_offset, dtype=np.int_)
        )
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
        sampling_rate_hz=sampling_rate_hz,
        segments_info=segments_info,
    )


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
