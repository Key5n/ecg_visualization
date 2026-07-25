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
        "pre_ar",
        "ar",
        "sinus_test",
    )


@dataclass(frozen=True, slots=True)
class SequenceSelectionSuccess:
    entity: ECGEntity
    concat: ConcatenatedSequence

    @property
    def segments_info(self) -> SegmentsInfo:
        return self.concat.segments_info


@dataclass(frozen=True, slots=True)
class SequenceSelectionFailure:
    entity: ECGEntity
    failure_reason: str


SequenceSelectionResult = SequenceSelectionSuccess | SequenceSelectionFailure


def iter_sequence_selection_results(
    config: RhythmEventSequencesConfig,
) -> Iterable[SequenceSelectionResult]:
    dataset = load_data_sources((config.dataset_id,))[0]
    for entity in dataset.get_entities():
        try:
            segments_info = _select_sinus_segments(
                dataset,
                entity,
                pre_ar_duration_sec=config.pre_ar_duration_sec,
                ar_duration_sec=config.ar_duration_sec,
                sinus_rr_median_threshold_sec=config.sinus_rr_median_threshold_sec,
            )
            concat = _build_concatenated_sequence(
                entity,
                segments_info,
                max_reasonable_rr_interval_sec=config.max_reasonable_rr_interval_sec,
            )
        except ValueError as exc:
            LOGGER.warning("Skipping %s: %s", entity.entity_id, exc)
            yield SequenceSelectionFailure(
                entity=entity,
                failure_reason=str(exc),
            )
            continue

        yield SequenceSelectionSuccess(
            entity=entity,
            concat=concat,
        )


def iter_concatenated_sequences(
    config: RhythmEventSequencesConfig,
) -> Iterable[tuple[ECGEntity, ConcatenatedSequence]]:
    for result in iter_sequence_selection_results(config):
        if isinstance(result, SequenceSelectionSuccess):
            yield result.entity, result.concat


def _segment_windows(segments_info: SegmentsInfo) -> list[tuple[str, SegmentWindow]]:
    return [
        ("sinus_train", segments_info.train),
        ("pre_ar", segments_info.pre_ar),
        ("ar", segments_info.ar),
        ("sinus_test", segments_info.test),
    ]


def _select_sinus_segments(
    dataset: ECGDataset,
    entity: ECGEntity,
    *,
    pre_ar_duration_sec: float,
    ar_duration_sec: float,
    sinus_rr_median_threshold_sec: float,
) -> SegmentsInfo:
    rr_intervals = entity.rr_intervals

    if rr_intervals.size < 2:
        raise ValueError(f"{entity} does not contain enough RR intervals")

    beat_times_sec = np.asarray(entity.beats, dtype=np.float64) / float(
        entity.dataset.sampling_rate_hz
    )
    median_rr_interval_sec = float(np.median(rr_intervals))
    near_median_mask = (
        np.abs(rr_intervals - median_rr_interval_sec) <= sinus_rr_median_threshold_sec
    )
    pre_ar_window, ar_window = resolve_event_windows(
        entity,
        pre_ar_duration_sec=pre_ar_duration_sec,
        ar_duration_sec=ar_duration_sec,
    )
    available_rr_mask = _build_available_rr_mask(
        beat_times_sec,
        pre_ar_window=pre_ar_window,
        ar_window=ar_window,
    )

    candidate_runs = find_true_runs(near_median_mask & available_rr_mask)
    before_pre_ar_runs = [
        run
        for run in candidate_runs
        if beat_times_sec[run[1]] <= pre_ar_window.start_sec
    ]
    before_pre_ar_runs.sort(key=_sinus_run_sort_key)
    if not before_pre_ar_runs:
        raise ValueError(f"{entity.entity_id} has no train sinus before pre-AR")

    test_after_ar_runs = [
        run for run in candidate_runs if beat_times_sec[run[0]] >= ar_window.end_sec
    ]
    test_after_ar_runs.sort(key=_sinus_run_sort_key)
    train_run = before_pre_ar_runs[0]
    if not test_after_ar_runs:
        LOGGER.warning(
            "%s has no test sinus after AR; falling back to pre-AR candidate runs",
            entity.entity_id,
        )
        test_runs = before_pre_ar_runs[1:]
    else:
        test_runs = test_after_ar_runs
    if not test_runs:
        raise ValueError(
            f"{entity.entity_id} has no non-overlapping test sinus before pre-AR"
        )
    test_run = test_runs[0]

    train_window = _rr_run_to_segment_window(beat_times_sec, train_run)
    test_window = _rr_run_to_segment_window(beat_times_sec, test_run)
    return SegmentsInfo(
        entity_id=entity.entity_id,
        train=train_window,
        test=test_window,
        pre_ar=pre_ar_window,
        ar=ar_window,
    )


def _sinus_run_sort_key(run: tuple[int, int]) -> tuple[int, int]:
    return (-(run[1] - run[0]), run[0])


def _build_available_rr_mask(
    beat_times_sec: npt.NDArray[np.float64],
    *,
    pre_ar_window: SegmentWindow,
    ar_window: SegmentWindow,
) -> npt.NDArray[np.bool_]:
    rr_start_times_sec = beat_times_sec[:-1]
    rr_end_times_sec = beat_times_sec[1:]
    overlaps_pre_ar = (rr_start_times_sec < pre_ar_window.end_sec) & (
        rr_end_times_sec > pre_ar_window.start_sec
    )
    overlaps_ar = (rr_start_times_sec < ar_window.end_sec) & (
        rr_end_times_sec > ar_window.start_sec
    )
    return ~(overlaps_pre_ar | overlaps_ar)


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
) -> ConcatenatedSequence:
    signal = entity.signals
    sampling_rate_hz = float(entity.dataset.sampling_rate_hz)

    _validate_segments_info(entity, segments_info)

    concatenated_samples: list[np.ndarray] = []
    concatenated_beats: list[npt.NDArray[np.int_]] = []
    concatenated_symbol_samples: list[npt.NDArray[np.int_]] = []
    concatenated_symbol_values: list[str] = []
    running_offset = 0
    for name, window in _segment_windows(segments_info):
        start_sample = int(np.round(window.start_sec * sampling_rate_hz))
        end_sample = int(np.round(window.end_sec * sampling_rate_hz))
        if end_sample > signal.size:
            raise ValueError(
                f"{name} segment exceeds record length for {entity.entity_id}"
            )

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


def _validate_segments_info(
    entity: ECGEntity,
    segments_info: SegmentsInfo,
) -> None:
    if segments_info.entity_id != entity.entity_id:
        raise ValueError(
            f"segments info entity_id '{segments_info.entity_id}' does not match "
            f"entity '{entity.entity_id}'"
        )

    total_duration_sec = entity.signals.size / float(entity.dataset.sampling_rate_hz)
    labels = {
        "sinus_train": "sinus train",
        "sinus_test": "sinus test",
        "pre_ar": "pre-AR",
        "ar": "AR",
    }
    for name, window in _segment_windows(segments_info):
        if window.end_sec > total_duration_sec:
            label = labels.get(name, name)
            raise ValueError(
                f"{label} window exceeds record length for {entity.entity_id} "
                f"({total_duration_sec:.1f}s)"
            )
