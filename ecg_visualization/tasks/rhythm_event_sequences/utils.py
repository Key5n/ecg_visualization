from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Iterable

import numpy as np
import numpy.typing as npt
import structlog

from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.physionet import load_data_sources
from ecg_visualization.tasks.rhythm_event_sequences.config import (
    SegmentsInfo,
    SegmentWindow,
    SinusExtractionConfig,
)
from ecg_visualization.tasks.rhythm_event_sequences.event_windows import (
    resolve_event_windows,
)
from ecg_visualization.utils.utils import find_true_runs

LOGGER = structlog.get_logger(__name__)


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


def select_sequence_result(
    entity: ECGEntity,
    *,
    pre_ar_duration_sec: float,
    sinus_extraction_config: SinusExtractionConfig,
) -> SequenceSelectionResult:
    try:
        segments_info = _select_sinus_segments(
            entity=entity,
            pre_ar_duration_sec=pre_ar_duration_sec,
            sinus_extraction_config=sinus_extraction_config,
        )
        concat = _build_concatenated_sequence(entity, segments_info)
    except ValueError as exc:
        LOGGER.warning("entity_skipped", entity=entity, reason=str(exc))
        return SequenceSelectionFailure(
            entity=entity,
            failure_reason=str(exc),
        )

    return SequenceSelectionSuccess(
        entity=entity,
        concat=concat,
    )


def iter_concatenated_sequences(
    dataset_id: str,
    *,
    pre_ar_duration_sec: float,
    sinus_extraction_config: SinusExtractionConfig,
) -> Iterable[tuple[ECGEntity, ConcatenatedSequence]]:
    dataset = load_data_sources((dataset_id,))[0]
    for entity in dataset.get_entities():
        result = select_sequence_result(
            entity,
            pre_ar_duration_sec=pre_ar_duration_sec,
            sinus_extraction_config=sinus_extraction_config,
        )
        if isinstance(result, SequenceSelectionSuccess):
            yield result.entity, result.concat


def _segment_windows(segments_info: SegmentsInfo) -> list[tuple[str, SegmentWindow]]:
    windows = [
        ("sinus_train", segments_info.train),
        ("pre_ar", segments_info.pre_ar),
        ("sinus_test", segments_info.test),
    ]
    if segments_info.ar is not None:
        windows.insert(2, ("ar", segments_info.ar))
    return windows


def _select_sinus_segments(
    entity: ECGEntity,
    *,
    pre_ar_duration_sec: float,
    sinus_extraction_config: SinusExtractionConfig,
) -> SegmentsInfo:
    rr_intervals = entity.rr_intervals

    if rr_intervals.size < 2:
        raise ValueError(f"{entity} does not contain enough RR intervals")

    beat_times_sec = np.asarray(entity.beats, dtype=np.float64) / float(
        entity.dataset.sampling_rate_hz
    )
    sinus_rr_mask = _build_sinus_rr_mask(
        rr_intervals,
        sinus_extraction_config,
    )
    pre_ar_window, ar_window = resolve_event_windows(
        entity,
        pre_ar_duration_sec=pre_ar_duration_sec,
    )
    available_rr_mask = _build_available_rr_mask(
        beat_times_sec,
        pre_ar_window=pre_ar_window,
        ar_window=ar_window,
    )

    candidate_runs = find_true_runs(sinus_rr_mask & available_rr_mask)
    before_pre_ar_runs = [
        run
        for run in candidate_runs
        if beat_times_sec[run[1]] <= pre_ar_window.start_sec
    ]
    before_pre_ar_runs.sort(key=_sinus_run_sort_key)
    if not before_pre_ar_runs:
        raise ValueError(f"{entity} has no train sinus before pre-AR")

    test_after_ar_runs = (
        [run for run in candidate_runs if beat_times_sec[run[0]] >= ar_window.end_sec]
        if ar_window is not None
        else []
    )
    test_after_ar_runs.sort(key=_sinus_run_sort_key)
    train_run = before_pre_ar_runs[0]
    if not test_after_ar_runs:
        LOGGER.info(
            "test_sinus_fallback",
            entity=entity,
            reason="no test sinus after AR",
        )
        test_runs = before_pre_ar_runs[1:]
    else:
        test_runs = test_after_ar_runs
    if not test_runs:
        raise ValueError(f"{entity} has no non-overlapping test sinus before pre-AR")
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


def _build_sinus_rr_mask(
    rr_intervals: npt.NDArray[np.float64],
    sinus_extraction_config: SinusExtractionConfig,
) -> npt.NDArray[np.bool_]:
    if sinus_extraction_config.method == "median_threshold":
        median_rr_interval_sec = float(np.median(rr_intervals))
        return np.asarray(
            np.abs(rr_intervals - median_rr_interval_sec)
            <= sinus_extraction_config.median_threshold.threshold_sec,
            dtype=np.bool_,
        )

    if sinus_extraction_config.method == "percentile_range":
        lower_rr_interval_sec, upper_rr_interval_sec = np.percentile(
            rr_intervals,
            (
                sinus_extraction_config.percentile_range.lower_percentile,
                sinus_extraction_config.percentile_range.upper_percentile,
            ),
        )
        return np.asarray(
            (rr_intervals >= lower_rr_interval_sec)
            & (rr_intervals <= upper_rr_interval_sec),
            dtype=np.bool_,
        )

    raise ValueError(
        f"Unknown sinus extraction method '{sinus_extraction_config.method}'"
    )


def _sinus_run_sort_key(run: tuple[int, int]) -> tuple[int, int]:
    return (-(run[1] - run[0]), run[0])


def _build_available_rr_mask(
    beat_times_sec: npt.NDArray[np.float64],
    *,
    pre_ar_window: SegmentWindow,
    ar_window: SegmentWindow | None,
) -> npt.NDArray[np.bool_]:
    rr_start_times_sec = beat_times_sec[:-1]
    rr_end_times_sec = beat_times_sec[1:]
    overlaps_pre_ar = (rr_start_times_sec < pre_ar_window.end_sec) & (
        rr_end_times_sec > pre_ar_window.start_sec
    )
    overlaps_ar = (
        (rr_start_times_sec < ar_window.end_sec)
        & (rr_end_times_sec > ar_window.start_sec)
        if ar_window is not None
        else np.zeros_like(overlaps_pre_ar)
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


def _build_concatenated_sequence(
    entity: ECGEntity,
    segments_info: SegmentsInfo,
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
        segment_beats = (
            entity.beats[(entity.beats >= start_sample) & (entity.beats < end_sample)]
            - start_sample
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
