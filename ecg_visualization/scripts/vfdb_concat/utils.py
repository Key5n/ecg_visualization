from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar, Iterable

import numpy as np
import numpy.typing as npt

from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.physionet import VFDB
from ecg_visualization.scripts.vfdb_concat.constants import (
    MAX_REASONABLE_RR_INTERVAL_SEC,
    SEGMENT_DURATION_SEC,
    SegmentsInfo,
    SegmentWindow,
    build_segments_info,
)
from ecg_visualization.utils.signal_processing.rpeak_detection import detect_rpeaks

LOGGER = logging.getLogger(__name__)

VF_RHYTHM_CODES = frozenset({"VF", "VFIB", "VFL"})
SINUS_RHYTHM_CODES = frozenset({"N", "NSR"})
RHYTHM_SINUS = "sinus"
RHYTHM_VF = "vf"


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


@dataclass(frozen=True, slots=True)
class RhythmInterval:
    rhythm: str
    start_sec: float
    end_sec: float


def iter_concatenated_sequences() -> Iterable[tuple[ECGEntity, ConcatenatedSequence]]:
    dataset = VFDB()
    segments_info_by_entity = _build_segments_info_by_entity(dataset.data_entities)
    for entity in dataset.data_entities:
        segments_info = segments_info_by_entity.get(entity.entity_id)
        if segments_info is None:
            LOGGER.info("Skipping %s: no VF/sinus segments found.", entity.entity_id)
            continue

        concat = _build_concatenated_sequence(entity, segments_info)
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


def _build_segments_info_by_entity(
    entities: Iterable[ECGEntity],
) -> dict[str, SegmentsInfo]:
    segments: list[SegmentsInfo] = []
    for entity in entities:
        segments_info = _select_segments_info(entity)
        if segments_info is None:
            continue
        segments.append(segments_info)

    return {segment.entity_id: segment for segment in segments}


def _select_segments_info(entity: ECGEntity) -> SegmentsInfo | None:
    total_duration_sec = float(entity.signals.size) / float(entity.sr)
    rhythm_intervals = _build_rhythm_intervals(entity, total_duration_sec)
    vf_intervals = [
        interval
        for interval in rhythm_intervals
        if interval.rhythm == RHYTHM_VF
        and interval.end_sec - interval.start_sec >= SEGMENT_DURATION_SEC
        and interval.start_sec >= SEGMENT_DURATION_SEC
    ]

    for vf_interval in sorted(vf_intervals, key=lambda interval: interval.start_sec):
        pre_vf = SegmentWindow(
            vf_interval.start_sec - SEGMENT_DURATION_SEC,
            vf_interval.start_sec,
        )
        vf = SegmentWindow(
            vf_interval.start_sec,
            vf_interval.start_sec + SEGMENT_DURATION_SEC,
        )
        sinus_windows = _select_sinus_windows(rhythm_intervals, pre_vf, vf)
        if len(sinus_windows) < 2:
            continue

        return build_segments_info(
            entity.entity_id,
            train=sinus_windows[0],
            test=sinus_windows[1],
            pre_vf=pre_vf,
            vf=vf,
        )

    LOGGER.warning("Skipping %s: no eligible VF event found.", entity.entity_id)
    return None


def _build_rhythm_intervals(
    entity: ECGEntity,
    total_duration_sec: float,
) -> tuple[RhythmInterval, ...]:
    intervals: list[RhythmInterval] = []
    current_rhythm: str | None = None
    current_start_sec: float | None = None

    samples = np.asarray(entity.annotation.sample, dtype=np.float64)
    for sample, aux_note in zip(samples, entity.aux_notes, strict=True):
        rhythm = _rhythm_from_aux_note(aux_note)
        if rhythm is None:
            continue

        sample_sec = float(sample) / float(entity.sr)
        if current_rhythm is not None and current_start_sec is not None:
            _append_rhythm_interval(
                intervals,
                rhythm=current_rhythm,
                start_sec=current_start_sec,
                end_sec=sample_sec,
            )
        current_rhythm = rhythm
        current_start_sec = sample_sec

    if current_rhythm is not None and current_start_sec is not None:
        _append_rhythm_interval(
            intervals,
            rhythm=current_rhythm,
            start_sec=current_start_sec,
            end_sec=total_duration_sec,
        )

    return tuple(intervals)


def _append_rhythm_interval(
    intervals: list[RhythmInterval],
    *,
    rhythm: str,
    start_sec: float,
    end_sec: float,
) -> None:
    if end_sec <= start_sec:
        return
    intervals.append(
        RhythmInterval(
            rhythm=rhythm,
            start_sec=start_sec,
            end_sec=end_sec,
        )
    )


def _rhythm_from_aux_note(aux_note: str) -> str | None:
    rhythm_code = _rhythm_code_from_aux_note(aux_note)
    if rhythm_code in VF_RHYTHM_CODES:
        return RHYTHM_VF
    if rhythm_code in SINUS_RHYTHM_CODES:
        return RHYTHM_SINUS
    return None


def _rhythm_code_from_aux_note(aux_note: str) -> str:
    note = aux_note.strip()
    if note.startswith("("):
        note = note[1:]
    return note.split(maxsplit=1)[0].rstrip(")")


def _select_sinus_windows(
    rhythm_intervals: tuple[RhythmInterval, ...],
    pre_vf: SegmentWindow,
    vf: SegmentWindow,
) -> tuple[SegmentWindow, ...]:
    excluded = SegmentWindow(pre_vf.start_sec, vf.end_sec)
    candidates: list[tuple[float, SegmentWindow]] = []
    for interval in rhythm_intervals:
        if interval.rhythm != RHYTHM_SINUS:
            continue

        for clipped in _clip_window_around_excluded(
            SegmentWindow(interval.start_sec, interval.end_sec),
            excluded,
        ):
            duration = clipped.end_sec - clipped.start_sec
            if duration < SEGMENT_DURATION_SEC:
                continue
            candidates.append(
                (
                    duration,
                    SegmentWindow(
                        clipped.start_sec,
                        clipped.start_sec + SEGMENT_DURATION_SEC,
                    ),
                )
            )

    candidates.sort(
        key=lambda candidate: (
            -candidate[0],
            candidate[1].start_sec,
            candidate[1].end_sec,
        )
    )
    return tuple(window for _, window in candidates[:2])


def _clip_window_around_excluded(
    window: SegmentWindow,
    excluded: SegmentWindow,
) -> tuple[SegmentWindow, ...]:
    if window.end_sec <= excluded.start_sec or window.start_sec >= excluded.end_sec:
        return (window,)

    clipped: list[SegmentWindow] = []
    if window.start_sec < excluded.start_sec:
        clipped.append(SegmentWindow(window.start_sec, excluded.start_sec))
    if window.end_sec > excluded.end_sec:
        clipped.append(SegmentWindow(excluded.end_sec, window.end_sec))
    return tuple(clipped)


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

    detected_beats = detect_rpeaks(segment_samples, entity.sr)
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
        sampling_rate_hz=sr,
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
