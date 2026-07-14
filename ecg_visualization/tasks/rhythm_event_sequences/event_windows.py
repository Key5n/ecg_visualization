from __future__ import annotations

from functools import singledispatch

import numpy as np

from ecg_visualization.core.dataset import ECGDataset
from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.physionet import LTAFDB, SDDB, VFDB
from ecg_visualization.tasks.rhythm_event_sequences.config import (
    SegmentWindow,
    build_fixed_vf_windows,
)

VFDB_RHYTHM_LABEL_PRIORITIES = (
    frozenset({"VF", "VFIB"}),
    frozenset({"VT"}),
    frozenset({"VFL"}),
)


@singledispatch
def resolve_event_windows(
    dataset: ECGDataset,
    entity: ECGEntity,
    *,
    segment_duration_sec: float,
) -> tuple[SegmentWindow, SegmentWindow]:
    raise ValueError(
        f"event window resolution is not configured for dataset '{dataset.dataset_id}'"
    )


@resolve_event_windows.register
def _resolve_sddb_event_windows(
    dataset: SDDB,
    entity: ECGEntity,
    *,
    segment_duration_sec: float,
) -> tuple[SegmentWindow, SegmentWindow]:
    return build_fixed_vf_windows(
        entity.entity_id,
        segment_duration_sec=segment_duration_sec,
        vf_onset_seconds=SDDB.vf_onset_seconds,
    )


@resolve_event_windows.register
def _resolve_ltafdb_event_windows(
    dataset: LTAFDB,
    entity: ECGEntity,
    *,
    segment_duration_sec: float,
) -> tuple[SegmentWindow, SegmentWindow]:
    event_window = _find_first_rhythm_event_window(
        entity,
        target_labels={"AF", "AFIB"},
        segment_duration_sec=segment_duration_sec,
    )
    return _build_pre_event_windows(
        event_window,
        segment_duration_sec=segment_duration_sec,
    )


@resolve_event_windows.register
def _resolve_vfdb_event_windows(
    dataset: VFDB,
    entity: ECGEntity,
    *,
    segment_duration_sec: float,
) -> tuple[SegmentWindow, SegmentWindow]:
    event_window = _find_first_rhythm_event_window_by_priority(
        entity,
        target_label_priorities=VFDB_RHYTHM_LABEL_PRIORITIES,
        segment_duration_sec=segment_duration_sec,
    )
    return _build_pre_event_windows(
        event_window,
        segment_duration_sec=segment_duration_sec,
    )


def _build_pre_event_windows(
    event_window: SegmentWindow,
    *,
    segment_duration_sec: float,
) -> tuple[SegmentWindow, SegmentWindow]:
    return (
        SegmentWindow(
            start_sec=event_window.start_sec - segment_duration_sec,
            end_sec=event_window.start_sec,
        ),
        event_window,
    )


def _find_first_rhythm_event_window_by_priority(
    entity: ECGEntity,
    *,
    target_label_priorities: tuple[frozenset[str], ...],
    segment_duration_sec: float,
) -> SegmentWindow:
    rhythm_events = _iter_rhythm_events(entity)
    total_duration_sec = float(entity.signals.size) / float(entity.sr)
    for target_labels in target_label_priorities:
        event_window = _find_first_rhythm_event_window_in_events(
            rhythm_events,
            target_labels=target_labels,
            total_duration_sec=total_duration_sec,
            segment_duration_sec=segment_duration_sec,
        )
        if event_window is not None:
            return event_window

    label_groups = [
        "/".join(sorted(target_labels)) for target_labels in target_label_priorities
    ]
    labels = _format_label_groups(label_groups)
    raise ValueError(
        f"no {labels} rhythm episode of at least {segment_duration_sec:.1f}s found"
    )


def _find_first_rhythm_event_window(
    entity: ECGEntity,
    *,
    target_labels: set[str],
    segment_duration_sec: float,
) -> SegmentWindow:
    rhythm_events = _iter_rhythm_events(entity)
    total_duration_sec = float(entity.signals.size) / float(entity.sr)
    event_window = _find_first_rhythm_event_window_in_events(
        rhythm_events,
        target_labels=target_labels,
        total_duration_sec=total_duration_sec,
        segment_duration_sec=segment_duration_sec,
    )
    if event_window is not None:
        return event_window

    labels = ", ".join(sorted(target_labels))
    raise ValueError(
        f"no {labels} rhythm episode of at least {segment_duration_sec:.1f}s found"
    )


def _find_first_rhythm_event_window_in_events(
    rhythm_events: list[tuple[float, str]],
    *,
    target_labels: set[str] | frozenset[str],
    total_duration_sec: float,
    segment_duration_sec: float,
) -> SegmentWindow | None:
    for idx, (start_sec, label) in enumerate(rhythm_events):
        if label not in target_labels:
            continue

        end_sec = total_duration_sec
        if idx + 1 < len(rhythm_events):
            end_sec = rhythm_events[idx + 1][0]

        if end_sec - start_sec < segment_duration_sec:
            continue

        return SegmentWindow(
            start_sec=start_sec,
            end_sec=start_sec + segment_duration_sec,
        )

    return None


def _format_label_groups(label_groups: list[str]) -> str:
    if len(label_groups) == 1:
        return label_groups[0]
    return f"{', '.join(label_groups[:-1])}, or {label_groups[-1]}"


def _iter_rhythm_events(entity: ECGEntity) -> list[tuple[float, str]]:
    samples = np.asarray(entity.annotation.sample, dtype=np.float64)
    rhythm_events: list[tuple[float, str]] = []
    for sample, note in zip(samples, entity.aux_notes, strict=True):
        label = _normalize_rhythm_label(note)
        if label:
            rhythm_events.append((float(sample) / float(entity.sr), label))

    return rhythm_events


def _normalize_rhythm_label(note: str) -> str:
    label = note.strip().strip("\x00").upper()
    while label.startswith("("):
        label = label[1:].strip()
    while label.endswith(")"):
        label = label[:-1].strip()
    return label
