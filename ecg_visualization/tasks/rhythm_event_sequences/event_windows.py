from __future__ import annotations

from functools import singledispatch

import numpy as np

from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.ltafdb import LTAFDBEntity
from ecg_visualization.datasets.sddb import SDDB, SDDBEntity
from ecg_visualization.datasets.vfdb import VFDBEntity
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
    entity: ECGEntity,
    *,
    pre_vf_duration_sec: float,
    vf_duration_sec: float,
) -> tuple[SegmentWindow, SegmentWindow]:
    raise ValueError(
        f"event window resolution is not configured for dataset "
        f"'{entity.dataset.dataset_id}'"
    )


@resolve_event_windows.register
def _resolve_sddb_event_windows(
    entity: SDDBEntity,
    *,
    pre_vf_duration_sec: float,
    vf_duration_sec: float,
) -> tuple[SegmentWindow, SegmentWindow]:
    return build_fixed_vf_windows(
        entity.entity_id,
        pre_vf_duration_sec=pre_vf_duration_sec,
        vf_duration_sec=vf_duration_sec,
        vf_onset_seconds=SDDB.vf_onset_seconds,
    )


@resolve_event_windows.register
def _resolve_ltafdb_event_windows(
    entity: LTAFDBEntity,
    *,
    pre_vf_duration_sec: float,
    vf_duration_sec: float,
) -> tuple[SegmentWindow, SegmentWindow]:
    event_window = _find_first_rhythm_event_window(
        entity,
        target_labels={"AF", "AFIB"},
        vf_duration_sec=vf_duration_sec,
    )
    return _build_pre_event_windows(
        event_window,
        pre_vf_duration_sec=pre_vf_duration_sec,
    )


@resolve_event_windows.register
def _resolve_vfdb_event_windows(
    entity: VFDBEntity,
    *,
    pre_vf_duration_sec: float,
    vf_duration_sec: float,
) -> tuple[SegmentWindow, SegmentWindow]:
    event_window = _find_first_rhythm_event_window_by_priority(
        entity,
        target_label_priorities=VFDB_RHYTHM_LABEL_PRIORITIES,
        vf_duration_sec=vf_duration_sec,
    )
    return _build_pre_event_windows(
        event_window,
        pre_vf_duration_sec=pre_vf_duration_sec,
    )


def _build_pre_event_windows(
    event_window: SegmentWindow,
    *,
    pre_vf_duration_sec: float,
) -> tuple[SegmentWindow, SegmentWindow]:
    return (
        SegmentWindow(
            start_sec=event_window.start_sec - pre_vf_duration_sec,
            end_sec=event_window.start_sec,
        ),
        event_window,
    )


def _find_first_rhythm_event_window_by_priority(
    entity: ECGEntity,
    *,
    target_label_priorities: tuple[frozenset[str], ...],
    vf_duration_sec: float,
) -> SegmentWindow:
    rhythm_events = _iter_rhythm_events(entity)
    total_duration_sec = float(entity.signals.size) / float(
        entity.dataset.sampling_rate_hz
    )
    for target_labels in target_label_priorities:
        event_window = _find_first_rhythm_event_window_in_events(
            rhythm_events,
            target_labels=target_labels,
            total_duration_sec=total_duration_sec,
            vf_duration_sec=vf_duration_sec,
        )
        if event_window is not None:
            return event_window

    label_groups = [
        "/".join(sorted(target_labels)) for target_labels in target_label_priorities
    ]
    labels = _format_label_groups(label_groups)
    raise ValueError(
        f"no {labels} rhythm episode of at least {vf_duration_sec:.1f}s found"
    )


def _find_first_rhythm_event_window(
    entity: ECGEntity,
    *,
    target_labels: set[str],
    vf_duration_sec: float,
) -> SegmentWindow:
    rhythm_events = _iter_rhythm_events(entity)
    total_duration_sec = float(entity.signals.size) / float(
        entity.dataset.sampling_rate_hz
    )
    event_window = _find_first_rhythm_event_window_in_events(
        rhythm_events,
        target_labels=target_labels,
        total_duration_sec=total_duration_sec,
        vf_duration_sec=vf_duration_sec,
    )
    if event_window is not None:
        return event_window

    labels = ", ".join(sorted(target_labels))
    raise ValueError(
        f"no {labels} rhythm episode of at least {vf_duration_sec:.1f}s found"
    )


def _find_first_rhythm_event_window_in_events(
    rhythm_events: list[tuple[float, str]],
    *,
    target_labels: set[str] | frozenset[str],
    total_duration_sec: float,
    vf_duration_sec: float,
) -> SegmentWindow | None:
    for idx, (start_sec, label) in enumerate(rhythm_events):
        if label not in target_labels:
            continue

        end_sec = total_duration_sec
        if idx + 1 < len(rhythm_events):
            end_sec = rhythm_events[idx + 1][0]

        if end_sec - start_sec < vf_duration_sec:
            continue

        return SegmentWindow(
            start_sec=start_sec,
            end_sec=start_sec + vf_duration_sec,
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
            rhythm_events.append(
                (float(sample) / float(entity.dataset.sampling_rate_hz), label)
            )

    return rhythm_events


def _normalize_rhythm_label(note: str) -> str:
    label = note.strip().strip("\x00").upper()
    while label.startswith("("):
        label = label[1:].strip()
    while label.endswith(")"):
        label = label[:-1].strip()
    return label
