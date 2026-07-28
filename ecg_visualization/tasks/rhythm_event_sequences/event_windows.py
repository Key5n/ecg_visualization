from __future__ import annotations

from functools import singledispatch

import numpy as np

from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.ltafdb import LTAFDBEntity
from ecg_visualization.datasets.sddb import SDDB, SDDBEntity
from ecg_visualization.datasets.vfdb import VFDBEntity
from ecg_visualization.tasks.rhythm_event_sequences.config import SegmentWindow

VFDB_RHYTHM_LABEL_PRIORITIES = (
    frozenset({"VF", "VFIB"}),
    frozenset({"VT"}),
    frozenset({"VFL"}),
)


@singledispatch
def resolve_event_windows(
    entity: ECGEntity,
    *,
    pre_ar_duration_sec: float,
) -> tuple[SegmentWindow, SegmentWindow | None]:
    raise ValueError(
        f"event window resolution is not configured for dataset "
        f"'{entity.dataset.dataset_id}'"
    )


@resolve_event_windows.register
def _resolve_sddb_event_windows(
    entity: SDDBEntity,
    *,
    pre_ar_duration_sec: float,
) -> tuple[SegmentWindow, None]:
    ar_onset_sec = SDDB.vf_onset_seconds.get(entity.entity_id)
    if ar_onset_sec is None:
        raise ValueError(f"AR onset is not configured for entity '{entity.entity_id}'.")
    pre_ar_window = SegmentWindow(
        ar_onset_sec - pre_ar_duration_sec,
        ar_onset_sec,
    )
    return (_adjust_window_to_rpeaks(entity, pre_ar_window), None)


@resolve_event_windows.register
def _resolve_ltafdb_event_windows(
    entity: LTAFDBEntity,
    *,
    pre_ar_duration_sec: float,
) -> tuple[SegmentWindow, SegmentWindow]:
    event_window = _find_first_rhythm_event_window(
        entity,
        target_labels={"AF", "AFIB"},
        min_start_sec=pre_ar_duration_sec,
    )
    pre_ar_window, ar_window = _build_pre_event_windows(
        event_window,
        pre_ar_duration_sec=pre_ar_duration_sec,
    )
    return (
        _adjust_window_to_rpeaks(entity, pre_ar_window),
        _adjust_window_to_rpeaks(entity, ar_window),
    )


@resolve_event_windows.register
def _resolve_vfdb_event_windows(
    entity: VFDBEntity,
    *,
    pre_ar_duration_sec: float,
) -> tuple[SegmentWindow, SegmentWindow]:
    event_window = _find_first_rhythm_event_window_by_priority(
        entity,
        target_label_priorities=VFDB_RHYTHM_LABEL_PRIORITIES,
        bridge_noise=True,
        min_start_sec=pre_ar_duration_sec,
    )
    pre_ar_window, ar_window = _build_pre_event_windows(
        event_window,
        pre_ar_duration_sec=pre_ar_duration_sec,
    )
    return (
        _adjust_window_to_rpeaks(entity, pre_ar_window),
        _adjust_window_to_rpeaks(entity, ar_window),
    )


def _adjust_window_to_rpeaks(
    entity: ECGEntity,
    window: SegmentWindow,
) -> SegmentWindow:
    """Move a sample-based window to equivalent half-open R-peak boundaries."""
    beat_times_sec = np.asarray(entity.beats, dtype=np.float64) / float(
        entity.dataset.sampling_rate_hz
    )
    start_idx = int(np.searchsorted(beat_times_sec, window.start_sec, side="left"))
    end_idx = int(np.searchsorted(beat_times_sec, window.end_sec, side="left"))

    # A rhythm episode can extend to the end of a record, after its final R-peak.
    # In that case the final R-peak is the only possible exclusive boundary.
    end_idx = min(end_idx, beat_times_sec.size - 1)
    if start_idx >= end_idx:
        raise ValueError(
            f"window {window.start_sec:g}s-{window.end_sec:g}s for {entity} "
            "does not contain a complete RR interval"
        )

    return SegmentWindow(
        start_sec=float(beat_times_sec[start_idx]),
        end_sec=float(beat_times_sec[end_idx]),
    )


def _build_pre_event_windows(
    event_window: SegmentWindow,
    *,
    pre_ar_duration_sec: float,
) -> tuple[SegmentWindow, SegmentWindow]:
    return (
        SegmentWindow(
            start_sec=event_window.start_sec - pre_ar_duration_sec,
            end_sec=event_window.start_sec,
        ),
        event_window,
    )


def _find_first_rhythm_event_window_by_priority(
    entity: ECGEntity,
    *,
    target_label_priorities: tuple[frozenset[str], ...],
    bridge_noise: bool = False,
    min_start_sec: float = 0.0,
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
            bridge_noise=bridge_noise,
            min_start_sec=min_start_sec,
        )
        if event_window is not None:
            return event_window

    label_groups = [
        "/".join(sorted(target_labels)) for target_labels in target_label_priorities
    ]
    labels = _format_label_groups(label_groups)
    raise ValueError(
        f"no {labels} rhythm episode found at or after {min_start_sec:g} seconds"
    )


def _find_first_rhythm_event_window(
    entity: ECGEntity,
    *,
    target_labels: set[str],
    min_start_sec: float = 0.0,
) -> SegmentWindow:
    rhythm_events = _iter_rhythm_events(entity)
    total_duration_sec = float(entity.signals.size) / float(
        entity.dataset.sampling_rate_hz
    )
    event_window = _find_first_rhythm_event_window_in_events(
        rhythm_events,
        target_labels=target_labels,
        total_duration_sec=total_duration_sec,
        min_start_sec=min_start_sec,
    )
    if event_window is not None:
        return event_window

    labels = ", ".join(sorted(target_labels))
    raise ValueError(
        f"no {labels} rhythm episode found at or after {min_start_sec:g} seconds"
    )


def _find_first_rhythm_event_window_in_events(
    rhythm_events: list[tuple[float, str]],
    *,
    target_labels: set[str] | frozenset[str],
    total_duration_sec: float,
    bridge_noise: bool = False,
    min_start_sec: float = 0.0,
) -> SegmentWindow | None:
    for idx, (start_sec, label) in enumerate(rhythm_events):
        if label not in target_labels or start_sec < min_start_sec:
            continue

        end_sec = _find_episode_end_sec(
            rhythm_events,
            start_idx=idx,
            total_duration_sec=total_duration_sec,
            bridge_noise=bridge_noise,
        )

        return SegmentWindow(
            start_sec=start_sec,
            end_sec=end_sec,
        )

    return None


def _find_episode_end_sec(
    rhythm_events: list[tuple[float, str]],
    *,
    start_idx: int,
    total_duration_sec: float,
    bridge_noise: bool,
) -> float:
    for next_start_sec, next_label in rhythm_events[start_idx + 1 :]:
        if bridge_noise and next_label == "NOISE":
            continue
        return next_start_sec

    return total_duration_sec


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
