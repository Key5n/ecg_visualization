from types import SimpleNamespace

import numpy as np

from ecg_visualization.tasks.rhythm_event_sequences.config import SegmentWindow
from ecg_visualization.tasks.rhythm_event_sequences.event_windows import (
    _adjust_window_to_rpeaks,
)


def _entity_with_beats(*beats: int, sampling_rate_hz: int = 10) -> SimpleNamespace:
    return SimpleNamespace(
        beats=np.asarray(beats, dtype=np.int_),
        dataset=SimpleNamespace(sampling_rate_hz=sampling_rate_hz),
    )


def test_adjust_window_to_rpeaks_preserves_half_open_beat_selection() -> None:
    entity = _entity_with_beats(10, 20, 30, 40, 50)

    adjusted = _adjust_window_to_rpeaks(entity, SegmentWindow(1.2, 3.8))

    assert adjusted == SegmentWindow(2.0, 4.0)


def test_adjust_window_to_rpeaks_uses_final_peak_for_record_end() -> None:
    entity = _entity_with_beats(10, 20, 30, 40, 50)

    adjusted = _adjust_window_to_rpeaks(entity, SegmentWindow(2.0, 6.0))

    assert adjusted == SegmentWindow(2.0, 5.0)
