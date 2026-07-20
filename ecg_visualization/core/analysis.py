import numpy as np

from ecg_visualization.config.settings import (
    MAX_NORMAL_RR_INTERVAL_SEC,
    MIN_NORMAL_RR_INTERVAL_SEC,
    NORMAL_SEGMENT_DURATION_SEC,
)
from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.utils.timed_sequence import TimedSequence
from ecg_visualization.utils.utils import merge_overlapping_windows


def extract_normal_segment(entity: ECGEntity) -> TimedSequence[np.float64]:
    """
    Extract and return the RR intervals that compose a 5-minute normal beat
    segment for this entity.

    Returns:
        TimedSequence: Consecutive RR intervals (in seconds) plus their start times.

    Raises:
        ValueError: If the entity does not contain enough information to
        determine such a segment.
    """

    beat_times = entity.beats / entity.sampling_rate_hz
    rr_intervals = entity.rr_intervals

    start_idx = 0
    for interval_idx, rr_interval in enumerate(rr_intervals):
        if (
            rr_interval < MIN_NORMAL_RR_INTERVAL_SEC
            or rr_interval > MAX_NORMAL_RR_INTERVAL_SEC
        ):
            start_idx = interval_idx + 1
            continue

        current_duration = beat_times[interval_idx + 1] - beat_times[start_idx]
        if current_duration >= NORMAL_SEGMENT_DURATION_SEC:
            rr_segment = rr_intervals[start_idx : interval_idx + 1]
            if rr_segment.size == 0:
                break

            interval_start_times = beat_times[start_idx : start_idx + rr_segment.size]
            return TimedSequence(
                values=np.asarray(rr_segment, dtype=np.float64),
                times=np.asarray(interval_start_times, dtype=np.float64),
            )

    raise ValueError(
        f"No 5-minute normal beat segment found for {entity.entity_id} "
        f"({entity.dataset_name})"
    )


def get_abnormal_windows(
    entity: ECGEntity,
    window_size: int,
    min_duration: float,
    max_duration: float,
) -> set[tuple[float, float]]:
    """
    Identify abnormal windows based on RR intervals.

    Args:
        entity (ECGEntity): ECG entity to analyze.
        window_size (int): Number of beats in each window.
        min_duration (float): Minimum duration for a normal window in seconds.
        max_duration (float): Maximum duration for a normal window in seconds.

    Returns:
        set[tuple[float, float]]: Set of tuples representing start and end times
        of abnormal windows.
    """

    abnormal_windows: set[tuple[float, float]] = set()
    for start_sample, end_sample in entity.get_window_durations(window_size):
        start_time = start_sample / entity.sampling_rate_hz
        end_time = end_sample / entity.sampling_rate_hz
        duration = end_time - start_time
        if duration < min_duration or duration > max_duration:
            abnormal_windows.add((start_time, end_time))

    abnormal_windows = merge_overlapping_windows(abnormal_windows)
    return abnormal_windows


def get_extreme_rr_windows(
    entity: ECGEntity,
    window_size: int,
    *,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
) -> set[tuple[float, float]]:
    """
    Collect start/end times for 10-R-peak windows in the lowest or highest
    percentile range of durations across all such windows.
    """

    windows = list(entity.get_window_durations(window_size))
    if not windows:
        return set()

    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100")

    durations_arr = np.array([(end - start) for start, end in windows], dtype=float)
    lower_bound = np.percentile(durations_arr, lower_percentile)
    upper_bound = np.percentile(durations_arr, upper_percentile)

    extreme_windows = set(
        (
            start_sample / entity.sampling_rate_hz,
            end_sample / entity.sampling_rate_hz,
        )
        for start_sample, end_sample in windows
        if (end_sample - start_sample) < lower_bound
        or (end_sample - start_sample) > upper_bound
    )
    return merge_overlapping_windows(extreme_windows)
