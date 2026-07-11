from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from wfdb.io import Annotation

from ecg_visualization.config.settings import (
    MAX_NORMAL_RR_INTERVAL_SEC,
    MIN_NORMAL_RR_INTERVAL_SEC,
    NORMAL_SEGMENT_DURATION_SEC,
)
from ecg_visualization.utils.timed_sequence import TimedSequence
from ecg_visualization.utils.utils import merge_overlapping_windows


@dataclass(frozen=True, slots=True)
class ECGEntity:
    """
    Class representing a single ECG record/entity

    Attributes:
        entity_id (str): Identifier for the ECG record
        dataset_name (str): Human-readable dataset label the record belongs to
        dataset_id (str): Identifier matching ECGDataset.dataset_id for stable storage lookups
        sr (int): Sampling rate of the ECG signal
        signals (npt.NDArray[np.float64]): ECG signal data
        annotation (Annotation): Annotation object containing metadata about the ECG record
        beats (npt.NDArray[np.int_]): Array of beat sample indices, each element divided by its sampling rate representing the times of beats in seconds
        aux_notes (tuple[str, ...]): Annotation auxiliary notes aligned with annotation.sample for rhythm labels
    """

    entity_id: str
    dataset_name: str
    dataset_id: str
    sr: int
    signals: npt.NDArray[np.float64]
    annotation: Annotation
    beats: npt.NDArray[np.int_]
    aux_notes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.beats.size < 2:
            raise ValueError(
                f"{self.entity_id} does not contain enough beats to analyze"
            )
        self.signals.setflags(write=False)
        self.beats.setflags(write=False)

    def __str__(self) -> str:
        return f"{self.dataset_id}/{self.entity_id}"

    def get_window_durations(
        self,
        window_size: int,
    ) -> Iterator[tuple[int, int]]:
        """
        Yield contiguous windows of RR intervals.

        Args:
            window_size (int): Number of RR intervals per window.

        Returns:
            Iterator[tuple[int, int]]: Sample index windows for the requested size.
            Yields nothing when insufficient beats are available.
        """

        if window_size < 2:
            raise ValueError("window_size must be at least 2 beats")

        if self.beats.size < window_size + 1:
            return

        for start_idx in range(self.beats.size - window_size):
            start_sample = int(self.beats[start_idx])
            end_sample = int(self.beats[start_idx + window_size])
            yield start_sample, end_sample

    def extract_normal_segment(self) -> TimedSequence[np.float64]:
        """
        Extract and return the RR intervals that compose a 5-minute normal beat
        segment for this entity.

        Returns:
            TimedSequence: Consecutive RR intervals (in seconds) plus their start times.

        Raises:
            ValueError: If the entity does not contain enough information to
            determine such a segment.
        """

        beat_times = self.beats / self.sr
        rr_intervals = self.compute_rr_intervals()

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

                interval_start_times = beat_times[
                    start_idx : start_idx + rr_segment.size
                ]
                return TimedSequence(
                    values=np.asarray(rr_segment, dtype=np.float64),
                    times=np.asarray(interval_start_times, dtype=np.float64),
                )

        raise ValueError(
            f"No 5-minute normal beat segment found for {self.entity_id} "
            f"({self.dataset_name})"
        )

    def compute_rr_intervals(self) -> npt.NDArray[np.float64]:
        """
        Return consecutive RR intervals (seconds) derived from beat indices.
        """
        beat_times = self.beats / self.sr
        return np.diff(beat_times)

    def get_abnormal_windows(
        self,
        window_size: int,
        min_duration: float,
        max_duration: float,
    ) -> set[tuple[float, float]]:
        """
        Identify abnormal windows based on RR intervals.

        Args:
            window_size (int): Number of beats in each window.
            min_duration (float): Minimum duration for a normal window in seconds.
            max_duration (float): Maximum duration for a normal window in seconds.

        Returns:
            set[tuple[float, float]]: Set of tuples representing start and end times
            of abnormal windows.
        """

        abnormal_windows: set[tuple[float, float]] = set()
        for start_sample, end_sample in self.get_window_durations(window_size):
            start_time = start_sample / self.sr
            end_time = end_sample / self.sr
            duration = end_time - start_time
            if duration < min_duration or duration > max_duration:
                abnormal_windows.add((start_time, end_time))

        abnormal_windows = merge_overlapping_windows(abnormal_windows)
        return abnormal_windows

    def get_extreme_rr_windows(
        self,
        window_size: int,
        *,
        lower_percentile: float = 5.0,
        upper_percentile: float = 95.0,
    ) -> set[tuple[float, float]]:
        """
        Collect start/end times for 10-R-peak windows in the lowest or
        highest percentile range of durations across all such windows.
        """

        windows = list(self.get_window_durations(window_size))
        if not windows:
            return set()

        if not 0 <= lower_percentile < upper_percentile <= 100:
            raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100")

        durations_arr = np.array([(end - start) for start, end in windows], dtype=float)
        lower_bound = np.percentile(durations_arr, lower_percentile)
        upper_bound = np.percentile(durations_arr, upper_percentile)

        extreme_windows = set(
            (
                start_sample / self.sr,
                end_sample / self.sr,
            )
            for start_sample, end_sample in windows
            if (end_sample - start_sample) < lower_bound
            or (end_sample - start_sample) > upper_bound
        )
        return merge_overlapping_windows(extreme_windows)
