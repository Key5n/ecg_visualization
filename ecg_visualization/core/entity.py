from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from wfdb.io import Annotation


@dataclass(frozen=True, slots=True)
class ECGEntity:
    """
    Class representing a single ECG record/entity

    Attributes:
        entity_id (str): Identifier for the ECG record
        dataset_name (str): Human-readable dataset label the record belongs to
        dataset_id (str): Identifier matching ECGDataset.dataset_id for stable storage lookups
        sampling_rate_hz (int): Sampling rate of the ECG signal in hertz
        signals (npt.NDArray[np.float64]): ECG signal data
        annotation (Annotation): Annotation object containing metadata about the ECG record
        beats (npt.NDArray[np.int_]): Array of beat sample indices, each element divided by its sampling rate representing the times of beats in seconds
        aux_notes (tuple[str, ...]): Annotation auxiliary notes aligned with annotation.sample for rhythm labels
    """

    entity_id: str
    dataset_name: str
    dataset_id: str
    sampling_rate_hz: int
    signals: npt.NDArray[np.float64]
    annotation: Annotation
    beats: npt.NDArray[np.int_]
    aux_notes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.beats.size < 2:
            raise ValueError(f"{self} does not contain enough beats to analyze")
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

    @property
    def rr_intervals(self) -> npt.NDArray[np.float64]:
        """
        Return consecutive RR intervals (seconds) derived from beat indices.
        """
        beat_times = self.beats / self.sampling_rate_hz
        return np.diff(beat_times)
