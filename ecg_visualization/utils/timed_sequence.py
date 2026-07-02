from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(slots=True)
class TimedSequence:
    """
    Represents raw samples along with the timestamp of each point.
    """

    values: npt.NDArray[Any]
    times: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values)
        self.times = np.asarray(self.times, dtype=np.float64)

        if self.values.ndim != 1:
            raise ValueError("TimedSequence.values must be 1D")
        if self.times.ndim != 1:
            raise ValueError("TimedSequence.times must be 1D")
        if self.values.shape[0] != self.times.shape[0]:
            raise ValueError(
                "TimedSequence.values and TimedSequence.times must have equal length"
            )

    @property
    def samples(self) -> tuple[tuple[float, Any], ...]:
        """
        Return tuples containing each sample's time and value.
        """

        return tuple(zip(self.times.tolist(), self.values.tolist(), strict=True))

    @property
    def start_time(self) -> float:
        if self.values.size == 0:
            raise ValueError("TimedSequence is empty")
        return float(self.times[0])

    @property
    def end_time(self) -> float:
        if self.values.size == 0:
            raise ValueError("TimedSequence is empty")
        return float(self.times[-1])

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def length(self) -> int:
        return self.values.shape[0]

    def slice_between(self, start_time: float, end_time: float) -> "TimedSequence":
        """
        Return a new TimedSequence containing samples whose timestamps fall within
        the provided [start_time, end_time] interval.
        """

        if end_time < start_time:
            raise ValueError("end_time must be greater than or equal to start_time")

        mask = np.logical_and(self.times >= start_time, self.times <= end_time)
        return TimedSequence(values=self.values[mask], times=self.times[mask])

    @classmethod
    def from_time_axis(
        cls,
        values: npt.NDArray[Any] | list[Any],
        *,
        time_axis: npt.NDArray[np.float64] | list[float],
    ) -> "TimedSequence":
        """
        Build a timed sequence from precomputed timestamps (e.g., beats in seconds).
        """

        value_array = np.asarray(values)
        time_array = np.asarray(time_axis, dtype=np.float64)
        if value_array.ndim != 1 or time_array.ndim != 1:
            raise ValueError("values and time_axis must be 1D")
        if value_array.shape[0] != time_array.shape[0]:
            raise ValueError("values and time_axis must share the same length")

        return cls(values=value_array, times=time_array)

    @classmethod
    def from_indices(
        cls,
        signal: npt.NDArray[np.float64] | list[float],
        *,
        indices: npt.NDArray[np.int_] | list[int],
        sampling_rate: float,
    ) -> "TimedSequence":
        """
        Extract a timed sequence from a full-resolution signal using indices and sampling rate.
        """

        signal_array = np.asarray(signal, dtype=np.float64)
        index_array = np.asarray(indices, dtype=np.int_)
        if signal_array.ndim != 1:
            raise ValueError("signal must be 1D")
        if index_array.ndim != 1:
            raise ValueError("indices must be 1D")
        if index_array.size == 0:
            raise ValueError("indices cannot be empty")
        if np.any(index_array < 0) or np.any(index_array >= signal_array.shape[0]):
            raise ValueError("indices must lie within the signal bounds")
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")

        times = index_array.astype(np.float64) / sampling_rate
        return cls(values=signal_array[index_array], times=times)

    @classmethod
    def from_slice(
        cls,
        signal: npt.NDArray[np.float64] | list[float],
        *,
        start: int,
        end: int,
        sampling_rate: float,
    ) -> "TimedSequence":
        """
        Extract a contiguous timed sequence using slice bounds and sampling rate.
        """

        signal_array = np.asarray(signal, dtype=np.float64)
        if signal_array.ndim != 1:
            raise ValueError("signal must be 1D")
        if start < 0 or end > signal_array.shape[0] or end <= start:
            raise ValueError("invalid slice bounds for TimedSequence")
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")

        indices = np.arange(start, end, dtype=np.int_)
        times = indices.astype(np.float64) / sampling_rate
        values = signal_array[start:end]
        return cls(values=values, times=times)
