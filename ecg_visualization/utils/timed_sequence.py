from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class TimedSequence[T: np.generic]:
    """
    Represents raw samples along with the timestamp of each point.
    """

    values: npt.NDArray[T]
    times: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.values.ndim != 1:
            raise ValueError("TimedSequence.values must be 1D")
        if self.times.ndim != 1:
            raise ValueError("TimedSequence.times must be 1D")
        if self.values.shape[0] != self.times.shape[0]:
            raise ValueError(
                "TimedSequence.values and TimedSequence.times must have equal length"
            )

    @property
    def samples(self) -> tuple[tuple[float, T], ...]:
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

    def slice_between(self, start_time: float, end_time: float) -> "TimedSequence[T]":
        """
        Return a new TimedSequence containing samples whose timestamps fall within
        the provided [start_time, end_time] interval.
        """

        if end_time < start_time:
            raise ValueError("end_time must be greater than or equal to start_time")

        mask = np.logical_and(self.times >= start_time, self.times <= end_time)
        return TimedSequence(values=self.values[mask], times=self.times[mask])
