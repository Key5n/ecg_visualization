from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import numpy.typing as npt
from wfdb.io import Annotation

if TYPE_CHECKING:
    from ecg_visualization.core.dataset import ECGDataset


@dataclass(frozen=True, slots=True)
class ECGEntity:
    """
    Class representing a single ECG record/entity

    Attributes:
        entity_id (str): Identifier for the ECG record
        dataset (type[ECGDataset]): Dataset class the record belongs to
        signals (npt.NDArray[np.float64]): ECG signal data
        annotation (Annotation): Annotation object containing metadata about the ECG record
        beats (npt.NDArray[np.int_]): Detected R-peak sample indices
        annotated_normal_beats (npt.NDArray[np.int_]): Normal beat sample indices
            read from the dataset's beat annotation file
        aux_notes (tuple[str, ...]): Annotation auxiliary notes aligned with annotation.sample for rhythm labels
    """

    dataset: ClassVar[type["ECGDataset"]]

    entity_id: str
    signals: npt.NDArray[np.float64]
    annotation: Annotation
    beats: npt.NDArray[np.int_]
    annotated_normal_beats: npt.NDArray[np.int_]
    aux_notes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.beats.size < 2:
            raise ValueError(f"{self} does not contain enough beats to analyze")
        self.signals.setflags(write=False)
        self.beats.setflags(write=False)
        self.annotated_normal_beats.setflags(write=False)

    def __str__(self) -> str:
        return f"{self.dataset.dataset_id}/{self.entity_id}"

    @property
    def rr_intervals(self) -> npt.NDArray[np.float64]:
        """
        Return consecutive RR intervals (seconds) derived from beat indices.
        """
        beat_times = self.beats / self.dataset.sampling_rate_hz
        return np.diff(beat_times)
