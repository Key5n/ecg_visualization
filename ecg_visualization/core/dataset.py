from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ecg_visualization.core.entity import ECGEntity


@dataclass(frozen=True, slots=True)
class ECGDataset:
    """
    Base class for ECG datasets

    Attributes:
        dir_path (str): Path to the dataset directory
        name (str): Name of the dataset
        dataset_id (str): Identifier for the dataset
        annotation_extention_priority (list[str]): List of annotation file extensions in order of priority
        beat_extention_priority (list[str]): List of beat annotation file extensions in order of priority
        data_entities (tuple[ECGEntity, ...]): Entities in the dataset
    """

    dir_path: ClassVar[str]
    name: ClassVar[str]
    dataset_id: ClassVar[str]
    annotation_extention_priority: ClassVar[tuple[str, ...]] = ("atr", "qrs", "ari")
    beat_extention_priority: ClassVar[tuple[str, ...]] = ("atr", "qrs", "ari")
    entity_cls: ClassVar[type[ECGEntity]] = ECGEntity
    data_entities: ClassVar[tuple[ECGEntity, ...]] = ()

    def extract_normal_segments(
        self,
    ) -> dict[str, npt.NDArray[np.float64]]:
        """
        Extract 10-minute normal RR-interval segments for all records.

        Returns:
            dict[str, np.ndarray]: mapping from record id to the RR-interval
            sequence representing the normal segment.
        """

        segments: dict[str, npt.NDArray[np.float64]] = {}
        for entity in self.data_entities:
            segment = entity.extract_normal_segment()
            segments[entity.entity_id] = segment

        return segments

    def __str__(self):
        return self.name
