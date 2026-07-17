from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
import numpy.typing as npt
import wfdb

from ecg_visualization.core.annotations import read_annotation, read_normal_beats
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
    data_entities: tuple[ECGEntity, ...] = ()

    @classmethod
    def load(cls) -> Self:
        return cls(
            data_entities=tuple(
                cls._load_entity(data_id) for data_id in cls._read_data_ids()
            )
        )

    @classmethod
    def _read_data_ids(cls) -> list[str]:
        record_path = os.path.join(cls.dir_path, "RECORDS")
        with open(record_path, "r") as f:
            return f.read().splitlines()

    @classmethod
    def _load_entity(cls, data_id: str) -> ECGEntity:
        """
        Load a single entity without instantiating the dataset and populating all
        records.
        """
        data_path = os.path.join(cls.dir_path, data_id)
        signals, _ = wfdb.rdsamp(
            data_path,
            channels=[0],
        )
        squeezed = np.squeeze(signals)

        annotation = read_annotation(cls, data_path)
        beats = read_normal_beats(cls, data_path)

        record = wfdb.rdheader(data_path)
        sr = record.fs
        return cls.entity_cls(
            entity_id=data_id,
            dataset_name=cls.name,
            dataset_id=cls.dataset_id,
            sr=sr,
            signals=squeezed,
            annotation=annotation,
            beats=beats,
            aux_notes=tuple(annotation.aux_note),
        )

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
