from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar

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
        get_entities: Generate entities in the dataset
    """

    dir_path: ClassVar[str]
    name: ClassVar[str]
    dataset_id: ClassVar[str]
    sr: ClassVar[int]
    annotation_extention_priority: ClassVar[tuple[str, ...]] = ("atr", "qrs", "ari")
    beat_extention_priority: ClassVar[tuple[str, ...]] = ("atr", "qrs", "ari")
    entity_cls: ClassVar[type[ECGEntity]] = ECGEntity
    entity_ids: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def load(cls, *, id: str) -> ECGEntity:
        data_path = os.path.join(cls.dir_path, id)
        signals, _ = wfdb.rdsamp(
            data_path,
            channels=[0],
        )
        squeezed = np.squeeze(signals)

        record = wfdb.rdheader(data_path)
        sampling_rate = float(record.fs)
        if sampling_rate != float(cls.sr):
            raise ValueError(
                f"{cls.dataset_id}/{id} has sampling rate {record.fs}, "
                f"expected {cls.sr}"
            )
        record_sr = int(sampling_rate)

        annotation = read_annotation(cls.annotation_extention_priority, data_path)
        beats = cls._read_beats(data_path, squeezed, record_sr)

        return cls.entity_cls(
            entity_id=id,
            dataset_name=cls.name,
            dataset_id=cls.dataset_id,
            sr=record_sr,
            signals=squeezed,
            annotation=annotation,
            beats=beats,
            aux_notes=tuple(annotation.aux_note),
        )

    @classmethod
    def _read_beats(
        cls,
        data_path: str,
        signals: npt.NDArray[np.float64],
        sr: int,
    ) -> npt.NDArray[np.int_]:
        return read_normal_beats(cls.beat_extention_priority, data_path)

    @classmethod
    def get_entities(cls) -> Iterator[ECGEntity]:
        for entity_id in cls.entity_ids:
            yield cls.load(id=entity_id)

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
        for entity in self.get_entities():
            segment = entity.extract_normal_segment()
            segments[entity.entity_id] = segment

        return segments

    def __str__(self):
        return self.name
