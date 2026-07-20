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
        annotation_extention (str): Annotation file extension
        beat_extention (str): Beat annotation file extension
        get_entities: Generate entities in the dataset
    """

    dir_path: ClassVar[str]
    name: ClassVar[str]
    dataset_id: ClassVar[str]
    sampling_rate_hz: ClassVar[int]
    annotation_extention: ClassVar[str] = "atr"
    beat_extention: ClassVar[str] = "atr"
    entity_cls: ClassVar[type[ECGEntity]] = ECGEntity
    entity_ids: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def get_entity(cls, *, entity_id: str) -> ECGEntity:
        data_path = os.path.join(cls.dir_path, entity_id)
        signals, _ = wfdb.rdsamp(
            data_path,
            channels=[0],
        )
        squeezed = np.squeeze(signals)

        record = wfdb.rdheader(data_path)
        sampling_rate = float(record.fs)
        if sampling_rate != float(cls.sampling_rate_hz):
            raise ValueError(
                f"{cls.dataset_id}/{entity_id} has sampling rate {record.fs}, "
                f"expected {cls.sampling_rate_hz}"
            )
        record_sampling_rate_hz = int(sampling_rate)

        annotation = read_annotation(cls.annotation_extention, data_path)
        beats = cls._read_beats(data_path, squeezed, record_sampling_rate_hz)

        return cls.entity_cls(
            entity_id=entity_id,
            dataset_name=cls.name,
            dataset_id=cls.dataset_id,
            sampling_rate_hz=record_sampling_rate_hz,
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
        sampling_rate_hz: int,
    ) -> npt.NDArray[np.int_]:
        return read_normal_beats(cls.beat_extention, data_path)

    @classmethod
    def get_entities(cls) -> Iterator[ECGEntity]:
        for entity_id in cls.entity_ids:
            yield cls.get_entity(entity_id=entity_id)

    def __str__(self):
        return self.name
