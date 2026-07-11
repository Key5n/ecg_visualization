import os
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import wfdb
from wfdb.io import Annotation

from ecg_visualization.core.entity import ECGEntity


@dataclass(slots=True)
class ECGDataset:
    """
    Base class for ECG datasets

    Attributes:
        dir_path (str): Path to the dataset directory
        name (str): Name of the dataset
        dataset_id (str): Identifier for the dataset
        annotation_extention_priority (list[str]): List of annotation file extensions in order of priority
        beat_extention_priority (list[str]): List of beat annotation file extensions in order of priority
        data_entities (list[ECGEntity]): List of ECG entities in the dataset
    """

    dir_path: ClassVar[str]
    name: ClassVar[str]
    dataset_id: ClassVar[str]
    annotation_extention_priority: ClassVar[tuple[str, ...]] = ("atr", "qrs", "ari")
    beat_extention_priority: ClassVar[tuple[str, ...]] = ("atr", "qrs", "ari")
    data_entities: list[ECGEntity] = field(default_factory=list)

    def __post_init__(self):
        record_path = os.path.join(self.dir_path, "RECORDS")
        with open(record_path, "r") as f:
            data_ids = f.read().splitlines()

        for data_id in data_ids:
            self.data_entities.append(self._load_entity(data_id))

    @classmethod
    def _read_annotation(cls, data_path: str) -> Annotation:
        for ext in cls.annotation_extention_priority:
            annotation_file = f"{data_path}.{ext}"
            if os.path.isfile(annotation_file):
                annotation = wfdb.rdann(data_path, ext)
                return annotation

    @classmethod
    def _read_normal_beats(cls, data_path: str) -> npt.NDArray[np.int_]:
        for ext in cls.beat_extention_priority:
            annotation_file = f"{data_path}.{ext}"
            if os.path.isfile(annotation_file):
                annotation = cls._read_annotation(data_path)
                if ext == "atr":
                    beats = np.array(
                        [
                            sample
                            for sample, symbol in zip(
                                annotation.sample, annotation.symbol
                            )
                            if symbol == "N"
                        ],
                        dtype=np.int_,
                    )

                    return beats

                return np.asarray(annotation.sample, dtype=np.int_)

        raise FileNotFoundError(f"No annotation file found for {data_path}")

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

        annotation = cls._read_annotation(data_path)
        beats = cls._read_normal_beats(data_path)

        record = wfdb.rdheader(data_path)
        sr = record.fs
        return ECGEntity(
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
