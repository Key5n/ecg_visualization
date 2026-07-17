import os
from dataclasses import dataclass
from typing import ClassVar, Sequence, Type

import numpy as np
import wfdb

from ecg_visualization.config.settings import DATASET_ROOT
from ecg_visualization.core.dataset import ECGDataset
from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.utils.signal_processing.rpeak_detection import detect_rpeaks

physionet_root_dir = os.path.join(DATASET_ROOT, "physionet.org", "files")


@dataclass(frozen=True, slots=True)
class CUDBEntity(ECGEntity):
    pass


# https://physionet.org/content/cudb/1.0.0/
@dataclass(slots=True)
class CUDB(ECGDataset):
    dir_path: ClassVar[str] = os.path.join(physionet_root_dir, "cudb", "1.0.0")
    name: ClassVar[str] = "Tachyarrythmia"
    dataset_id: ClassVar[str] = "cudb"
    sr: ClassVar[int] = 250
    entity_cls: ClassVar[type[ECGEntity]] = CUDBEntity


@dataclass(frozen=True, slots=True)
class AFPDBEntity(ECGEntity):
    pass


# https://physionet.org/content/afpdb/1.0.0/
@dataclass(slots=True)
class AFPDB(ECGDataset):
    dir_path: ClassVar[str] = os.path.join(physionet_root_dir, "afpdb", "1.0.0")
    name: ClassVar[str] = "PAF Prediction Challenge Database"
    dataset_id: ClassVar[str] = "afpdb"
    sr: ClassVar[int] = 128
    entity_cls: ClassVar[type[ECGEntity]] = AFPDBEntity


@dataclass(frozen=True, slots=True)
class MITDBEntity(ECGEntity):
    pass


# https://physionet.org/content/mitdb/1.0.0/
@dataclass(slots=True)
class MITDB(ECGDataset):
    dir_path: ClassVar[str] = os.path.join(physionet_root_dir, "mitdb", "1.0.0")
    name: ClassVar[str] = "MIT-BIH Arrhythmia Database"
    dataset_id: ClassVar[str] = "mitdb"
    sr: ClassVar[int] = 360
    entity_cls: ClassVar[type[ECGEntity]] = MITDBEntity


@dataclass(frozen=True, slots=True)
class AFDBEntity(ECGEntity):
    pass


# https://physionet.org/content/afdb/1.0.0/
@dataclass(slots=True)
class AFDB(ECGDataset):
    dir_path: ClassVar[str] = os.path.join(physionet_root_dir, "afdb", "1.0.0")
    name: ClassVar[str] = "MIT-BIH Atrial Fibrillation Database"
    dataset_id: ClassVar[str] = "afdb"
    sr: ClassVar[int] = 250
    entity_cls: ClassVar[type[ECGEntity]] = AFDBEntity

    def __post_init__(self):
        record_path = os.path.join(self.dir_path, "RECORDS")
        with open(record_path, "r") as f:
            data_ids = f.read().splitlines()

        data_ids = list(
            filter(lambda data_id: data_id not in ["00735", "03665"], data_ids)
        )

        for data_id in data_ids:
            self.data_entities.append(self._load_entity(data_id))


@dataclass(frozen=True, slots=True)
class LTAFDBEntity(ECGEntity):
    pass


# https://physionet.org/content/ltafdb/1.0.0/
@dataclass(slots=True)
class LTAFDB(ECGDataset):
    dir_path: ClassVar[str] = os.path.join(physionet_root_dir, "ltafdb", "1.0.0")
    name: ClassVar[str] = "Long Term AF Database"
    dataset_id: ClassVar[str] = "ltafdb"
    sr: ClassVar[int] = 128
    entity_cls: ClassVar[type[ECGEntity]] = LTAFDBEntity


@dataclass(frozen=True, slots=True)
class SHDBAFEntity(ECGEntity):
    pass


# https://physionet.org/content/shdb-af/1.0.1/
@dataclass(slots=True)
class SHDBAF(ECGDataset):
    dir_path: ClassVar[str] = os.path.join(physionet_root_dir, "shdb-af", "1.0.1")
    name: ClassVar[str] = (
        "SHDB-AF: a Japanese Holter ECG database of atrial fibrillation"
    )
    dataset_id: ClassVar[str] = "shdb-af"
    sr: ClassVar[int] = 200
    beat_extention_priority: ClassVar[tuple[str, ...]] = ("qrs",)
    entity_cls: ClassVar[type[ECGEntity]] = SHDBAFEntity

    def __post_init__(self):
        record_path = os.path.join(self.dir_path, "RECORDS.txt")
        with open(record_path, "r") as f:
            data_ids = f.read().splitlines()

        for data_id in data_ids:
            try:
                self.data_entities.append(self._load_entity(data_id))
            except FileNotFoundError:
                continue


@dataclass(frozen=True, slots=True)
class SDDBEntity(ECGEntity):
    pass


# https://physionet.org/content/sddb/1.0.0/
@dataclass(slots=True)
class SDDB(ECGDataset):
    dir_path: ClassVar[str] = os.path.join(physionet_root_dir, "sddb", "1.0.0")
    name: ClassVar[str] = "Sudden Cardiac Death Holter Database"
    dataset_id: ClassVar[str] = "sddb"
    sr: ClassVar[int] = 250
    entity_cls: ClassVar[type[ECGEntity]] = SDDBEntity
    vf_onset_seconds: ClassVar[dict[str, int]] = {
        "30": 28473,
        "31": 49344,
        "32": 60318,
        "33": 17179,
        "34": 23744,
        "35": 88496,
        "36": 68341,
        "37": 5473,
        "38": 28914,
        "39": 16671,
        "41": 10764,
        "43": 56231,
        "44": 70725,
        "45": 65357,
        "46": 13307,
        "47": 22381,
        "48": 8980,
        "50": 42343,
        "51": 82703,
        "52": 9160,
    }


@dataclass(frozen=True, slots=True)
class VFDBEntity(ECGEntity):
    pass


# https://physionet.org/content/vfdb/1.0.0/
@dataclass(slots=True)
class VFDB(ECGDataset):
    dir_path: ClassVar[str] = os.path.join(physionet_root_dir, "vfdb", "1.0.0")
    name: ClassVar[str] = "MIT-BIH Malignant Ventricular Ectopy Database"
    dataset_id: ClassVar[str] = "vfdb"
    sr: ClassVar[int] = 250
    entity_cls: ClassVar[type[ECGEntity]] = VFDBEntity

    def __post_init__(self):
        data_ids = sorted(
            {
                filename[: -len(".hea")]
                for filename in os.listdir(self.dir_path)
                if filename.endswith(".hea")
            }
        )

        for data_id in data_ids:
            self.data_entities.append(self._load_entity(data_id))

    @classmethod
    def _load_entity(cls, data_id: str) -> ECGEntity:
        data_path = os.path.join(cls.dir_path, data_id)
        signals, _ = wfdb.rdsamp(
            data_path,
            channels=[0],
        )
        squeezed = np.squeeze(signals)

        annotation = cls._read_annotation(data_path)
        record = wfdb.rdheader(data_path)
        sr = record.fs
        beats = detect_rpeaks(squeezed, sr)

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


DATASET_CLASSES: tuple[Type[ECGDataset], ...] = (
    CUDB,
    AFPDB,
    MITDB,
    AFDB,
    LTAFDB,
    SHDBAF,
    SDDB,
    VFDB,
)


DATASET_REGISTRY: dict[str, Type[ECGDataset]] = {
    dataset_cls.dataset_id: dataset_cls for dataset_cls in DATASET_CLASSES
}


def _load_data_sources(dataset_ids: Sequence[str]) -> list[ECGDataset]:
    data_sources: list[ECGDataset] = []
    for dataset_id in dataset_ids:
        normalized_id = dataset_id.lower()
        dataset_cls = DATASET_REGISTRY.get(normalized_id)
        if dataset_cls is None:
            available_datasets = ", ".join(DATASET_REGISTRY)
            raise ValueError(
                f"Unknown dataset id '{dataset_id}'. "
                f"Available options: {available_datasets}."
            )
        data_sources.append(dataset_cls())
    return data_sources
