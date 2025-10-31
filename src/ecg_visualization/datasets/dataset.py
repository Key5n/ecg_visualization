from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
import os

import wfdb
from wfdb.io import Annotation

dataset_root_dir = os.path.join("physionet.org", "files")


@dataclass
class ECG_Entity:
    data_id: str
    data_kind: str
    sr: int
    signals: npt.NDArray[np.float64]
    annotation: Annotation
    beats: npt.NDArray[np.int_]

    def __str__(self):
        return self.data_id


@dataclass
class ECG_Dataset:
    dir_path: str
    name: str
    dataset_id: str
    annotation_extention_priority: list[str] = field(
        default_factory=lambda: ["atr", "ari"]
    )
    beat_extention_priority: list[str] = field(default_factory=lambda: ["atr", "qrs"])
    data_ids: list[str] = field(init=False)
    data_entities: list[ECG_Entity] = field(default_factory=list)

    def __post_init__(self):
        record_path = os.path.join(self.dir_path, "RECORDS")
        with open(record_path, "r") as f:
            self.data_ids = f.read().splitlines()

        for data_id in self.data_ids:
            self.data_entities.append(self._load_entity(data_id))

    def _read_annotation(self, data_path: str) -> Annotation:
        for ext in self.annotation_extention_priority:
            annotation_file = f"{data_path}.{ext}"
            if os.path.isfile(annotation_file):
                annotation = wfdb.rdann(data_path, ext)
                return annotation

    def _read_normal_beats(self, data_path: str) -> npt.NDArray[np.int_]:
        for ext in self.beat_extention_priority:
            annotation_file = f"{data_path}.{ext}"
            if os.path.isfile(annotation_file):
                annotation = self._read_annotation(data_path)
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

    def _load_entity(self, data_id: str) -> ECG_Entity:
        data_path = os.path.join(self.dir_path, data_id)
        signals, _ = wfdb.rdsamp(
            data_path,
            channels=[0],
        )
        squeezed = np.squeeze(signals)

        annotation = self._read_annotation(data_path)
        beats = self._read_normal_beats(data_path)

        record = wfdb.rdheader(data_path)
        sr = record.fs
        return ECG_Entity(data_id, self.name, sr, squeezed, annotation, beats)

    def __str__(self):
        return self.name


# https://physionet.org/content/cudb/1.0.0/
@dataclass
class CUDB(ECG_Dataset):
    dir_path: str = os.path.join(dataset_root_dir, "cudb", "1.0.0")
    name: str = "Tachyarrythmia"
    dataset_id: str = "cudb"
    sr: int = 250


# https://physionet.org/content/afpdb/1.0.0/
@dataclass
class AFPDB(ECG_Dataset):
    dir_path: str = os.path.join(dataset_root_dir, "afpdb", "1.0.0")
    name: str = "PAF Prediction Challenge Database"
    dataset_id: str = "afpdb"
    annotation_extention: str = "qrs"
    sr: int = 128


# https://physionet.org/content/mitdb/1.0.0/
@dataclass
class MITDB(ECG_Dataset):
    dir_path: str = os.path.join(dataset_root_dir, "mitdb", "1.0.0")
    name: str = "MIT-BIH Arrhythmia Database"
    dataset_id: str = "mitdb"
    sr: int = 360


# https://physionet.org/content/afdb/1.0.0/
@dataclass
class AFDB(ECG_Dataset):
    dir_path: str = os.path.join(dataset_root_dir, "afdb", "1.0.0")
    name: str = "MIT-BIH Atrial Fibrillation Database"
    dataset_id: str = "afdb"
    sr: int = 250

    def __post_init__(self):
        record_path = os.path.join(self.dir_path, "RECORDS")
        with open(record_path, "r") as f:
            self.data_ids = f.read().splitlines()

        self.data_ids = list(
            filter(lambda data_id: not data_id in ["00735", "03665"], self.data_ids)
        )

        for data_id in self.data_ids:
            self.data_entities.append(self._load_entity(data_id))


# https://physionet.org/content/ltafdb/1.0.0/
@dataclass
class LTAFDB(ECG_Dataset):
    dir_path: str = os.path.join(dataset_root_dir, "ltafdb", "1.0.0")
    name: str = "Long Term AF Database"
    dataset_id: str = "ltafdb"
    sr: int = 128


# https://physionet.org/content/shdb-af/1.0.1/
@dataclass
class SHDBAF(ECG_Dataset):
    dir_path: str = os.path.join(dataset_root_dir, "shdb-af", "1.0.1")
    name: str = "SHDB-AF: a Japanese Holter ECG database of atrial fibrillation"
    dataset_id: str = "shdb-af"
    sr: int = 200

    def __post_init__(self):
        record_path = os.path.join(self.dir_path, "RECORDS.txt")
        with open(record_path, "r") as f:
            self.data_ids = f.read().splitlines()

        for data_id in self.data_ids:
            try:
                self.data_entities.append(self._load_entity(data_id))
            except FileNotFoundError:
                continue


# https://physionet.org/content/sddb/1.0.0/
@dataclass
class SDDB(ECG_Dataset):
    dir_path: str = os.path.join(dataset_root_dir, "sddb", "1.0.0")
    name: str = "Sudden Cardiac Death Holter Database"
    dataset_id: str = "sddb"
    beat_extention_priority: list[str] = field(default_factory=lambda: ["atr", "ari"])
    sr: int = 250
