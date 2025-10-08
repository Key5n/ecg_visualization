from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
import os

import wfdb

dataset_root_dir = os.path.join("physionet.org", "files")


@dataclass
class ECG_Entity:
    data_id: str
    data_kind: str
    sr: int
    signals: npt.NDArray[np.float64]

    def __str__(self):
        return self.data_id


@dataclass
class ECG_Dataset:
    dir_path: str
    name: str
    dataset_id: str
    data_ids: list[str] = field(init=False)
    data_entities: list[ECG_Entity] = field(default_factory=list)

    def __post_init__(self):
        record_path = os.path.join(self.dir_path, "RECORDS")
        with open(record_path, "r") as f:
            self.data_ids = f.read().splitlines()

        for data_id in self.data_ids:
            data_path = os.path.join(self.dir_path, data_id)
            signals, _ = wfdb.rdsamp(
                data_path,
                channels=[0],
            )
            squeezed = np.squeeze(signals)

            record = wfdb.rdheader(data_path)
            sr = record.fs
            self.data_entities.append(ECG_Entity(data_id, self.name, sr, squeezed))

    def __str__(self):
        return self.name


@dataclass
class CUDB(ECG_Dataset):
    dir_path: str = os.path.join(dataset_root_dir, "cudb", "1.0.0")
    name: str = "Tachyarrythmia"
    dataset_id: str = "cudb"
    sr: int = 250


@dataclass
class AFPDB(ECG_Dataset):
    dir_path: str = os.path.join(dataset_root_dir, "afpdb", "1.0.0")
    name: str = "PAF Prediction Challenge Database"
    dataset_id: str = "afpdb"
    sr: int = 128


@dataclass
class MITDB(ECG_Dataset):
    dir_path: str = os.path.join(dataset_root_dir, "mitdb", "1.0.0")
    name: str = "MIT-BIH Arrhythmia Database"
    dataset_id: str = "mitdb"
    sr: int = 360


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
            data_path = os.path.join(self.dir_path, data_id)
            signals, _ = wfdb.rdsamp(
                data_path,
                channels=[0],
            )
            squeezed = np.squeeze(signals)

            record = wfdb.rdheader(data_path)
            sr = record.fs
            self.data_entities.append(ECG_Entity(data_id, self.name, sr, squeezed))


@dataclass
class LTAFDB(ECG_Dataset):
    dir_path: str = os.path.join(dataset_root_dir, "ltafdb", "1.0.0")
    name: str = "Long Term AF Database"
    dataset_id: str = "ltafdb"
    sr: int = 128


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
            data_path = os.path.join(self.dir_path, data_id)
            signals, _ = wfdb.rdsamp(
                data_path,
                channels=[0],
            )
            squeezed = np.squeeze(signals)

            record = wfdb.rdheader(data_path)
            sr = record.fs
            self.data_entities.append(ECG_Entity(data_id, self.name, sr, squeezed))


@dataclass
class SDDB(ECG_Dataset):
    dir_path: str = os.path.join(dataset_root_dir, "sddb", "1.0.0")
    name: str = "Sudden Cardiac Death Holter Database"
    dataset_id: str = "sddb"
    sr: int = 250
