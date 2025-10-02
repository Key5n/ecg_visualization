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
    signals: npt.NDArray[np.float64]

    def __str__(self):
        return self.data_id


@dataclass
class ECG_Dataset:
    dir_path: str
    data_kind: str
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
            self.data_entities.append(ECG_Entity(data_id, self.data_kind, signals))

    def __str__(self):
        return self.data_kind


@dataclass
class CUDB(ECG_Dataset):
    dir_path: str = os.path.join(dataset_root_dir, "cudb", "1.0.0")
    data_kind: str = "Tachyarrythmia"


@dataclass
class AFPDB(ECG_Dataset):
    dir_path: str = os.path.join(dataset_root_dir, "afpdb", "1.0.0")
    data_kind: str = "PAF Prediction Challenge Database"


@dataclass
class MITDB(ECG_Dataset):
    dir_path: str = os.path.join(dataset_root_dir, "mitdb", "1.0.0")
    data_kind: str = "MIT-BIH Arrhythmia Database"


@dataclass
class AFDB(ECG_Dataset):
    dir_path: str = os.path.join(dataset_root_dir, "afdb", "1.0.0")
    data_kind: str = "MIT-BIH Atrial Fibrillation Database"


@dataclass
class LTAFDB(ECG_Dataset):
    dir_path: str = os.path.join(dataset_root_dir, "ltafdb", "1.0.0")
    data_kind: str = "Long Term AF Database"


@dataclass
class SHDBAF(ECG_Dataset):
    dir_path: str = os.path.join(dataset_root_dir, "shdb-af", "1.0.0")
    data_kind: str = "SHDB-AF: a Japanese Holter ECG database of atrial fibrillation"


@dataclass
class SDDB(ECG_Dataset):
    dir_path: str = os.path.join(dataset_root_dir, "sddb", "1.0.0")
    data_kind: str = "Sudden Cardiac Death Holter Database"
