import os
from dataclasses import dataclass
from typing import ClassVar

from ecg_visualization.core.dataset import ECGDataset
from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.utils import physionet_root_dir


# https://physionet.org/content/shdb-af/1.0.1/
@dataclass(frozen=True, slots=True)
class SHDBAF(ECGDataset):
    dir_path: ClassVar[str] = os.path.join(physionet_root_dir, "shdb-af", "1.0.1")
    name: ClassVar[str] = (
        "SHDB-AF: a Japanese Holter ECG database of atrial fibrillation"
    )
    dataset_id: ClassVar[str] = "shdb-af"
    sampling_rate_hz: ClassVar[int] = 200
    records_file_name: ClassVar[str] = "RECORDS.txt"
    beat_extention: ClassVar[str] = "qrs"


@dataclass(frozen=True, slots=True)
class SHDBAFEntity(ECGEntity):
    dataset: ClassVar[type[SHDBAF]] = SHDBAF


SHDBAF.entity_cls = SHDBAFEntity
