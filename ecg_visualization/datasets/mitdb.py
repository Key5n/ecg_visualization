import os
from dataclasses import dataclass
from typing import ClassVar

from ecg_visualization.core.dataset import ECGDataset
from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.utils import physionet_root_dir


# https://physionet.org/content/mitdb/1.0.0/
@dataclass(frozen=True, slots=True)
class MITDB(ECGDataset):
    dir_path: ClassVar[str] = os.path.join(physionet_root_dir, "mitdb", "1.0.0")
    name: ClassVar[str] = "MIT-BIH Arrhythmia Database"
    dataset_id: ClassVar[str] = "mitdb"
    sampling_rate_hz: ClassVar[int] = 360


@dataclass(frozen=True, slots=True)
class MITDBEntity(ECGEntity):
    dataset: ClassVar[type[MITDB]] = MITDB


MITDB.entity_cls = MITDBEntity
