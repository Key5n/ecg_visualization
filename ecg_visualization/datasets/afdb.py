import os
from dataclasses import dataclass
from typing import ClassVar

from ecg_visualization.core.dataset import ECGDataset
from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.utils import physionet_root_dir


# https://physionet.org/content/afdb/1.0.0/
@dataclass(frozen=True, slots=True)
class AFDB(ECGDataset):
    dir_path: ClassVar[str] = os.path.join(physionet_root_dir, "afdb", "1.0.0")
    name: ClassVar[str] = "MIT-BIH Atrial Fibrillation Database"
    dataset_id: ClassVar[str] = "afdb"
    sampling_rate_hz: ClassVar[int] = 250


@dataclass(frozen=True, slots=True)
class AFDBEntity(ECGEntity):
    dataset: ClassVar[type[AFDB]] = AFDB


AFDB.entity_cls = AFDBEntity
