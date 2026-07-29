import os
from dataclasses import dataclass
from typing import ClassVar

from ecg_visualization.core.dataset import ECGDataset
from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.utils import physionet_root_dir


# https://physionet.org/content/cudb/1.0.0/
@dataclass(frozen=True, slots=True)
class CUDB(ECGDataset):
    dir_path: ClassVar[str] = os.path.join(physionet_root_dir, "cudb", "1.0.0")
    name: ClassVar[str] = "Tachyarrythmia"
    dataset_id: ClassVar[str] = "cudb"
    sampling_rate_hz: ClassVar[int] = 250


@dataclass(frozen=True, slots=True)
class CUDBEntity(ECGEntity):
    dataset: ClassVar[type[CUDB]] = CUDB


CUDB.entity_cls = CUDBEntity
