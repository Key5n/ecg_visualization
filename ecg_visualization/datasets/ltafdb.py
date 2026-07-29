import os
from dataclasses import dataclass
from typing import ClassVar

from ecg_visualization.core.dataset import ECGDataset
from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.utils import physionet_root_dir


# https://physionet.org/content/ltafdb/1.0.0/
@dataclass(frozen=True, slots=True)
class LTAFDB(ECGDataset):
    dir_path: ClassVar[str] = os.path.join(physionet_root_dir, "ltafdb", "1.0.0")
    name: ClassVar[str] = "Long Term AF Database"
    dataset_id: ClassVar[str] = "ltafdb"
    sampling_rate_hz: ClassVar[int] = 128


@dataclass(frozen=True, slots=True)
class LTAFDBEntity(ECGEntity):
    dataset: ClassVar[type[LTAFDB]] = LTAFDB


LTAFDB.entity_cls = LTAFDBEntity
