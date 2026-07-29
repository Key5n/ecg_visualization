import os
from dataclasses import dataclass
from typing import ClassVar

from ecg_visualization.core.dataset import ECGDataset
from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.utils import physionet_root_dir


# https://physionet.org/content/afpdb/1.0.0/
@dataclass(frozen=True, slots=True)
class AFPDB(ECGDataset):
    dir_path: ClassVar[str] = os.path.join(physionet_root_dir, "afpdb", "1.0.0")
    name: ClassVar[str] = "PAF Prediction Challenge Database"
    dataset_id: ClassVar[str] = "afpdb"
    sampling_rate_hz: ClassVar[int] = 128
    annotation_extention: ClassVar[str] = "qrs"
    beat_extention: ClassVar[str] = "qrs"


@dataclass(frozen=True, slots=True)
class AFPDBEntity(ECGEntity):
    dataset: ClassVar[type[AFPDB]] = AFPDB


AFPDB.entity_cls = AFPDBEntity
