import os
from dataclasses import dataclass
from typing import ClassVar

from ecg_visualization.core.dataset import ECGDataset
from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.utils import physionet_root_dir

AFDB_ENTITY_IDS: tuple[str, ...] = (
    "04015",
    "04043",
    "04048",
    "04126",
    "04746",
    "04908",
    "04936",
    "05091",
    "05121",
    "05261",
    "06426",
    "06453",
    "06995",
    "07162",
    "07859",
    "07879",
    "07910",
    "08215",
    "08219",
    "08378",
    "08405",
    "08434",
    "08455",
)


@dataclass(frozen=True, slots=True)
class AFDBEntity(ECGEntity):
    pass


# https://physionet.org/content/afdb/1.0.0/
@dataclass(frozen=True, slots=True)
class AFDB(ECGDataset):
    dir_path: ClassVar[str] = os.path.join(physionet_root_dir, "afdb", "1.0.0")
    name: ClassVar[str] = "MIT-BIH Atrial Fibrillation Database"
    dataset_id: ClassVar[str] = "afdb"
    sr: ClassVar[int] = 250
    entity_cls: ClassVar[type[ECGEntity]] = AFDBEntity
    entity_ids: ClassVar[tuple[str, ...]] = AFDB_ENTITY_IDS
