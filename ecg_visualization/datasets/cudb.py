import os
from dataclasses import dataclass
from typing import ClassVar

from ecg_visualization.core.dataset import ECGDataset
from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.utils import physionet_root_dir

CUDB_ENTITY_IDS: tuple[str, ...] = tuple(
    f"cu{record_id:02d}" for record_id in range(1, 36)
)


@dataclass(frozen=True, slots=True)
class CUDBEntity(ECGEntity):
    pass


# https://physionet.org/content/cudb/1.0.0/
@dataclass(frozen=True, slots=True)
class CUDB(ECGDataset):
    dir_path: ClassVar[str] = os.path.join(physionet_root_dir, "cudb", "1.0.0")
    name: ClassVar[str] = "Tachyarrythmia"
    dataset_id: ClassVar[str] = "cudb"
    sampling_rate_hz: ClassVar[int] = 250
    entity_cls: ClassVar[type[ECGEntity]] = CUDBEntity
    entity_ids: ClassVar[tuple[str, ...]] = CUDB_ENTITY_IDS
