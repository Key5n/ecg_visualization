import os
from dataclasses import dataclass
from typing import ClassVar

from ecg_visualization.core.dataset import ECGDataset
from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.utils import (
    load_entities,
    physionet_root_dir,
)

CUDB_ENTITY_IDS: tuple[str, ...] = tuple(
    f"cu{record_id:02d}" for record_id in range(1, 36)
)


@dataclass(frozen=True, slots=True)
class CUDBEntity(ECGEntity):
    dir_path: ClassVar[str] = os.path.join(physionet_root_dir, "cudb", "1.0.0")
    source_name: ClassVar[str] = "Tachyarrythmia"
    source_dataset_id: ClassVar[str] = "cudb"


# https://physionet.org/content/cudb/1.0.0/
@dataclass(frozen=True, slots=True)
class CUDB(ECGDataset):
    dir_path: ClassVar[str] = CUDBEntity.dir_path
    name: ClassVar[str] = CUDBEntity.source_name
    dataset_id: ClassVar[str] = CUDBEntity.source_dataset_id
    sr: ClassVar[int] = 250
    entity_cls: ClassVar[type[ECGEntity]] = CUDBEntity
    data_entities: ClassVar[tuple[CUDBEntity, ...]] = load_entities(
        CUDBEntity,
        CUDB_ENTITY_IDS,
    )
