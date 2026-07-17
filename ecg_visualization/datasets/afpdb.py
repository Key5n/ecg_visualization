import os
from dataclasses import dataclass
from typing import ClassVar

from ecg_visualization.core.dataset import ECGDataset
from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.utils import (
    load_entities,
    physionet_root_dir,
)

AFPDB_ENTITY_IDS: tuple[str, ...] = tuple(
    (
        f"{prefix}{record_id:02d}{suffix}"
        for prefix, stop in (("n", 50), ("p", 50))
        for record_id in range(1, stop + 1)
        for suffix in ("", "c")
    )
) + (
    tuple(f"t{record_id:02d}" for record_id in range(1, 11))
    + ("t100",)
    + tuple(f"t{record_id:02d}" for record_id in range(11, 100))
)


@dataclass(frozen=True, slots=True)
class AFPDBEntity(ECGEntity):
    dir_path: ClassVar[str] = os.path.join(physionet_root_dir, "afpdb", "1.0.0")
    source_name: ClassVar[str] = "PAF Prediction Challenge Database"
    source_dataset_id: ClassVar[str] = "afpdb"


# https://physionet.org/content/afpdb/1.0.0/
@dataclass(frozen=True, slots=True)
class AFPDB(ECGDataset):
    dir_path: ClassVar[str] = AFPDBEntity.dir_path
    name: ClassVar[str] = AFPDBEntity.source_name
    dataset_id: ClassVar[str] = AFPDBEntity.source_dataset_id
    sr: ClassVar[int] = 128
    entity_cls: ClassVar[type[ECGEntity]] = AFPDBEntity
    data_entities: ClassVar[tuple[AFPDBEntity, ...]] = load_entities(
        AFPDBEntity,
        AFPDB_ENTITY_IDS,
    )
