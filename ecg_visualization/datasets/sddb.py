import os
from dataclasses import dataclass
from typing import ClassVar

from ecg_visualization.core.dataset import ECGDataset
from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.utils import (
    load_entities,
    physionet_root_dir,
)

SDDB_ENTITY_IDS: tuple[str, ...] = tuple(str(record_id) for record_id in range(30, 53))


@dataclass(frozen=True, slots=True)
class SDDBEntity(ECGEntity):
    dir_path: ClassVar[str] = os.path.join(physionet_root_dir, "sddb", "1.0.0")
    source_name: ClassVar[str] = "Sudden Cardiac Death Holter Database"
    source_dataset_id: ClassVar[str] = "sddb"


# https://physionet.org/content/sddb/1.0.0/
@dataclass(frozen=True, slots=True)
class SDDB(ECGDataset):
    dir_path: ClassVar[str] = SDDBEntity.dir_path
    name: ClassVar[str] = SDDBEntity.source_name
    dataset_id: ClassVar[str] = SDDBEntity.source_dataset_id
    sr: ClassVar[int] = 250
    entity_cls: ClassVar[type[ECGEntity]] = SDDBEntity
    data_entities: ClassVar[tuple[SDDBEntity, ...]] = load_entities(
        SDDBEntity,
        SDDB_ENTITY_IDS,
    )
    vf_onset_seconds: ClassVar[dict[str, int]] = {
        "30": 28473,
        "31": 49344,
        "32": 60318,
        "33": 17179,
        "34": 23744,
        "35": 88496,
        "36": 68341,
        "37": 5473,
        "38": 28914,
        "39": 16671,
        "41": 10764,
        "43": 56231,
        "44": 70725,
        "45": 65357,
        "46": 13307,
        "47": 22381,
        "48": 8980,
        "50": 42343,
        "51": 82703,
        "52": 9160,
    }
