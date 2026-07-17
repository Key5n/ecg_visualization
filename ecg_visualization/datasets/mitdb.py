import os
from dataclasses import dataclass
from typing import ClassVar

from ecg_visualization.core.dataset import ECGDataset
from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.utils import (
    load_entities,
    physionet_root_dir,
)

MITDB_ENTITY_IDS: tuple[str, ...] = (
    "100",
    "101",
    "102",
    "103",
    "104",
    "105",
    "106",
    "107",
    "108",
    "109",
    "111",
    "112",
    "113",
    "114",
    "115",
    "116",
    "117",
    "118",
    "119",
    "121",
    "122",
    "123",
    "124",
    "200",
    "201",
    "202",
    "203",
    "205",
    "207",
    "208",
    "209",
    "210",
    "212",
    "213",
    "214",
    "215",
    "217",
    "219",
    "220",
    "221",
    "222",
    "223",
    "228",
    "230",
    "231",
    "232",
    "233",
    "234",
)


@dataclass(frozen=True, slots=True)
class MITDBEntity(ECGEntity):
    dir_path: ClassVar[str] = os.path.join(physionet_root_dir, "mitdb", "1.0.0")
    source_name: ClassVar[str] = "MIT-BIH Arrhythmia Database"
    source_dataset_id: ClassVar[str] = "mitdb"


# https://physionet.org/content/mitdb/1.0.0/
@dataclass(frozen=True, slots=True)
class MITDB(ECGDataset):
    dir_path: ClassVar[str] = MITDBEntity.dir_path
    name: ClassVar[str] = MITDBEntity.source_name
    dataset_id: ClassVar[str] = MITDBEntity.source_dataset_id
    sr: ClassVar[int] = 360
    entity_cls: ClassVar[type[ECGEntity]] = MITDBEntity
    data_entities: ClassVar[tuple[MITDBEntity, ...]] = load_entities(
        MITDBEntity,
        MITDB_ENTITY_IDS,
    )
