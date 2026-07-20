import os
from dataclasses import dataclass
from typing import ClassVar

from ecg_visualization.core.dataset import ECGDataset
from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.utils import physionet_root_dir

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


# https://physionet.org/content/afpdb/1.0.0/
@dataclass(frozen=True, slots=True)
class AFPDB(ECGDataset):
    dir_path: ClassVar[str] = os.path.join(physionet_root_dir, "afpdb", "1.0.0")
    name: ClassVar[str] = "PAF Prediction Challenge Database"
    dataset_id: ClassVar[str] = "afpdb"
    sampling_rate_hz: ClassVar[int] = 128
    entity_ids: ClassVar[tuple[str, ...]] = AFPDB_ENTITY_IDS


@dataclass(frozen=True, slots=True)
class AFPDBEntity(ECGEntity):
    dataset: ClassVar[type[AFPDB]] = AFPDB


AFPDB.entity_cls = AFPDBEntity
