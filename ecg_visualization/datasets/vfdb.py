import os
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from wfdb.io import Annotation

from ecg_visualization.core.dataset import ECGDataset
from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.utils import (
    load_entities,
    physionet_root_dir,
)
from ecg_visualization.utils.signal_processing.rpeak_detection import detect_rpeaks

VFDB_ENTITY_IDS: tuple[str, ...] = (
    "418",
    "419",
    "420",
    "421",
    "422",
    "423",
    "424",
    "425",
    "426",
    "427",
    "428",
    "429",
    "430",
    "602",
    "605",
    "607",
    "609",
    "610",
    "611",
    "612",
    "614",
    "615",
)


@dataclass(frozen=True, slots=True)
class VFDBEntity(ECGEntity):
    dir_path: ClassVar[str] = os.path.join(physionet_root_dir, "vfdb", "1.0.0")
    source_name: ClassVar[str] = "MIT-BIH Malignant Ventricular Ectopy Database"
    source_dataset_id: ClassVar[str] = "vfdb"

    @classmethod
    def _read_beats(
        cls,
        data_path: str,
        signals: npt.NDArray[np.float64],
        annotation: Annotation,
        sr: int,
    ) -> npt.NDArray[np.int_]:
        return detect_rpeaks(signals, sr)


# https://physionet.org/content/vfdb/1.0.0/
@dataclass(frozen=True, slots=True)
class VFDB(ECGDataset):
    dir_path: ClassVar[str] = VFDBEntity.dir_path
    name: ClassVar[str] = VFDBEntity.source_name
    dataset_id: ClassVar[str] = VFDBEntity.source_dataset_id
    sr: ClassVar[int] = 250
    entity_cls: ClassVar[type[ECGEntity]] = VFDBEntity
    data_entities: ClassVar[tuple[VFDBEntity, ...]] = load_entities(
        VFDBEntity,
        VFDB_ENTITY_IDS,
    )
