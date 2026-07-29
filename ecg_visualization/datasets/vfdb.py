import os
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ecg_visualization.core.dataset import ECGDataset
from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.utils import physionet_root_dir
from ecg_visualization.utils.signal_processing.rpeak_detection import detect_rpeaks


# https://physionet.org/content/vfdb/1.0.0/
@dataclass(frozen=True, slots=True)
class VFDB(ECGDataset):
    dir_path: ClassVar[str] = os.path.join(physionet_root_dir, "vfdb", "1.0.0")
    name: ClassVar[str] = "MIT-BIH Malignant Ventricular Ectopy Database"
    dataset_id: ClassVar[str] = "vfdb"
    sampling_rate_hz: ClassVar[int] = 250

    @classmethod
    def _read_beats(
        cls,
        data_path: str,
        signals: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.int_]:
        return detect_rpeaks(signals, cls.sampling_rate_hz)


@dataclass(frozen=True, slots=True)
class VFDBEntity(ECGEntity):
    dataset: ClassVar[type[VFDB]] = VFDB


VFDB.entity_cls = VFDBEntity
