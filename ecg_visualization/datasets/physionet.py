from typing import Sequence

from ecg_visualization.core.dataset import ECGDataset
from ecg_visualization.datasets.afdb import AFDB
from ecg_visualization.datasets.afpdb import AFPDB
from ecg_visualization.datasets.cudb import CUDB
from ecg_visualization.datasets.ltafdb import LTAFDB
from ecg_visualization.datasets.mitdb import MITDB
from ecg_visualization.datasets.sddb import SDDB
from ecg_visualization.datasets.shdbaf import SHDBAF
from ecg_visualization.datasets.vfdb import VFDB

DATASET_CLASSES: tuple[type[ECGDataset], ...] = (
    CUDB,
    AFPDB,
    MITDB,
    AFDB,
    LTAFDB,
    SHDBAF,
    SDDB,
    VFDB,
)


DATASET_REGISTRY: dict[str, type[ECGDataset]] = {
    dataset_cls.dataset_id: dataset_cls for dataset_cls in DATASET_CLASSES
}


def _load_data_sources(dataset_ids: Sequence[str]) -> list[ECGDataset]:
    data_sources: list[ECGDataset] = []
    for dataset_id in dataset_ids:
        normalized_id = dataset_id.lower()
        dataset_cls = DATASET_REGISTRY.get(normalized_id)
        if dataset_cls is None:
            available_datasets = ", ".join(DATASET_REGISTRY)
            raise ValueError(
                f"Unknown dataset id '{dataset_id}'. "
                f"Available options: {available_datasets}."
            )
        data_sources.append(dataset_cls())
    return data_sources
