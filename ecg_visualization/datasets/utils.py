import os
from typing import TypeVar

from ecg_visualization.config.settings import DATASET_ROOT
from ecg_visualization.core.entity import ECGEntity

physionet_root_dir = os.path.join(DATASET_ROOT, "physionet.org", "files")
ECGEntityT = TypeVar("ECGEntityT", bound=ECGEntity)


def load_entities(
    entity_cls: type[ECGEntityT],
    entity_ids: tuple[str, ...],
) -> tuple[ECGEntityT, ...]:
    return tuple(entity_cls.load(id=entity_id) for entity_id in entity_ids)
