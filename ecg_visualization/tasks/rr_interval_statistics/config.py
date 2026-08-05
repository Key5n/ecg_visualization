from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ecg_visualization.datasets.physionet import DATASET_CLASSES
from ecg_visualization.tasks.config import load_task_config


@dataclass(slots=True)
class RRIntervalStatisticsConfig:
    dataset_ids: tuple[str, ...] = tuple(cls.dataset_id for cls in DATASET_CLASSES)
    output_path: Path = Path("result/rr_interval_statistics/rr_interval_statistics.pdf")


def load_rr_interval_statistics_config() -> RRIntervalStatisticsConfig:
    return load_task_config(RRIntervalStatisticsConfig())
