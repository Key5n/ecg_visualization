from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ecg_visualization.tasks.anomaly_detection_example.config import (
    AnomalyDetectionExampleConfig,
)
from ecg_visualization.tasks.config import load_task_config


@dataclass(slots=True)
class DatasetBatchConfig:
    dataset_id: str
    pre_ar_duration_sec: float


def _default_datasets() -> tuple[DatasetBatchConfig, ...]:
    return (
        DatasetBatchConfig("ltafdb", 600.0),
        DatasetBatchConfig("sddb", 60.0),
    )


@dataclass(slots=True)
class AllAnomalyDetectionScoresConfig:
    output_path: Path = Path("result/all_anomaly_detection_scores.pdf")
    datasets: tuple[DatasetBatchConfig, ...] = field(default_factory=_default_datasets)
    model: AnomalyDetectionExampleConfig = field(
        default_factory=AnomalyDetectionExampleConfig
    )


def load_all_anomaly_detection_scores_config() -> AllAnomalyDetectionScoresConfig:
    return load_task_config(AllAnomalyDetectionScoresConfig())
