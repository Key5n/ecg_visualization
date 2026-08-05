from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ecg_visualization.models.md_rs.md_rs import MDRSConfig
from ecg_visualization.tasks.config import load_task_config
from ecg_visualization.tasks.rhythm_event_sequences.config import SinusExtractionConfig


@dataclass(slots=True)
class ExampleRecordConfig:
    dataset_id: str
    entity_id: str
    pre_ar_duration_sec: float


def _default_records() -> tuple[ExampleRecordConfig, ...]:
    return (
        ExampleRecordConfig("ltafdb", "42", 600.0),
        ExampleRecordConfig("sddb", "43", 60.0),
    )


@dataclass(slots=True)
class AnomalyDetectionExampleConfig:
    output_path: Path = Path("result/figure6_anomaly_detection_example.pdf")
    window_size: int = 10
    threshold_scale: float = 10.0
    records: tuple[ExampleRecordConfig, ...] = field(default_factory=_default_records)
    sinus_extraction: SinusExtractionConfig = field(
        default_factory=SinusExtractionConfig
    )
    segment_colors: dict[str, str] = field(
        default_factory=lambda: {
            "sinus_train": "#2a9d8f",
            "pre_ar": "#f4a261",
            "ar": "#e63946",
            "sinus_test": "#264653",
        }
    )
    mdrs: MDRSConfig = field(default_factory=MDRSConfig)


def load_anomaly_detection_example_config() -> AnomalyDetectionExampleConfig:
    return load_task_config(AnomalyDetectionExampleConfig())
