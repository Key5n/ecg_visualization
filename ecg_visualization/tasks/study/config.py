from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ecg_visualization.core.analysis import NormalSegmentConfig
from ecg_visualization.models.md_rs.md_rs import MDRSConfig
from ecg_visualization.tasks.config import load_task_config


@dataclass(slots=True)
class StudyConfig:
    dataset_ids: tuple[str, ...] = ("sddb",)
    artifact_root: Path = Path("result/artifacts")
    n_trials: int = 1
    window_size: int = 10
    normal_segment: NormalSegmentConfig = field(default_factory=NormalSegmentConfig)
    model: MDRSConfig = field(default_factory=MDRSConfig)


def load_study_config() -> StudyConfig:
    return load_task_config(
        StudyConfig(),
    )
