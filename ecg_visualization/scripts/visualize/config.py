from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ecg_visualization.scripts.config import load_task_config
from ecg_visualization.visualization.layouts import PaginationConfig


@dataclass(frozen=True, slots=True)
class VisualizeConfig:
    artifact_root: Path = Path("result/artifacts")
    visualization_root: Path = Path("result/visualize")
    rr_window_beats: int = 100
    max_workers: int | None = None
    pagination: PaginationConfig = PaginationConfig()


def load_visualize_config() -> VisualizeConfig:
    return load_task_config(
        VisualizeConfig(),
    )
