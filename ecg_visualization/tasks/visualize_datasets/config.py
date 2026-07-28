from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ecg_visualization.tasks.config import load_task_config
from ecg_visualization.visualization.layouts import PaginationConfig


@dataclass(slots=True)
class VisualizeDatasetsConfig:
    dataset_ids: tuple[str, ...] = ()
    output_dir: Path = Path("result/visualize-datasets")
    max_workers: int | None = None
    pagination: PaginationConfig = field(default_factory=PaginationConfig)
    signal_ylim_lower: float = -5.0
    signal_ylim_upper: float = 5.0


def load_visualize_datasets_config() -> VisualizeDatasetsConfig:
    return load_task_config(
        VisualizeDatasetsConfig(),
    )
