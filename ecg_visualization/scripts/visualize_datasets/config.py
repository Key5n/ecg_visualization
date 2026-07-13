from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ecg_visualization.scripts.config import load_task_config
from ecg_visualization.visualization.layouts import PaginationConfig


@dataclass(frozen=True, slots=True)
class VisualizeDatasetsConfig:
    dataset_ids: tuple[str, ...] = ()
    output_dir: Path = Path("result/visualize-datasets")
    pagination: PaginationConfig = PaginationConfig(seconds_per_row=10, rows_per_page=6)
    signal_ylim_lower: float = -5.0
    signal_ylim_upper: float = 5.0


def load_config() -> VisualizeDatasetsConfig:
    return load_task_config(
        VisualizeDatasetsConfig(),
    )
