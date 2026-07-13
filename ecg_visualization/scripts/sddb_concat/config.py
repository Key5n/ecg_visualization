from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ecg_visualization.models.md_rs.md_rs import MDRSConfig
from ecg_visualization.scripts.config import load_task_config
from ecg_visualization.visualization.layouts import PaginationConfig


@dataclass(frozen=True, slots=True)
class RRHistogramConfig:
    xmin_sec: float = 0.0
    xmax_sec: float = 2.0
    bin_width_sec: float = 0.025


@dataclass(frozen=True, slots=True)
class SddbConcatConfig:
    score_output_dir: Path = Path("result/sddb_concat/mdrs_scores")
    visualize_output_dir: Path = Path("result/sddb_concat/visualize")
    window_size: int = 10
    segment_duration_sec: float = 600.0
    max_reasonable_rr_interval_sec: float = 3.0
    sinus_rr_median_threshold_sec: float = 0.1
    model: MDRSConfig = MDRSConfig()
    pagination: PaginationConfig = PaginationConfig(seconds_per_row=10, rows_per_page=6)
    rr_histogram: RRHistogramConfig = RRHistogramConfig()


def load_config() -> SddbConcatConfig:
    return load_task_config(
        SddbConcatConfig(),
    )
