from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ecg_visualization.tasks.config import load_task_config


@dataclass(slots=True)
class SDDBRRHistogramsConfig:
    output_path: Path = Path("result/sddb_rri_histograms/rr_interval_histograms.pdf")
    entity_ids: tuple[str, str, str, str] = ("38", "51", "46", "33")


def load_sddb_rr_histograms_config() -> SDDBRRHistogramsConfig:
    return load_task_config(SDDBRRHistogramsConfig())
