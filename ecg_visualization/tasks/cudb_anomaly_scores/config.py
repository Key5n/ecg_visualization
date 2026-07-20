from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ecg_visualization.models.md_rs.md_rs import MDRSConfig
from ecg_visualization.tasks.config import load_task_config


@dataclass(slots=True)
class SinusWindowConfig:
    start_sec: float = 0.0
    end_sec: float = 60.0


@dataclass(slots=True)
class SinusDurationConfig:
    entity_id: str = ""
    windows: tuple[SinusWindowConfig, ...] = ()


def _default_sinus_durations() -> tuple[SinusDurationConfig, ...]:
    return tuple(
        SinusDurationConfig(
            entity_id=entity_id,
            windows=(SinusWindowConfig(start_sec=0.0, end_sec=60.0),),
        )
        for entity_id in (
            "cu01",
            "cu06",
            "cu07",
            "cu10",
            "cu11",
            "cu12",
            "cu13",
            "cu14",
            "cu15",
            "cu16",
            "cu17",
            "cu18",
            "cu19",
            "cu20",
            "cu22",
            "cu23",
            "cu24",
            "cu25",
            "cu26",
            "cu27",
            "cu29",
            "cu32",
            "cu33",
            "cu34",
        )
    )


@dataclass(slots=True)
class CudbAnomalyScoresConfig:
    output_path: Path = Path("result/cudb/anomaly_scores.pdf")
    window_size: int = 10
    model: MDRSConfig = field(default_factory=MDRSConfig)
    sinus_durations: tuple[SinusDurationConfig, ...] = field(
        default_factory=_default_sinus_durations
    )


def load_cudb_anomaly_scores_config() -> CudbAnomalyScoresConfig:
    return load_task_config(
        CudbAnomalyScoresConfig(),
    )
