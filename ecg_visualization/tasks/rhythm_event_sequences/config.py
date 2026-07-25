from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, cast

from omegaconf import OmegaConf

from ecg_visualization.models.md_rs.md_rs import MDRSConfig
from ecg_visualization.visualization.layouts import PaginationConfig


def _generate_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d/%H-%M-%S")


@dataclass(slots=True)
class SegmentWindow:
    start_sec: float
    end_sec: float

    def __post_init__(self) -> None:
        if self.end_sec <= self.start_sec:
            raise ValueError("segment window end_sec must be greater than start_sec")

        if self.start_sec < 0:
            raise ValueError("segment window start_sec must be non-negative")


@dataclass(slots=True)
class SegmentsInfo:
    entity_id: str
    train: SegmentWindow
    test: SegmentWindow
    pre_ar: SegmentWindow
    ar: SegmentWindow


@dataclass(slots=True)
class RRHistogramConfig:
    xmin_sec: float = 0.0
    xmax_sec: float = 2.0
    bin_width_sec: float = 0.025


@dataclass(slots=True)
class RhythmEventSequencesConfig:
    dataset_id: str
    root_dir: Path = Path("result") / "rhythm_event_sequences"
    run_id: str = field(default_factory=_generate_run_id)
    window_size: int = 10
    pre_ar_duration_sec: float = 10 * 60
    ar_duration_sec: float = 10 * 60
    max_reasonable_rr_interval_sec: float = 3.0
    sinus_rr_median_threshold_sec: float = 0.1
    segment_colors: dict[str, str] = field(
        default_factory=lambda: {
            "sinus_train": "#2a9d8f",
            "pre_ar": "#f4a261",
            "ar": "#e63946",
            "sinus_test": "#264653",
        }
    )
    segment_labels: dict[str, str] = field(
        default_factory=lambda: {
            "sinus_train": "sinus_train",
            "pre_ar": "pre_ar",
            "ar": "ar",
            "sinus_test": "sinus_test",
        }
    )
    model: MDRSConfig = field(default_factory=MDRSConfig)
    pagination: PaginationConfig = field(default_factory=PaginationConfig)
    rr_histogram: RRHistogramConfig = field(default_factory=RRHistogramConfig)

    @property
    def output_dir(self) -> Path:
        return self.root_dir / self.dataset_id / "outputs" / Path(self.run_id)

    @property
    def score_output_dir(self) -> Path:
        return self.output_dir / "mdrs_scores"

    @property
    def visualize_output_dir(self) -> Path:
        return self.output_dir / "visualize"

    @property
    def config_path(self) -> Path:
        return self.output_dir / "config.yaml"


def ltafdb_rhythm_event_sequences_config() -> RhythmEventSequencesConfig:
    return RhythmEventSequencesConfig(
        dataset_id="ltafdb",
        pre_ar_duration_sec=60,
        ar_duration_sec=60,
        segment_labels={
            "sinus_train": "sinus_train",
            "pre_ar": "pre_af",
            "ar": "af",
            "sinus_test": "sinus_test",
        },
    )


def sddb_rhythm_event_sequences_config() -> RhythmEventSequencesConfig:
    return RhythmEventSequencesConfig(
        dataset_id="sddb",
        pre_ar_duration_sec=60,
        ar_duration_sec=30,
        segment_labels={
            "sinus_train": "sinus_train",
            "pre_ar": "pre_vf",
            "ar": "vf",
            "sinus_test": "sinus_test",
        },
    )


def vfdb_rhythm_event_sequences_config() -> RhythmEventSequencesConfig:
    return RhythmEventSequencesConfig(
        dataset_id="vfdb",
        pre_ar_duration_sec=60,
        ar_duration_sec=10,
        segment_labels={
            "sinus_train": "sinus_train",
            "pre_ar": "pre_vf",
            "ar": "vf",
            "sinus_test": "sinus_test",
        },
    )


RHYTHM_EVENT_SEQUENCES_CONFIGS: dict[str, Callable[[], RhythmEventSequencesConfig]] = {
    "ltafdb": ltafdb_rhythm_event_sequences_config,
    "sddb": sddb_rhythm_event_sequences_config,
    "vfdb": vfdb_rhythm_event_sequences_config,
}


def load_rhythm_event_sequences_config() -> RhythmEventSequencesConfig:
    cli_config = OmegaConf.from_cli()
    config_name = str(cli_config.pop("config_name", "sddb")).lower()
    config_factory = RHYTHM_EVENT_SEQUENCES_CONFIGS.get(config_name)
    if config_factory is None:
        available_configs = ", ".join(RHYTHM_EVENT_SEQUENCES_CONFIGS)
        raise ValueError(
            f"Unknown rhythm_event_sequences config '{config_name}'. "
            f"Available options: {available_configs}."
        )

    structured = OmegaConf.structured(config_factory(), flags={"allow_objects": True})
    merged = OmegaConf.merge(structured, cli_config)
    return cast(RhythmEventSequencesConfig, OmegaConf.to_object(merged))
