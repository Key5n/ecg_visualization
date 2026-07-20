from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, cast

from omegaconf import OmegaConf

from ecg_visualization.models.md_rs.md_rs import MDRSConfig
from ecg_visualization.tasks.config import _set_readonly_recursive
from ecg_visualization.visualization.layouts import PaginationConfig


def _generate_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d/%H-%M-%S")


@dataclass(frozen=True, slots=True)
class SegmentWindow:
    start_sec: float
    end_sec: float


@dataclass(frozen=True, slots=True)
class SegmentsInfo:
    entity_id: str
    train: SegmentWindow
    test: SegmentWindow
    pre_vf: SegmentWindow
    vf: SegmentWindow


@dataclass(frozen=True, slots=True)
class RRHistogramConfig:
    xmin_sec: float = 0.0
    xmax_sec: float = 2.0
    bin_width_sec: float = 0.025


@dataclass(frozen=True, slots=True)
class RhythmEventSequencesConfig:
    dataset_id: str = "sddb"
    root_dir: Path = Path("result") / "rhythm_event_sequences"
    run_id: str = field(default_factory=_generate_run_id)
    window_size: int = 10
    segment_duration_sec: float = 10 * 60
    max_reasonable_rr_interval_sec: float = 3.0
    sinus_rr_median_threshold_sec: float = 0.1
    segment_colors: dict[str, str] = field(
        default_factory=lambda: {
            "sinus_train": "#2a9d8f",
            "pre_vf": "#f4a261",
            "vf": "#e63946",
            "sinus_test": "#264653",
        }
    )
    model: MDRSConfig = MDRSConfig()
    pagination: PaginationConfig = PaginationConfig()
    rr_histogram: RRHistogramConfig = RRHistogramConfig()

    @property
    def run_output_dir(self) -> Path:
        return self.root_dir / self.dataset_id / "outputs" / Path(self.run_id)

    @property
    def score_output_dir(self) -> Path:
        return self.run_output_dir / "mdrs_scores"

    @property
    def visualize_output_dir(self) -> Path:
        return self.run_output_dir / "visualize"

    @property
    def config_path(self) -> Path:
        return self.run_output_dir / "config.txt"


def ltafdb_rhythm_event_sequences_config() -> RhythmEventSequencesConfig:
    return RhythmEventSequencesConfig(
        dataset_id="ltafdb",
        segment_duration_sec=60,
    )


def sddb_rhythm_event_sequences_config() -> RhythmEventSequencesConfig:
    return RhythmEventSequencesConfig(
        dataset_id="sddb",
        segment_duration_sec=30,
    )


def vfdb_rhythm_event_sequences_config() -> RhythmEventSequencesConfig:
    return RhythmEventSequencesConfig(
        dataset_id="vfdb",
        segment_duration_sec=10,
    )


RHYTHM_EVENT_SEQUENCES_CONFIGS: dict[str, Callable[[], RhythmEventSequencesConfig]] = {
    "ltafdb": ltafdb_rhythm_event_sequences_config,
    "sddb": sddb_rhythm_event_sequences_config,
    "vfdb": vfdb_rhythm_event_sequences_config,
}


def build_fixed_vf_windows(
    entity_id: str,
    *,
    segment_duration_sec: float,
    vf_onset_seconds: dict[str, float],
) -> tuple[SegmentWindow, SegmentWindow]:
    vf_onset_sec = vf_onset_seconds.get(entity_id)
    if vf_onset_sec is None:
        raise ValueError(f"VF onset is not configured for entity '{entity_id}'.")

    return (
        SegmentWindow(
            vf_onset_sec - segment_duration_sec,
            vf_onset_sec,
        ),
        SegmentWindow(
            vf_onset_sec,
            vf_onset_sec + segment_duration_sec,
        ),
    )


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
    _set_readonly_recursive(structured, False)
    merged = OmegaConf.merge(structured, cli_config)
    return cast(RhythmEventSequencesConfig, OmegaConf.to_object(merged))
