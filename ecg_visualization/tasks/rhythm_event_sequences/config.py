from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, cast

from omegaconf import DictConfig, ListConfig, OmegaConf

from ecg_visualization.models.md_rs.md_rs import MDRSConfig
from ecg_visualization.visualization.layouts import PaginationConfig


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
    score_output_dir: Path = (
        Path("result") / "rhythm_event_sequences" / "sddb" / "mdrs_scores"
    )
    visualize_output_dir: Path = (
        Path("result") / "rhythm_event_sequences" / "sddb" / "visualize"
    )
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
    model: MDRSConfig = field(
        default_factory=lambda: MDRSConfig(
            N_x=256,
            input_scale=1.0,
            rho=0.9,
            leaking_rate=0.9,
            delta=1e-3,
            trans_length=10,
            N_x_tilde=256,
            seed=0,
        )
    )
    pagination: PaginationConfig = PaginationConfig(seconds_per_row=10, rows_per_page=6)
    rr_histogram: RRHistogramConfig = RRHistogramConfig()


def ltafdb_rhythm_event_sequences_config() -> RhythmEventSequencesConfig:
    return RhythmEventSequencesConfig(
        dataset_id="ltafdb",
        visualize_output_dir=Path("result")
        / "rhythm_event_sequences"
        / "ltafdb"
        / "visualize",
        score_output_dir=Path("result")
        / "rhythm_event_sequences"
        / "ltafdb"
        / "mdrs_scores",
        segment_duration_sec=60,
    )


def sddb_rhythm_event_sequences_config() -> RhythmEventSequencesConfig:
    return RhythmEventSequencesConfig(
        dataset_id="sddb",
        visualize_output_dir=Path("result")
        / "rhythm_event_sequences"
        / "sddb"
        / "visualize",
        score_output_dir=Path("result")
        / "rhythm_event_sequences"
        / "sddb"
        / "mdrs_scores",
        segment_duration_sec=30,
    )


def vfdb_rhythm_event_sequences_config() -> RhythmEventSequencesConfig:
    return RhythmEventSequencesConfig(
        dataset_id="vfdb",
        visualize_output_dir=Path("result")
        / "rhythm_event_sequences"
        / "vfdb"
        / "visualize",
        score_output_dir=Path("result")
        / "rhythm_event_sequences"
        / "vfdb"
        / "mdrs_scores",
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


def build_segments_info(
    entity_id: str,
    train: SegmentWindow,
    test: SegmentWindow,
    *,
    segment_duration_sec: float,
    vf_onset_seconds: dict[str, float],
) -> SegmentsInfo:
    pre_vf, vf = build_fixed_vf_windows(
        entity_id,
        segment_duration_sec=segment_duration_sec,
        vf_onset_seconds=vf_onset_seconds,
    )
    return SegmentsInfo(
        entity_id=entity_id,
        train=train,
        test=test,
        pre_vf=pre_vf,
        vf=vf,
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


def _set_readonly_recursive(config: DictConfig | ListConfig, readonly: bool) -> None:
    OmegaConf.set_readonly(config, readonly)
    values = config.values() if isinstance(config, DictConfig) else config
    for value in values:
        if isinstance(value, DictConfig | ListConfig):
            _set_readonly_recursive(value, readonly)
