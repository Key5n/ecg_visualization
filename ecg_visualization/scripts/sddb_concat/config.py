from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ecg_visualization.datasets.physionet import SDDB
from ecg_visualization.models.md_rs.md_rs import MDRSConfig
from ecg_visualization.scripts.config import load_task_config
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
class SddbConcatConfig:
    score_output_dir: Path = Path("result") / "sddb_concat" / "mdrs_scores"
    visualize_output_dir: Path = Path("result") / "sddb_concat" / "visualize"
    window_size: int = 10
    segment_duration_sec: float = 10 * 60
    max_reasonable_rr_interval_sec: float = 3.0
    sinus_rr_median_threshold_sec: float = 0.1
    vf_onset_seconds: dict[str, float] = field(
        default_factory=lambda: dict(SDDB.vf_onset_seconds)
    )
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


def load_sddb_concat_config() -> SddbConcatConfig:
    return load_task_config(
        SddbConcatConfig(),
    )
