from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ecg_visualization.datasets.physionet import SDDB
from ecg_visualization.models.md_rs.md_rs import MDRSConfig

DEFAULT_MD_RS_CONFIG: Final[MDRSConfig] = MDRSConfig(
    N_x=256,
    input_scale=1.0,
    rho=0.9,
    leaking_rate=0.9,
    delta=1e-3,
    trans_length=10,
    N_x_tilde=256,
    seed=0,
)

WINDOW_SIZE = 10
OUTPUT_DIR = Path("result") / "sddb_concat" / "mdrs_scores"
VISUALIZE_OUTPUT_DIR = Path("result") / "sddb_concat" / "visualize"
SEGMENT_DURATION_SEC = 10 * 60
MAX_REASONABLE_RR_INTERVAL_SEC = 3.0
SINUS_RR_MEDIAN_THRESHOLD_SEC = 0.1
VF_ONSET_SECONDS = SDDB.vf_onset_seconds
SEGMENT_COLORS = {
    "sinus_train": "#2a9d8f",
    "pre_vf": "#f4a261",
    "vf": "#e63946",
    "sinus_test": "#264653",
}


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


def build_fixed_vf_windows(entity_id: str) -> tuple[SegmentWindow, SegmentWindow]:
    vf_onset_sec = VF_ONSET_SECONDS.get(entity_id)
    if vf_onset_sec is None:
        raise ValueError(f"VF onset is not configured for entity '{entity_id}'.")

    return (
        SegmentWindow(
            vf_onset_sec - SEGMENT_DURATION_SEC,
            vf_onset_sec,
        ),
        SegmentWindow(
            vf_onset_sec,
            vf_onset_sec + SEGMENT_DURATION_SEC,
        ),
    )


def build_segments_info(
    entity_id: str,
    train: SegmentWindow,
    test: SegmentWindow,
) -> SegmentsInfo:
    pre_vf, vf = build_fixed_vf_windows(entity_id)
    return SegmentsInfo(
        entity_id=entity_id,
        train=train,
        test=test,
        pre_vf=pre_vf,
        vf=vf,
    )
