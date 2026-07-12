from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

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
OUTPUT_DIR = Path("result") / "ltafdb_concat" / "mdrs_scores"
VISUALIZE_OUTPUT_DIR = Path("result") / "ltafdb_concat" / "visualize"
SEGMENT_DURATION_SEC = 10 * 60
MAX_REASONABLE_RR_INTERVAL_SEC = 3.0
SEGMENT_COLORS = {
    "sinus_train": "#2a9d8f",
    "pre_af": "#f4a261",
    "af": "#e63946",
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
    pre_af: SegmentWindow
    af: SegmentWindow


def build_segments_info(
    entity_id: str,
    train: SegmentWindow,
    test: SegmentWindow,
    pre_af: SegmentWindow,
    af: SegmentWindow,
) -> SegmentsInfo:
    return SegmentsInfo(
        entity_id=entity_id,
        train=train,
        test=test,
        pre_af=pre_af,
        af=af,
    )
