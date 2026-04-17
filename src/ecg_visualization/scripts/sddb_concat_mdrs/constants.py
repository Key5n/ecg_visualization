from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ecg_visualization.datasets.dataset import SDDB

DEFAULT_MD_RS_CONFIG: dict[str, float | int] = {
    "N_x": 256,
    "input_scale": 1.0,
    "rho": 0.9,
    "leaking_rate": 0.9,
    "delta": 1e-3,
    "trans_length": 10,
    "N_x_tilde": 256,
    "seed": 0,
}

WINDOW_SIZE = 10
OUTPUT_DIR = Path("result") / "sddb_concat" / "mdrs_scores"
SEGMENT_DURATION_SEC = 10 * 60
MAX_REASONABLE_RR_INTERVAL_SEC = 3.0
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


SINUS_SEGMENTS: tuple[SegmentsInfo, ...] = (
    SegmentsInfo(
        entity_id="30",
        train=SegmentWindow(2280.0, 2280.0 + SEGMENT_DURATION_SEC),
        test=SegmentWindow(30000.0, 30000.0 + SEGMENT_DURATION_SEC),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["30"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["30"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["30"],
            VF_ONSET_SECONDS["30"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="31",
        train=SegmentWindow(600.0, 600.0 + SEGMENT_DURATION_SEC),
        test=SegmentWindow(4200.0, 4200.0 + SEGMENT_DURATION_SEC),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["31"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["31"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["31"],
            VF_ONSET_SECONDS["31"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="32",
        train=SegmentWindow(0.0, 600.0),
        test=SegmentWindow(4980.0, 4980.0 + SEGMENT_DURATION_SEC),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["32"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["32"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["32"],
            VF_ONSET_SECONDS["32"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="33",
        train=SegmentWindow(0.0, SEGMENT_DURATION_SEC),
        test=SegmentWindow(60000.0, 60000.0 + SEGMENT_DURATION_SEC),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["33"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["33"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["33"],
            VF_ONSET_SECONDS["33"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="34",
        train=SegmentWindow(0.0, 0.0 + SEGMENT_DURATION_SEC),
        test=SegmentWindow(6000.0, 6000.0 + SEGMENT_DURATION_SEC),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["34"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["34"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["34"],
            VF_ONSET_SECONDS["34"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="35",
        train=SegmentWindow(0.0, SEGMENT_DURATION_SEC),
        test=SegmentWindow(30000.0, 30000.0 + SEGMENT_DURATION_SEC),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["35"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["35"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["35"],
            VF_ONSET_SECONDS["35"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="36",
        train=SegmentWindow(0.0, SEGMENT_DURATION_SEC),
        test=SegmentWindow(30000.0, 30000.0 + SEGMENT_DURATION_SEC),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["36"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["36"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["36"],
            VF_ONSET_SECONDS["36"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="37",
        train=SegmentWindow(30000.0, 30000.0 + SEGMENT_DURATION_SEC),
        test=SegmentWindow(60000.0, 60000.0 + SEGMENT_DURATION_SEC),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["37"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["37"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["37"],
            VF_ONSET_SECONDS["37"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="38",
        train=SegmentWindow(0.0, SEGMENT_DURATION_SEC),
        test=SegmentWindow(18000.0, 18000.0 + SEGMENT_DURATION_SEC),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["38"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["38"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["38"],
            VF_ONSET_SECONDS["38"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="39",
        train=SegmentWindow(0.0, 600.0),
        test=SegmentWindow(6000.0, 6000.0 + SEGMENT_DURATION_SEC),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["39"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["39"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["39"],
            VF_ONSET_SECONDS["39"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="41",
        train=SegmentWindow(0.0, 600.0),
        test=SegmentWindow(6000.0, 6000.0 + SEGMENT_DURATION_SEC),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["41"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["41"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["41"],
            VF_ONSET_SECONDS["41"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="43",
        train=SegmentWindow(0.0, 600.0),
        test=SegmentWindow(66000.0, 66000.0 + SEGMENT_DURATION_SEC),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["43"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["43"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["43"],
            VF_ONSET_SECONDS["43"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="43",
        train=SegmentWindow(0.0, 600.0),
        test=SegmentWindow(72000.0, 72600.0),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["43"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["43"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["43"],
            VF_ONSET_SECONDS["43"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="44",
        train=SegmentWindow(21000.0, 21000.0 + SEGMENT_DURATION_SEC),
        test=SegmentWindow(30000.0, 30000.0 + SEGMENT_DURATION_SEC),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["44"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["44"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["44"],
            VF_ONSET_SECONDS["44"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="45",
        train=SegmentWindow(300.0, 300.0 + SEGMENT_DURATION_SEC),
        test=SegmentWindow(72000.0, 72600.0),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["45"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["45"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["45"],
            VF_ONSET_SECONDS["45"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="46",
        train=SegmentWindow(0.0, 600.0),
        test=SegmentWindow(2100.0, 2100.0 + SEGMENT_DURATION_SEC),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["46"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["46"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["46"],
            VF_ONSET_SECONDS["46"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="47",
        train=SegmentWindow(24000.0, 24000.0 + SEGMENT_DURATION_SEC),
        test=SegmentWindow(69720.0, 69720.0 + SEGMENT_DURATION_SEC),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["47"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["47"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["47"],
            VF_ONSET_SECONDS["47"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="48",
        train=SegmentWindow(34000.0, 34000.0 + SEGMENT_DURATION_SEC),
        test=SegmentWindow(78000.0, 78000.0 + SEGMENT_DURATION_SEC),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["48"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["48"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["48"],
            VF_ONSET_SECONDS["48"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="50",
        train=SegmentWindow(2400.0, 2400.0 + SEGMENT_DURATION_SEC),
        test=SegmentWindow(60000.0, 60000.0 + SEGMENT_DURATION_SEC),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["50"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["50"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["50"],
            VF_ONSET_SECONDS["50"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="51",
        train=SegmentWindow(2160.0, 2160.0 + SEGMENT_DURATION_SEC),
        test=SegmentWindow(21600.0, 21600.0 + SEGMENT_DURATION_SEC),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["51"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["51"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["51"],
            VF_ONSET_SECONDS["51"] + SEGMENT_DURATION_SEC,
        ),
    ),
    SegmentsInfo(
        entity_id="52",
        train=SegmentWindow(600.0, 600.0 + SEGMENT_DURATION_SEC),
        test=SegmentWindow(12000.0, 12000.0 + SEGMENT_DURATION_SEC),
        pre_vf=SegmentWindow(
            VF_ONSET_SECONDS["52"] - SEGMENT_DURATION_SEC,
            VF_ONSET_SECONDS["52"],
        ),
        vf=SegmentWindow(
            VF_ONSET_SECONDS["52"],
            VF_ONSET_SECONDS["52"] + SEGMENT_DURATION_SEC,
        ),
    ),
)
