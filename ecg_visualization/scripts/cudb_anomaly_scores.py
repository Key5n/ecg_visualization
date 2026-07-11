from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ecg_visualization.datasets.dataset import CUDB, ECGEntity
from ecg_visualization.logging.config import configure_root_logging
from ecg_visualization.models.md_rs.md_rs import MDRS
from ecg_visualization.utils.timed_sequence import TimedSequence
from ecg_visualization.utils.utils import (
    merge_overlapping_windows,
    prepare_sequences,
    sliding_window_sequences,
)
from ecg_visualization.visualization.export import pdf_exporter
from ecg_visualization.visualization.plotters import (
    highlight_windows,
    plot_anomaly_score,
    plot_signal,
)
from ecg_visualization.visualization.styles import (
    TRAINING_INTERVAL_COLOR,
    apply_default_style,
)

DEFAULT_MD_RS_CONFIG: dict[str, float | int] = {
    "N_x": 256,
    "input_scale": 0.5,
    "rho": 0.9,
    "leaking_rate": 0.3,
    "delta": 1e-3,
    "trans_length": 10,
    "N_x_tilde": 128,
    "seed": 0,
}

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SinusWindow:
    start_sec: float
    end_sec: float


@dataclass(frozen=True, slots=True)
class SinusDuration:
    entity_id: str
    windows: tuple[SinusWindow, ...]


# Define sinus windows per entity here (seconds).
# Example:
# SINUS_DURATIONS = (
#     SinusDuration(
#         entity_id="00001",
#         windows=(SinusWindow(120.0, 420.0), SinusWindow(900.0, 1200.0)),
#     ),
#     SinusDuration(
#         entity_id="00002",
#         windows=(SinusWindow(60.0, 540.0),),
#     ),
# )
SINUS_DURATIONS: tuple[SinusDuration, ...] = (
    SinusDuration(
        entity_id="cu01",
        windows=(SinusWindow(0.0, 60.0),),
    ),
    SinusDuration(
        entity_id="cu06",
        windows=(SinusWindow(0, 60.0),),
    ),
    SinusDuration(
        entity_id="cu07",
        windows=(SinusWindow(0, 60.0),),
    ),
    SinusDuration(
        entity_id="cu10",
        windows=(SinusWindow(0, 60.0),),
    ),
    SinusDuration(
        entity_id="cu11",
        windows=(SinusWindow(0, 60.0),),
    ),
    SinusDuration(
        entity_id="cu12",
        windows=(SinusWindow(0, 60.0),),
    ),
    SinusDuration(
        entity_id="cu13",
        windows=(SinusWindow(0, 60.0),),
    ),
    SinusDuration(
        entity_id="cu14",
        windows=(SinusWindow(0, 60.0),),
    ),
    SinusDuration(
        entity_id="cu15",
        windows=(SinusWindow(0, 60.0),),
    ),
    SinusDuration(
        entity_id="cu16",
        windows=(SinusWindow(0, 60.0),),
    ),
    SinusDuration(
        entity_id="cu17",
        windows=(SinusWindow(0, 60.0),),
    ),
    SinusDuration(
        entity_id="cu18",
        windows=(SinusWindow(0, 60.0),),
    ),
    SinusDuration(
        entity_id="cu19",
        windows=(SinusWindow(0, 60.0),),
    ),
    SinusDuration(
        entity_id="cu20",
        windows=(SinusWindow(0, 60.0),),
    ),
    SinusDuration(
        entity_id="cu22",
        windows=(SinusWindow(0, 60.0),),
    ),
    SinusDuration(
        entity_id="cu23",
        windows=(SinusWindow(0, 60.0),),
    ),
    SinusDuration(
        entity_id="cu24",
        windows=(SinusWindow(0, 60.0),),
    ),
    SinusDuration(
        entity_id="cu25",
        windows=(SinusWindow(0, 60.0),),
    ),
    SinusDuration(
        entity_id="cu26",
        windows=(SinusWindow(0, 60.0),),
    ),
    SinusDuration(
        entity_id="cu27",
        windows=(SinusWindow(0, 60.0),),
    ),
    SinusDuration(
        entity_id="cu29",
        windows=(SinusWindow(0, 60.0),),
    ),
    SinusDuration(
        entity_id="cu32",
        windows=(SinusWindow(0, 60.0),),
    ),
    SinusDuration(
        entity_id="cu33",
        windows=(SinusWindow(0, 60.0),),
    ),
    SinusDuration(
        entity_id="cu34",
        windows=(SinusWindow(0, 60.0),),
    ),
)


OUTPUT_PATH = Path("result/cudb/anomaly_scores.pdf")
WINDOW_SIZE = 10


def cudb_anomaly_scores() -> None:
    configure_root_logging()

    output_path = OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = DEFAULT_MD_RS_CONFIG.copy()
    apply_default_style()

    dataset = CUDB()
    processed = 0
    skipped = 0

    sinus_windows_by_entity = _build_sinus_windows_by_entity(SINUS_DURATIONS)

    with pdf_exporter(str(output_path)) as exporter:
        for entity in tqdm(dataset.data_entities, desc="CUDB"):
            sinus_windows = sinus_windows_by_entity.get(entity.entity_id)
            if not sinus_windows:
                LOGGER.warning(
                    "Skipping %s: no sinus windows provided.", entity.entity_id
                )
                skipped += 1
                continue

            try:
                score_sequence, used_windows = _score_entity(
                    entity=entity,
                    sinus_windows=sinus_windows,
                    window_size=WINDOW_SIZE,
                    model_config=config,
                )
            except ValueError as exc:
                LOGGER.warning("Skipping %s: %s", entity.entity_id, exc)
                skipped += 1
                continue

            fig = _plot_entity_scores(
                entity=entity,
                score_sequence=score_sequence,
                training_windows=used_windows,
            )
            exporter.add_page(fig)
            plt.close(fig)
            processed += 1

    LOGGER.info(
        "Finished scoring. processed=%d skipped=%d output_path=%s",
        processed,
        skipped,
        output_path,
    )


def _score_entity(
    *,
    entity: ECGEntity,
    sinus_windows: Iterable[tuple[float, float]],
    window_size: int,
    model_config: dict[str, float | int],
) -> tuple[TimedSequence[np.float64], list[tuple[float, float]]]:
    if window_size < 1:
        raise ValueError("window_size must be positive")

    signal_duration = entity.signals.size / entity.sr
    normalized_windows = _normalize_windows(sinus_windows, signal_duration)
    if not normalized_windows:
        raise ValueError("no valid sinus windows after clipping")

    beat_times = entity.beats / entity.sr
    if beat_times.size < window_size + 1:
        raise ValueError("not enough beats to score")

    rr_intervals = np.diff(beat_times)
    train_rr = _extract_rr_intervals_in_windows(
        beat_times, rr_intervals, normalized_windows
    )

    if train_rr.size < window_size:
        raise ValueError("sinus windows too short for training")
    if rr_intervals.size < window_size:
        raise ValueError("signal too short for scoring")

    train_windows = sliding_window_sequences(train_rr, window_size)
    test_windows = sliding_window_sequences(rr_intervals, window_size)

    train_sequence, test_sequence = prepare_sequences(train_windows, test_windows)

    tuned_config = {
        **model_config,
        "N_u": train_sequence.shape[1],
    }

    model = MDRS(**tuned_config)
    model.train(train_sequence)
    model.reset_states()

    scores = model.predict(test_sequence)

    score_times = beat_times[window_size:]
    score_sequence = TimedSequence(
        values=scores,
        times=score_times,
    )

    return score_sequence, normalized_windows


def _plot_entity_scores(
    *,
    entity: ECGEntity,
    score_sequence: TimedSequence[np.float64],
    training_windows: Iterable[tuple[float, float]],
) -> plt.Figure:
    ts = np.arange(entity.signals.size, dtype=float) / entity.sr
    signal = entity.signals

    signal_min = float(np.nanmin(signal))
    signal_max = float(np.nanmax(signal))
    signal_margin = (signal_max - signal_min) * 0.05 or 1.0
    signal_ylim = (signal_min - signal_margin, signal_max + signal_margin)

    score_min = float(np.nanmin(score_sequence.values))
    score_max = float(np.nanmax(score_sequence.values))
    score_margin = (score_max - score_min) * 0.1 or 1.0
    score_ylim = (score_min - score_margin, score_max + score_margin)

    fig, ax = plt.subplots(figsize=(128, 4))
    plot_signal(
        ax,
        ts,
        signal,
        ylim_lower=signal_ylim[0],
        ylim_upper=signal_ylim[1],
    )

    highlight_windows(
        ax,
        training_windows,
        window_start=float(ts[0]),
        window_end=float(ts[-1]),
        ylim_upper=signal_ylim[1],
        color=TRAINING_INTERVAL_COLOR,
    )

    score_ax = ax.twinx()
    plot_anomaly_score(
        score_ax,
        score_sequence.times.tolist(),
        score_sequence.values.tolist(),
        ylim_lower=score_ylim[0],
        ylim_upper=score_ylim[1],
        label="Anomaly Score",
    )
    beat_times = entity.beats / entity.sr
    if beat_times.size:
        ax.scatter(
            beat_times,
            np.zeros_like(beat_times),
            s=8,
            color="tab:green",
            alpha=0.8,
            label="R-peaks",
            zorder=3,
        )

    ax.set_title(f"{entity.dataset_name} {entity.entity_id}")
    ax.set_xlabel("Time (sec)")
    fig.tight_layout()
    return fig


def _normalize_windows(
    windows: Iterable[tuple[float, float]],
    signal_duration: float,
) -> list[tuple[float, float]]:
    clipped: set[tuple[float, float]] = set()
    for start_sec, end_sec in windows:
        start = max(0.0, float(start_sec))
        end = min(signal_duration, float(end_sec))
        if end > start:
            clipped.add((start, end))

    merged = merge_overlapping_windows(clipped)
    return sorted(merged)


def _build_sinus_windows_by_entity(
    durations: Iterable[SinusDuration],
) -> dict[str, list[tuple[float, float]]]:
    windows_by_entity: dict[str, list[tuple[float, float]]] = {}
    for duration in durations:
        windows = windows_by_entity.setdefault(duration.entity_id, [])
        windows.extend(
            (float(window.start_sec), float(window.end_sec))
            for window in duration.windows
        )
    return windows_by_entity


def _extract_rr_intervals_in_windows(
    beat_times: np.ndarray,
    rr_intervals: np.ndarray,
    windows: Iterable[tuple[float, float]],
) -> np.ndarray:
    rr_start_times = beat_times[:-1]
    rr_end_times = beat_times[1:]

    mask = np.zeros(rr_intervals.shape, dtype=bool)
    for start_sec, end_sec in windows:
        window_mask = (rr_start_times >= start_sec) & (rr_end_times <= end_sec)
        mask |= window_mask

    return rr_intervals[mask]
