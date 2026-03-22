from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ecg_visualization.datasets.dataset import SDDB, ECG_Entity
from ecg_visualization.models.md_rs.md_rs import MDRS
from ecg_visualization.scripts.sddb_concat import (
    SINUS_SEGMENTS,
    ConcatenatedSequence,
    _build_concatenated_sequence,
    _build_segments_info_by_entity,
    _highlight_segments,
    _segment_windows,
)
from ecg_visualization.utils.utils import prepare_sequences, sliding_window_sequences
from ecg_visualization.visualization.styles import apply_default_style

LOGGER = logging.getLogger(__name__)

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

WINDOW_SIZE = 256
OUTPUT_DIR = Path("result") / "sddb_concat" / "mdrs_scores"


@dataclass(frozen=True, slots=True)
class ScoreResult:
    times_sec: np.ndarray
    scores: np.ndarray


def sddb_concat_mdrs_scores() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    apply_default_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = SDDB()
    segments_info_by_entity = _build_segments_info_by_entity(SINUS_SEGMENTS)

    processed = 0
    skipped = 0
    for entity in dataset.data_entities:
        segments_info = segments_info_by_entity.get(entity.entity_id)
        if segments_info is None:
            LOGGER.info("Skipping %s: no sinus segments configured.", entity.entity_id)
            skipped += 1
            continue

        concat = _build_concatenated_sequence(entity, segments_info)
        if concat is None:
            skipped += 1
            continue

        try:
            score_result = _score_concatenated_sequence(concat)
        except ValueError as exc:
            LOGGER.warning("Skipping %s: %s", entity.entity_id, exc)
            skipped += 1
            continue

        fig = _plot_concat_scores(entity, concat, score_result)
        output_path = OUTPUT_DIR / f"{entity.entity_id}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        LOGGER.info("Saved MD-RS scores to %s", output_path)
        processed += 1

    LOGGER.info(
        "Finished scoring. processed=%d skipped=%d output_dir=%s",
        processed,
        skipped,
        OUTPUT_DIR,
    )


def _score_concatenated_sequence(concat: ConcatenatedSequence) -> ScoreResult:
    beat_samples = np.asarray(concat.beats, dtype=np.int_)
    if beat_samples.size < WINDOW_SIZE + 1:
        raise ValueError("sequence does not contain enough beats")

    train_segment_samples = int(
        np.round(
            (
                concat.segments_info.train.end_sec
                - concat.segments_info.train.start_sec
            )
            * concat.sampling_rate_hz
        )
    )
    train_beats = beat_samples[beat_samples < train_segment_samples]
    if train_beats.size < WINDOW_SIZE + 1:
        raise ValueError("sinus_train segment does not contain enough beats")

    beat_times = beat_samples.astype(np.float64) / concat.sampling_rate_hz
    train_beat_times = train_beats.astype(np.float64) / concat.sampling_rate_hz
    rr_intervals = np.diff(beat_times)
    train_rr_intervals = np.diff(train_beat_times)

    train_windows = sliding_window_sequences(train_rr_intervals, WINDOW_SIZE)
    test_windows = sliding_window_sequences(rr_intervals, WINDOW_SIZE)

    train_sequence, test_sequence = prepare_sequences(train_windows, test_windows)

    config = {
        **DEFAULT_MD_RS_CONFIG,
        "N_u": train_sequence.shape[1],
    }
    model = MDRS(**config)
    model.train(train_sequence)
    model.reset_states()

    scores = model.predict(test_sequence)
    score_times = beat_times[WINDOW_SIZE:]
    return ScoreResult(times_sec=score_times, scores=scores)


def _plot_concat_scores(
    entity: ECG_Entity,
    concat: ConcatenatedSequence,
    score_result: ScoreResult,
) -> plt.Figure:
    samples = np.asarray(concat.samples, dtype=float)
    sr = float(concat.sampling_rate_hz)
    ts = np.arange(samples.size, dtype=float) / sr

    signal_min = float(np.nanmin(samples))
    signal_max = float(np.nanmax(samples))
    signal_margin = (signal_max - signal_min) * 0.05 or 1.0
    signal_ylim = (signal_min - signal_margin, signal_max + signal_margin)

    scores = np.asarray(score_result.scores, dtype=float)
    positive_scores = scores[scores > 0]
    score_min = float(np.nanmin(positive_scores)) if positive_scores.size else 1e-6
    score_max = float(np.nanmax(scores)) if scores.size else 1.0
    score_margin = min(
        (score_max - score_min) * 0.1 or score_min * 0.1 or 1e-6,
        score_min * 0.99,
    )
    score_ylim = (score_min - score_margin, score_max + score_margin)

    fig, (signal_ax, score_ax) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(12, 4.5),
        gridspec_kw={"height_ratios": [2, 1]},
    )

    signal_ax.plot(ts, samples, "-", linewidth=0.8)
    signal_ax.set_ylabel("Voltage [mV]")
    signal_ax.set_ylim(*signal_ylim)

    score_ax.plot(
        score_result.times_sec,
        scores,
        color="black",
        linewidth=1.0,
    )
    score_ax.set_ylabel("Score")
    score_ax.set_xlabel("Time (sec)")
    score_ax.set_yscale("log")
    score_ax.set_ylim(*score_ylim)

    _highlight_concat_segments(signal_ax, concat, ylim_upper=signal_ylim[1])
    _highlight_concat_segments(score_ax, concat, ylim_upper=score_ylim[1])

    signal_ax.set_title(f"ID: {entity.entity_id}")
    fig.tight_layout()
    return fig


def _highlight_concat_segments(
    ax: plt.Axes,
    concat: ConcatenatedSequence,
    *,
    ylim_upper: float,
) -> None:
    segments: list[tuple[str, float, float]] = []
    running_start = 0
    for name, window in _segment_windows(concat.segments_info):
        segment_samples = int(
            np.round((window.end_sec - window.start_sec) * concat.sampling_rate_hz)
        )
        start_sec = running_start / concat.sampling_rate_hz
        end_sec = (running_start + segment_samples) / concat.sampling_rate_hz
        segments.append((name, start_sec, end_sec))
        running_start += segment_samples

    _highlight_segments(
        ax,
        segments,
        window_start=segments[0][1],
        window_end=segments[-1][2],
        ylim_upper=ylim_upper,
    )
