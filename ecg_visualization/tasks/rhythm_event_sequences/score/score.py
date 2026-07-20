from __future__ import annotations

import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from ecg_visualization.logging.config import configure_root_logging
from ecg_visualization.models.md_rs.md_rs import MDRS, MDRSConfig
from ecg_visualization.tasks.config import save_config_text
from ecg_visualization.tasks.rhythm_event_sequences.config import (
    RhythmEventSequencesConfig,
)
from ecg_visualization.tasks.rhythm_event_sequences.score.plot_concat_scores import (
    _plot_concat_scores,
)
from ecg_visualization.tasks.rhythm_event_sequences.utils import (
    ConcatenatedSequence,
    iter_concatenated_sequences,
)
from ecg_visualization.utils.utils import prepare_sequences, sliding_window_sequences
from ecg_visualization.visualization.styles import apply_default_style

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScoreResult:
    times_sec: np.ndarray
    scores: np.ndarray


def rhythm_event_sequence_scores(config: RhythmEventSequencesConfig) -> None:
    configure_root_logging()
    apply_default_style()
    config.score_output_dir.mkdir(parents=True, exist_ok=True)
    save_config_text(config, config.config_path)

    processed = 0
    for entity, concat in iter_concatenated_sequences(config):
        try:
            score_result = _score_concatenated_sequence(
                concat,
                window_size=config.window_size,
                model_config=config.model,
            )
        except ValueError as exc:
            LOGGER.warning("Skipping %s: %s", entity.entity_id, exc)
            continue

        fig = _plot_concat_scores(entity, concat, score_result, config=config)
        output_path = config.score_output_dir / f"{entity.entity_id}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        LOGGER.info("Saved MD-RS scores to %s", output_path)
        processed += 1

    LOGGER.info(
        "Finished scoring. processed=%d output_dir=%s",
        processed,
        config.score_output_dir,
    )


def _score_concatenated_sequence(
    concat: ConcatenatedSequence,
    *,
    window_size: int,
    model_config: MDRSConfig,
) -> ScoreResult:
    beat_samples = np.asarray(concat.beats, dtype=np.int_)
    if beat_samples.size < window_size + 1:
        raise ValueError("sequence does not contain enough beats")

    train_segment_samples = int(
        np.round(
            (concat.segments_info.train.end_sec - concat.segments_info.train.start_sec)
            * concat.sampling_rate_hz
        )
    )
    train_beats = beat_samples[beat_samples < train_segment_samples]
    if train_beats.size < window_size + 1:
        raise ValueError("sinus_train segment does not contain enough beats")

    beat_times = beat_samples.astype(np.float64) / concat.sampling_rate_hz
    train_beat_times = train_beats.astype(np.float64) / concat.sampling_rate_hz
    rr_intervals = np.diff(beat_times)
    train_rr_intervals = np.diff(train_beat_times)

    train_windows = sliding_window_sequences(train_rr_intervals, window_size)
    test_windows = sliding_window_sequences(rr_intervals, window_size)

    train_sequence, test_sequence = prepare_sequences(train_windows, test_windows)

    model = MDRS(model_config)
    model.train(train_sequence)
    model.reset_states()

    scores = model.predict(test_sequence)
    scores[: model_config.trans_length] = np.nan
    score_times = beat_times[window_size:]
    return ScoreResult(times_sec=score_times, scores=scores)
