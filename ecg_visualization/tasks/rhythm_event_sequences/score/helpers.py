from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ecg_visualization.models.md_rs.md_rs import MDRS, MDRSConfig
from ecg_visualization.tasks.rhythm_event_sequences.utils import ConcatenatedSequence
from ecg_visualization.utils.utils import prepare_sequences, sliding_window_sequences


@dataclass(frozen=True, slots=True)
class ScoreResult:
    times_sec: npt.NDArray[np.float64]
    scores: npt.NDArray[np.float64]


def score_concatenated_sequence(
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
    # Include the beat at the train/pre-AR boundary so the final all-normal RR
    # window ending at that beat is represented during training.
    train_beats = beat_samples[beat_samples <= train_segment_samples]
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
