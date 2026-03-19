from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ecg_visualization.datasets.dataset import (
    MAX_NORMAL_RR_INTERVAL_SEC,
    MIN_NORMAL_RR_INTERVAL_SEC,
    SDDB,
    ECG_Entity,
)

LOGGER = logging.getLogger(__name__)

SEGMENT_DURATION_SEC = 10 * 60
OUTPUT_DIR = Path("result") / "sddb_concat"

VF_ONSET_SECONDS = SDDB.vf_onset_seconds


def concat_sddb() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = SDDB()
    for entity in dataset.data_entities:
        entity_id = entity.entity_id
        if entity_id not in SDDB.vf_onset_seconds:
            LOGGER.info(
                "Skipping %s: no VF onset time available or marked no-VF.",
                entity_id,
            )
            continue

        concat = _build_concatenated_sequence(entity, SDDB.vf_onset_seconds[entity_id])
        if concat is None:
            continue

        output_path = OUTPUT_DIR / f"{entity_id}.npz"
        np.savez_compressed(output_path, **concat)
        LOGGER.info("Saved concatenated sequence to %s", output_path)


def _build_concatenated_sequence(
    entity: ECG_Entity,
    vf_onset_sec: int,
) -> dict[str, np.ndarray] | None:
    signal = entity.signals
    sr = float(entity.sr)
    segment_samples = int(SEGMENT_DURATION_SEC * sr)
    total_duration_sec = signal.size / sr

    pre_vf_start_sec = vf_onset_sec - SEGMENT_DURATION_SEC
    pre_vf_end_sec = vf_onset_sec
    vf_start_sec = vf_onset_sec
    vf_end_sec = vf_onset_sec + SEGMENT_DURATION_SEC

    if pre_vf_start_sec < 0:
        LOGGER.warning(
            "Skipping %s: pre-VF start is before record start.",
            entity.entity_id,
        )
        return None

    if vf_end_sec > total_duration_sec:
        LOGGER.warning(
            "Skipping %s: VF window exceeds record length (%.1fs).",
            entity.entity_id,
            total_duration_sec,
        )
        return None

    train_start_sec = _find_normal_segment_start(
        entity,
        start_bound=0.0,
        end_bound=pre_vf_start_sec,
        duration_sec=SEGMENT_DURATION_SEC,
    )
    if train_start_sec is None:
        LOGGER.warning(
            "Skipping %s: no 10-min sinus segment before pre-VF window.",
            entity.entity_id,
        )
        return None

    test_start_sec = _find_normal_segment_start(
        entity,
        start_bound=vf_end_sec,
        end_bound=total_duration_sec,
        duration_sec=SEGMENT_DURATION_SEC,
    )
    if test_start_sec is None:
        LOGGER.warning(
            "Skipping %s: no 10-min sinus segment after VF window.",
            entity.entity_id,
        )
        return None

    segments = [
        ("sinus_train", train_start_sec),
        ("pre_vf", pre_vf_start_sec),
        ("vf", vf_start_sec),
        ("sinus_test", test_start_sec),
    ]

    concatenated_samples: list[np.ndarray] = []
    segment_names: list[str] = []
    segment_start_samples: list[int] = []
    segment_end_samples: list[int] = []
    source_start_seconds: list[float] = []
    source_end_seconds: list[float] = []

    running_start = 0
    for name, start_sec in segments:
        start_sample = int(np.round(start_sec * sr))
        end_sample = start_sample + segment_samples
        if end_sample > signal.size:
            LOGGER.warning(
                "Skipping %s: %s segment exceeds record length.",
                entity.entity_id,
                name,
            )
            return None

        concatenated_samples.append(signal[start_sample:end_sample])
        segment_names.append(name)
        segment_start_samples.append(running_start)
        running_start += segment_samples
        segment_end_samples.append(running_start)
        source_start_seconds.append(start_sample / sr)
        source_end_seconds.append(end_sample / sr)

    return {
        "samples": np.concatenate(concatenated_samples),
        "sampling_rate_hz": np.array([sr], dtype=np.float64),
        "segment_names": np.asarray(segment_names, dtype=object),
        "segment_start_samples": np.asarray(segment_start_samples, dtype=np.int_),
        "segment_end_samples": np.asarray(segment_end_samples, dtype=np.int_),
        "source_start_seconds": np.asarray(source_start_seconds, dtype=np.float64),
        "source_end_seconds": np.asarray(source_end_seconds, dtype=np.float64),
    }


def _find_normal_segment_start(
    entity: ECG_Entity,
    *,
    start_bound: float,
    end_bound: float,
    duration_sec: float,
) -> float | None:
    beat_times = entity.beats.astype(np.float64) / entity.sr
    if beat_times.size < 2:
        return None

    rr_intervals = np.diff(beat_times)
    normal_mask = (rr_intervals >= MIN_NORMAL_RR_INTERVAL_SEC) & (
        rr_intervals <= MAX_NORMAL_RR_INTERVAL_SEC
    )
    abnormal_prefix = np.concatenate(([0], np.cumsum(~normal_mask, dtype=np.int_)))

    for start_idx in range(beat_times.size - 1):
        start_time = float(beat_times[start_idx])
        if start_time < start_bound:
            continue

        end_required = start_time + duration_sec
        if end_required > end_bound:
            break

        end_idx = int(np.searchsorted(beat_times, end_required, side="left"))
        if end_idx <= start_idx or end_idx >= beat_times.size:
            continue

        if abnormal_prefix[end_idx] - abnormal_prefix[start_idx] == 0:
            return start_time

    return None
