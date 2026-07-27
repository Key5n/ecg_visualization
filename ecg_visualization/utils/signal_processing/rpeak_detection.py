from __future__ import annotations

import numpy as np
import numpy.typing as npt
from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter
from biosppy.signals.tools import filter_signal


def detect_rpeaks(
    signal: npt.NDArray[np.float64],
    sampling_rate_hz: int | float,
    CHUNK_DURATION_SEC=120.0,
    CHUNK_OVERLAP_SEC=10.0,
) -> npt.NDArray[np.int_]:
    samples = np.asarray(signal, dtype=np.float64)
    if samples.size < 3:
        raise ValueError("signal must contain at least 3 samples")

    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    sampling_rate = float(sampling_rate_hz)

    chunk_size = int(round(CHUNK_DURATION_SEC * sampling_rate))
    overlap_size = int(round(CHUNK_OVERLAP_SEC * sampling_rate))
    detected_chunks: list[npt.NDArray[np.int_]] = []

    for core_start in range(0, samples.size, chunk_size):
        core_end = min(core_start + chunk_size, samples.size)
        chunk_start = max(0, core_start - overlap_size)
        chunk_end = min(samples.size, core_end + overlap_size)
        chunk_rpeaks = _detect_rpeaks_chunk(
            samples[chunk_start:chunk_end],
            sampling_rate,
        )
        global_rpeaks = chunk_rpeaks + chunk_start

        # Each peak belongs to exactly one non-overlapping core. The surrounding
        # overlap only supplies detector context near chunk boundaries.
        in_core = (global_rpeaks >= core_start) & (global_rpeaks < core_end)
        detected_chunks.append(global_rpeaks[in_core])

    return np.asarray(np.concatenate(detected_chunks), dtype=np.int_)


def _detect_rpeaks_chunk(
    signal: npt.NDArray[np.float64],
    sampling_rate_hz: float,
) -> npt.NDArray[np.int_]:
    order = int(1.5 * sampling_rate_hz)
    filtered, _, _ = filter_signal(
        signal=signal,
        ftype="FIR",
        band="bandpass",
        order=order,
        frequency=[0.67, 45],
        sampling_rate=sampling_rate_hz,
    )

    filtered = filtered - np.mean(filtered)  # remove DC offset

    (rpeaks,) = hamilton_segmenter(signal=filtered, sampling_rate=sampling_rate_hz)

    (rpeaks,) = correct_rpeaks(
        signal=filtered, rpeaks=rpeaks, sampling_rate=sampling_rate_hz, tol=0.05
    )

    return np.asarray(rpeaks, dtype=np.int_)
