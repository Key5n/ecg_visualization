from typing import Tuple
import numpy.typing as npt
import math
import numpy as np
from sklearn.preprocessing import StandardScaler


def padding_reshape(array: npt.NDArray, shape: Tuple, fill_value=np.nan):
    total = math.prod(shape)

    if len(array) < total:
        pad_length = total - len(array)
        array = np.concatenate([array, np.full(pad_length, fill_value)])

    return np.reshape(array, shape)


def omit_nan(array: npt.NDArray):
    return array[~np.isnan(array)]


def merge_overlapping_windows(
    windows: set[tuple[float, float]],
) -> set[tuple[float, float]]:
    if not windows:
        return set()

    sorted_windows = sorted(windows)
    merged: list[list[float]] = []
    for start_time, end_time in sorted_windows:
        if not merged or start_time > merged[-1][1]:
            merged.append([start_time, end_time])
        else:
            merged[-1][1] = max(merged[-1][1], end_time)
    return {(pair[0], pair[1]) for pair in merged}


def prepare_sequences(
    train_window: npt.NDArray[np.float64],
    full_signal: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    train = _ensure_2d(np.asarray(train_window, dtype=np.float64))
    test = _ensure_2d(np.asarray(full_signal, dtype=np.float64))

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    return train_scaled, test_scaled


def _ensure_2d(signal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if signal.ndim == 1:
        return signal.reshape(-1, 1)
    return signal
