from typing import Tuple
from numpy.typing import NDArray
import math
import numpy as np


def padding_reshape(array: NDArray, shape: Tuple, fill_value=np.nan):
    total = math.prod(shape)

    if len(array) < total:
        pad_length = total - len(array)
        array = np.concatenate([array, np.full(pad_length, fill_value)])

    return np.reshape(array, shape)


def omit_nan(array: NDArray):
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
