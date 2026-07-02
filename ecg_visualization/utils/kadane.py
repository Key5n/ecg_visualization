from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class KadaneResult:
    max_value: float
    subarray: npt.NDArray[np.float64]


def kadane(array: npt.NDArray[np.float64] | list[float]) -> KadaneResult:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("kadane expects a 1D array-like input")
    if values.size == 0:
        raise ValueError("kadane expects at least one value")

    max_so_far = float(values[0])
    current_max = float(values[0])
    best_start = 0
    best_end = 1
    current_start = 0

    for idx, value in enumerate(values[1:], start=1):
        if value > current_max + value:
            current_max = float(value)
            current_start = idx
        else:
            current_max += float(value)

        if current_max > max_so_far:
            max_so_far = current_max
            best_start = current_start
            best_end = idx + 1

    return KadaneResult(
        max_value=max_so_far,
        subarray=values[best_start:best_end].copy(),
    )
