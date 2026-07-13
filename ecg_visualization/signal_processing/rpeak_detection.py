from __future__ import annotations

import numpy as np
import numpy.typing as npt
from biosppy.signals.ecg import ecg as biosppy_ecg


def detect_rpeaks(
    signal: npt.NDArray[np.float64],
    sampling_rate_hz: int | float,
) -> npt.NDArray[np.int_]:
    samples = np.asarray(signal, dtype=np.float64)
    if samples.size < 3:
        raise ValueError("signal must contain at least 3 samples")

    result = biosppy_ecg(
        signal=samples,
        sampling_rate=float(sampling_rate_hz),
        show=False,
    )
    return np.asarray(result["rpeaks"], dtype=np.int_)
