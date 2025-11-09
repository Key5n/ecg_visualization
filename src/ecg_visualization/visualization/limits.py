import numpy as np

from ecg_visualization.utils.utils import omit_nan


def compute_signal_ylim(
    signals: np.ndarray,
    *,
    scale: float = 1.1,
    lower_bound: float = -5.0,
    upper_bound: float = 5.0,
) -> tuple[float, float]:
    """Return (ylim_lower, ylim_upper) for the provided signals."""

    cleaned = omit_nan(signals)
    if cleaned.size == 0:
        return lower_bound, upper_bound

    scaled_max = np.max(cleaned) * scale
    scaled_min = np.min(cleaned) * scale

    ylim_upper = float(min(scaled_max, upper_bound))
    ylim_lower = float(max(scaled_min, lower_bound))
    return ylim_lower, ylim_upper
