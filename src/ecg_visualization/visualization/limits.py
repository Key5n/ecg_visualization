import numpy as np

from ecg_visualization.utils.utils import omit_nan


def compute_ylim(
    signals: np.ndarray,
    *,
    scale: float = 1.1,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> tuple[float, float]:
    """Return (ylim_lower, ylim_upper) for the provided signals."""

    cleaned = omit_nan(signals)
    if cleaned.size == 0:
        default_lower = -5.0 if lower_bound is None else lower_bound
        default_upper = 5.0 if upper_bound is None else upper_bound
        return float(default_lower), float(default_upper)

    scaled_max = float(np.max(cleaned) * scale)
    scaled_min = float(np.min(cleaned) * scale)

    ylim_upper = (
        scaled_max if upper_bound is None else float(min(scaled_max, upper_bound))
    )
    ylim_lower = (
        scaled_min if lower_bound is None else float(max(scaled_min, lower_bound))
    )
    return ylim_lower, ylim_upper
