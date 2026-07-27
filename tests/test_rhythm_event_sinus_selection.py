import numpy as np
import pytest

from ecg_visualization.tasks.rhythm_event_sequences.config import (
    MedianThresholdSinusExtractionConfig,
    PercentileRangeSinusExtractionConfig,
    SinusExtractionConfig,
)
from ecg_visualization.tasks.rhythm_event_sequences.utils import _build_sinus_rr_mask


def test_median_threshold_sinus_definition() -> None:
    rr_intervals = np.asarray([0.7, 0.8, 0.9, 1.2], dtype=np.float64)

    result = _build_sinus_rr_mask(
        rr_intervals,
        SinusExtractionConfig(
            median_threshold=MedianThresholdSinusExtractionConfig(threshold_sec=0.1)
        ),
    )

    np.testing.assert_array_equal(result, [False, True, True, False])


def test_percentile_range_sinus_definition_includes_limits() -> None:
    rr_intervals = np.asarray([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64)

    result = _build_sinus_rr_mask(
        rr_intervals,
        SinusExtractionConfig(method="percentile_range"),
    )

    np.testing.assert_array_equal(result, [False, True, True, True, False])


def test_percentile_range_uses_configured_limits() -> None:
    rr_intervals = np.asarray([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64)

    result = _build_sinus_rr_mask(
        rr_intervals,
        SinusExtractionConfig(
            method="percentile_range",
            percentile_range=PercentileRangeSinusExtractionConfig(
                lower_percentile=0,
                upper_percentile=50,
            ),
        ),
    )

    np.testing.assert_array_equal(result, [True, True, True, False, False])


@pytest.mark.parametrize(
    ("lower_percentile", "upper_percentile"),
    [(-1, 75), (25, 101), (75, 25), (50, 50)],
)
def test_rejects_invalid_sinus_rr_percentiles(
    lower_percentile: float,
    upper_percentile: float,
) -> None:
    with pytest.raises(ValueError, match="sinus RR percentiles must satisfy"):
        PercentileRangeSinusExtractionConfig(
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )


def test_rejects_unknown_sinus_extraction_method() -> None:
    with pytest.raises(ValueError, match="Unknown sinus extraction method"):
        SinusExtractionConfig(method="unknown")
