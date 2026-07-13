import numpy as np
import pytest

from ecg_visualization.utils.utils import find_true_runs


def test_returns_all_consecutive_true_ranges() -> None:
    result = find_true_runs([True, True, False, True, True, True, False, True, False])

    assert result == [(0, 2), (3, 6), (7, 8)]


def test_returns_empty_list_when_input_has_no_true_values() -> None:
    result = find_true_runs([False, False, False])

    assert result == []


def test_returns_single_range_when_all_values_are_true() -> None:
    result = find_true_runs([True, True, True, True])

    assert result == [(0, 4)]


def test_accepts_numpy_boolean_arrays() -> None:
    result = find_true_runs(np.array([False, True, True, False], dtype=np.bool_))

    assert result == [(1, 3)]


def test_returns_empty_list_for_empty_input() -> None:
    result = find_true_runs([])

    assert result == []


def test_rejects_non_1d_input() -> None:
    with pytest.raises(ValueError, match="1D array-like"):
        find_true_runs(np.array([[True, False], [False, True]], dtype=np.bool_))
