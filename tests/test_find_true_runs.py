import unittest

import numpy as np

from ecg_visualization.utils.utils import find_true_runs


class FindTrueRunsTests(unittest.TestCase):
    def test_returns_all_consecutive_true_ranges(self) -> None:
        result = find_true_runs(
            [True, True, False, True, True, True, False, True, False]
        )

        self.assertEqual(result, [(0, 2), (3, 6), (7, 8)])

    def test_returns_empty_list_when_input_has_no_true_values(self) -> None:
        result = find_true_runs([False, False, False])

        self.assertEqual(result, [])

    def test_returns_single_range_when_all_values_are_true(self) -> None:
        result = find_true_runs([True, True, True, True])

        self.assertEqual(result, [(0, 4)])

    def test_accepts_numpy_boolean_arrays(self) -> None:
        result = find_true_runs(np.array([False, True, True, False], dtype=np.bool_))

        self.assertEqual(result, [(1, 3)])

    def test_returns_empty_list_for_empty_input(self) -> None:
        result = find_true_runs([])

        self.assertEqual(result, [])

    def test_rejects_non_1d_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "1D array-like"):
            find_true_runs(np.array([[True, False], [False, True]], dtype=np.bool_))


if __name__ == "__main__":
    unittest.main()
