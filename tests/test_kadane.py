import unittest

import numpy as np

from ecg_visualization.utils.kadane import kadane


class KadaneTests(unittest.TestCase):
    def test_returns_max_sum_and_subarray_for_mixed_values(self) -> None:
        result = kadane([1.0, -3.0, 2.0, 1.0, -1.0, 3.0, -2.0])

        self.assertEqual(result.max_value, 5.0)
        np.testing.assert_array_equal(result.subarray, np.array([2.0, 1.0, -1.0, 3.0]))

    def test_returns_single_largest_value_for_all_negative_input(self) -> None:
        result = kadane([-5.0, -2.0, -7.0])

        self.assertEqual(result.max_value, -2.0)
        np.testing.assert_array_equal(result.subarray, np.array([-2.0]))

    def test_returns_only_value_for_single_element_input(self) -> None:
        result = kadane([4.0])

        self.assertEqual(result.max_value, 4.0)
        np.testing.assert_array_equal(result.subarray, np.array([4.0]))

    def test_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one value"):
            kadane([])

    def test_rejects_non_1d_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "1D array-like"):
            kadane(np.array([[1.0, 2.0], [3.0, 4.0]]))


if __name__ == "__main__":
    unittest.main()
