import importlib
import os
import unittest
from unittest.mock import patch

from ecg_visualization.config import settings as settings_module
from ecg_visualization.core import entity as entity_module


class EntityConfigTests(unittest.TestCase):
    def tearDown(self) -> None:
        importlib.reload(settings_module)
        importlib.reload(entity_module)

    def reload_entity_config(self):
        importlib.reload(settings_module)
        return importlib.reload(entity_module)

    def test_uses_default_normal_segment_thresholds(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MIN_NORMAL_RR_INTERVAL_SEC": "",
                "MAX_NORMAL_RR_INTERVAL_SEC": "",
                "NORMAL_SEGMENT_DURATION_SEC": "",
            },
            clear=False,
        ):
            for env_var in (
                "MIN_NORMAL_RR_INTERVAL_SEC",
                "MAX_NORMAL_RR_INTERVAL_SEC",
                "NORMAL_SEGMENT_DURATION_SEC",
            ):
                os.environ.pop(env_var)

            reloaded = self.reload_entity_config()

        self.assertEqual(reloaded.MIN_NORMAL_RR_INTERVAL_SEC, 0.6)
        self.assertEqual(reloaded.MAX_NORMAL_RR_INTERVAL_SEC, 1.0)
        self.assertEqual(reloaded.NORMAL_SEGMENT_DURATION_SEC, 300.0)

    def test_uses_env_overrides_for_normal_segment_thresholds(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MIN_NORMAL_RR_INTERVAL_SEC": "0.7",
                "MAX_NORMAL_RR_INTERVAL_SEC": "1.2",
                "NORMAL_SEGMENT_DURATION_SEC": "120",
            },
            clear=False,
        ):
            reloaded = self.reload_entity_config()

        self.assertEqual(reloaded.MIN_NORMAL_RR_INTERVAL_SEC, 0.7)
        self.assertEqual(reloaded.MAX_NORMAL_RR_INTERVAL_SEC, 1.2)
        self.assertEqual(reloaded.NORMAL_SEGMENT_DURATION_SEC, 120.0)

    def test_rejects_invalid_env_override(self) -> None:
        with patch.dict(
            os.environ,
            {"MIN_NORMAL_RR_INTERVAL_SEC": "not-a-number"},
            clear=False,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "could not convert string to float",
            ):
                self.reload_entity_config()


if __name__ == "__main__":
    unittest.main()
