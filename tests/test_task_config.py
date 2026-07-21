from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ecg_visualization.tasks.config import save_config_text


@dataclass(slots=True)
class NestedConfig:
    activation_func: object = np.tanh
    values: object = field(default_factory=lambda: np.array([1.0, 2.0]))


@dataclass(slots=True)
class ExampleConfig:
    root_dir: Path = Path("result/example")
    colors: dict[str, str] | None = None
    nested: NestedConfig | None = None


def test_save_config_text_writes_human_readable_yaml(tmp_path: Path) -> None:
    config = ExampleConfig(
        colors={"train": "#2a9d8f"},
        nested=NestedConfig(),
    )

    config_path = save_config_text(config, tmp_path / "config.txt")

    assert config_path.read_text() == (
        "root_dir: result/example\n"
        "colors:\n"
        "  train: '#2a9d8f'\n"
        "nested:\n"
        "  activation_func: numpy.tanh\n"
        "  values:\n"
        "  - 1.0\n"
        "  - 2.0\n"
    )
