from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

from omegaconf import OmegaConf

T = TypeVar("T")


def load_task_config(
    config: T,
) -> T:
    structured = OmegaConf.structured(config, flags={"allow_objects": True})
    cli_config = OmegaConf.from_cli()
    merged = OmegaConf.merge(structured, cli_config)
    return cast(T, OmegaConf.to_object(merged))


def save_config_text(
    config: object,
    config_path: Path | str,
) -> Path:
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_yaml = OmegaConf.to_yaml(
        OmegaConf.create(_to_config_container(config)),
        sort_keys=False,
    )
    config_path.write_text(config_yaml)
    return config_path


def _to_config_container(value: object) -> object:
    if OmegaConf.is_config(value):
        return _to_config_container(OmegaConf.to_object(value))
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _to_config_container(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            _to_config_container(key): _to_config_container(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_to_config_container(item) for item in value]
    if hasattr(value, "tolist"):
        return _to_config_container(value.tolist())
    if callable(value):
        return _callable_name(value)
    return value


def _callable_name(value: Any) -> str:
    module = getattr(value, "__module__", None)
    name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if module and name:
        return f"{module}.{name}"
    if name:
        return str(name)
    return repr(value)
