from __future__ import annotations

from typing import TypeVar, cast

from omegaconf import DictConfig, ListConfig, OmegaConf

T = TypeVar("T")


def load_task_config(
    config: T,
) -> T:
    structured = OmegaConf.structured(config, flags={"allow_objects": True})
    _set_readonly_recursive(structured, False)
    cli_config = OmegaConf.from_cli()
    merged = OmegaConf.merge(structured, cli_config)
    return cast(T, OmegaConf.to_object(merged))


def _set_readonly_recursive(config: DictConfig | ListConfig, readonly: bool) -> None:
    OmegaConf.set_readonly(config, readonly)
    values = config.values() if isinstance(config, DictConfig) else config
    for value in values:
        if isinstance(value, DictConfig | ListConfig):
            _set_readonly_recursive(value, readonly)
