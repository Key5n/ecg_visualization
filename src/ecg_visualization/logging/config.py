import logging
import os

DEFAULT_LOG_LEVEL = logging.WARNING
LOG_LEVEL_ENV_VAR = "LOG_LEVEL"
DEFAULT_FORMAT = "%(levelname)s: %(message)s"


def get_log_level() -> int:
    default_level_name = logging.getLevelName(DEFAULT_LOG_LEVEL)
    env_value = os.getenv(LOG_LEVEL_ENV_VAR, default_level_name)
    normalized = env_value.strip().upper()
    level = logging.getLevelNamesMapping().get(normalized)
    if isinstance(level, int):
        return level

    return DEFAULT_LOG_LEVEL


def configure_root_logging(
    *,
    fmt: str = DEFAULT_FORMAT,
    force: bool = False,
) -> int:
    level = get_log_level()
    logging.basicConfig(level=level, format=fmt, force=force)
    return level
