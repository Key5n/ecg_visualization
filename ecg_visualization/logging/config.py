import logging

from ecg_visualization.config.settings import DEFAULT_LOG_LEVEL, LOG_LEVEL

DEFAULT_FORMAT = "%(levelname)s: %(message)s"


def get_log_level() -> int:
    normalized = LOG_LEVEL.strip().upper()
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
