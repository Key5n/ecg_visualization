import logging

import structlog

from ecg_visualization.config.settings import LOG_LEVEL

DEFAULT_LOG_LEVEL = logging.WARNING


def configure_root_logging() -> None:
    level = logging.getLevelName(LOG_LEVEL.upper())

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(level),
    )
