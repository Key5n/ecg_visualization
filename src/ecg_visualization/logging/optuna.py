import logging

import optuna
from tqdm import tqdm


class _OptunaTqdmHandler(logging.Handler):
    """Route optuna logs through tqdm.write to preserve progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def configure_optuna_logging() -> None:
    """
    Configure optuna to use tqdm-aware logging once for the lifetime of the process.
    """

    logger = optuna.logging.get_logger("optuna")
    if any(isinstance(handler, _OptunaTqdmHandler) for handler in logger.handlers):
        return

    optuna.logging.disable_default_handler()
    handler = _OptunaTqdmHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
