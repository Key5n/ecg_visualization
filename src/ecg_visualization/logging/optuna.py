import logging

import optuna

from ecg_visualization.logging.tqdm_multiprocessing import TqdmLoggingHandler


def configure_optuna_logging() -> None:
    """
    Configure optuna to propagate logs through Python's logging system while ensuring
    a tqdm-friendly handler exists for callers that have not configured logging yet.
    """

    logger = optuna.logging.get_logger("optuna")

    optuna.logging.disable_default_handler()
    logger.handlers.clear()
    logger.propagate = True

    root = logging.getLogger()
    has_tqdm_handler = any(
        isinstance(handler, TqdmLoggingHandler) for handler in root.handlers
    )
    if not has_tqdm_handler:
        handler = TqdmLoggingHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(message)s"))
        if root.level == logging.NOTSET:
            root.setLevel(logging.INFO)
        root.addHandler(handler)
