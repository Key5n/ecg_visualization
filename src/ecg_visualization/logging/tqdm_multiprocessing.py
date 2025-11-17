"""
Utilities to integrate logging, tqdm, and multiprocessing without mangling progress bars.
"""

from __future__ import annotations

import contextlib
import logging
import logging.handlers
import multiprocessing as mp
from collections.abc import Iterator

from tqdm import tqdm

DEFAULT_FORMAT = "%(asctime)s - %(message)s"

__all__ = [
    "TqdmLoggingHandler",
    "queue_logging_context",
    "worker_logging_initializer",
]


class TqdmLoggingHandler(logging.Handler):
    """Route logging records through tqdm.write to preserve the progress bar."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - IO heavy
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


@contextlib.contextmanager
def queue_logging_context(
    level: int = logging.INFO,
    fmt: str = DEFAULT_FORMAT,
) -> Iterator[mp.Queue]:
    """
    Replace the root logger handlers with a queue handler and forward records through
    a tqdm-aware listener until the context exits.
    """

    log_queue: mp.Queue = mp.Queue()
    listener_handler = _create_tqdm_handler(level=level, fmt=fmt)
    listener = logging.handlers.QueueListener(
        log_queue,
        listener_handler,
        respect_handler_level=True,
    )

    root = logging.getLogger()
    previous_handlers = root.handlers[:]
    previous_level = root.level

    _install_queue_handler(root, log_queue, level)

    listener.start()
    try:
        yield log_queue
    finally:
        listener.stop()
        root.handlers = previous_handlers
        root.setLevel(previous_level)


def worker_logging_initializer(
    log_queue: mp.Queue,
    level: int = logging.INFO,
) -> None:
    """Configure worker processes to send logs to the shared queue."""

    root = logging.getLogger()
    _install_queue_handler(root, log_queue, level)


def _create_tqdm_handler(level: int, fmt: str) -> logging.Handler:
    handler = TqdmLoggingHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt))
    return handler


def _install_queue_handler(root: logging.Logger, log_queue: mp.Queue, level: int) -> None:
    """Attach a QueueHandler to the provided logger, replacing existing handlers."""

    queue_handler = logging.handlers.QueueHandler(log_queue)
    root.handlers.clear()
    root.addHandler(queue_handler)
    if root.level == logging.NOTSET:
        root.setLevel(level)
