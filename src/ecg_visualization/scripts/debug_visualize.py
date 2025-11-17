"""
Standalone debug helper for the visualize_all_studies workflow.

This script exercises queue_logging_context + multiprocessing.Pool without relying
on Optuna or dataset code so multiprocessing/logging issues can be reproduced easily.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import random
import time

from tqdm import tqdm

from ecg_visualization.logging import (
    queue_logging_context,
    worker_logging_initializer,
)

LOGGER = logging.getLogger(__name__)


def run_debug_visualize(
    *,
    items: int = 16,
    worker_count: int | None = None,
) -> None:
    """Run a simplified version of visualize_all_studies."""

    tasks = list(range(items))
    resolved_workers = _determine_worker_count(worker_count)

    with queue_logging_context() as log_queue:
        if resolved_workers == 1:
            LOGGER.info("Running sequential debug flow.")
            for task in tqdm(tasks, desc="debug tasks"):
                _handle_task(*_run_task(task))
            return

        LOGGER.info(f"Running debug flow with {resolved_workers} workers.")
        with mp.Pool(
            processes=resolved_workers,
            initializer=worker_logging_initializer,
            initargs=(log_queue,),
        ) as pool:
            results = pool.imap(_run_task, tasks, chunksize=1)
            for task_id, error in tqdm(
                results,
                total=len(tasks),
                desc="debug tasks",
            ):
                _handle_task(task_id, error)


def _determine_worker_count(worker_count: int | None) -> int:
    if worker_count is not None:
        return max(1, worker_count)
    return max(1, os.cpu_count() or 1)


def _run_task(task_id: int) -> tuple[int, Exception | None]:
    try:
        _simulate_work(task_id)
        return task_id, None
    except Exception as exc:  # pragma: no cover - debug helper
        return task_id, exc


def _simulate_work(task_id: int) -> None:
    LOGGER.info(f"start task-{task_id}")
    # Add entropy so interleaving is easier to spot in logs.
    time.sleep(random.uniform(0.1, 0.5))
    if task_id % 5 == 0:
        raise RuntimeError(f"debug failure for task-{task_id}")
    LOGGER.info(f"finish task-{task_id}")


def _handle_task(task_id: int, error: Exception | None) -> None:
    if error:
        LOGGER.error(f"Task {task_id} failed: {error}")
    else:
        LOGGER.info(f"Task {task_id} completed successfully.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run_debug_visualize()
