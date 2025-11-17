import logging
import multiprocessing as mp
import os
from pathlib import Path

import optuna
from dotenv import load_dotenv
from tqdm import tqdm

from ecg_visualization.logging import (
    queue_logging_context,
    worker_logging_initializer,
)
from ecg_visualization.utils.optuna_record import (
    build_storage_name,
    create_artifact_store,
    get_study_identifiers,
    load_studies,
)
from ecg_visualization.visualization.layouts import PaginationConfig
from ecg_visualization.visualization.styles import apply_default_style
from ecg_visualization.visualization.study_visualizer import StudyVisualizer

RR_WINDOW_BEATS = 100
PAGINATION_CONFIG = PaginationConfig()
ARTIFACT_ROOT = Path("result") / "artifacts"
VISUALIZATION_ROOT = Path("result") / "visualize"

load_dotenv()
WORKER_ENV_VAR = "ECG_VISUALIZE_WORKERS"
LOGGER = logging.getLogger(__name__)


def visualize_all_studies(max_workers: int | None = None):
    storage_name = build_storage_name()
    studies = load_studies(storage_name)
    if not studies:
        return

    worker_count = _determine_worker_count(max_workers)
    if worker_count == 1:
        for study in tqdm(studies, desc="visualizations"):
            visualize_study(study)
        return

    with queue_logging_context() as log_queue:
        with mp.Pool(
            processes=worker_count,
            initializer=worker_logging_initializer,
            initargs=(log_queue,),
        ) as pool:
            results = pool.imap(_run_visualization, studies, chunksize=1)
            for dataset_id, entity_id, error in tqdm(
                results,
                total=len(studies),
                desc="visualizations",
            ):
                if error:
                    LOGGER.error(
                        f"Visualization failed for {dataset_id}/{entity_id}: {error}"
                    )


def visualize_study(study: optuna.Study):
    apply_default_style()

    artifact_store = create_artifact_store(ARTIFACT_ROOT)

    visualizer = StudyVisualizer(
        study=study,
        artifact_store=artifact_store,
        pagination_config=PAGINATION_CONFIG,
        visualization_root=VISUALIZATION_ROOT,
        rr_window_beats=RR_WINDOW_BEATS,
    )
    output_path = visualizer.visualize()
    if output_path:
        LOGGER.info(f"Saved visualization to {output_path}")


def _determine_worker_count(max_workers: int | None) -> int:
    if max_workers is not None:
        return max(1, max_workers)

    env_value = os.getenv(WORKER_ENV_VAR)
    if env_value and env_value.isdigit():
        resolved = int(env_value)
        if resolved > 0:
            return resolved

    return max(1, os.cpu_count() or 1)


def _run_visualization(study: optuna.Study) -> tuple[str, str, Exception | None]:
    dataset_id, entity_id = get_study_identifiers(study)
    try:
        visualize_study(study)
        return dataset_id, entity_id, None
    except Exception as exc:  # pragma: no cover - worker error propagation
        return dataset_id, entity_id, exc

