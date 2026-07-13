import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import optuna
from tqdm import tqdm

from ecg_visualization.config.settings import ECG_VISUALIZE_WORKERS
from ecg_visualization.logging.config import configure_root_logging
from ecg_visualization.logging.tqdm_multiprocessing import (
    queue_logging_context,
    worker_logging_initializer,
)
from ecg_visualization.scripts.visualize.config import VisualizeConfig
from ecg_visualization.utils.optuna_record import (
    build_storage_name,
    create_artifact_store,
    get_study_identifiers,
    load_studies,
)
from ecg_visualization.visualization.study_visualizer import StudyVisualizer
from ecg_visualization.visualization.styles import apply_default_style

LOGGER = logging.getLogger(__name__)


def visualize_all_studies(config: VisualizeConfig):
    configure_root_logging()
    storage_name = build_storage_name()
    studies = load_studies(storage_name)
    if not studies:
        return

    worker_count = _determine_worker_count(config.max_workers)
    if worker_count == 1:
        for study in tqdm(studies, desc="visualizations"):
            visualize_study(study, config)
        return

    with queue_logging_context() as log_queue:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=worker_logging_initializer,
            initargs=(log_queue,),
        ) as executor:
            futures = [
                executor.submit(_run_visualization, study, config) for study in studies
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="visualizations",
            ):
                dataset_id, entity_id, error = future.result()
                if error:
                    LOGGER.error(
                        f"Visualization failed for {dataset_id}/{entity_id}: {error}"
                    )


def visualize_study(study: optuna.Study, config: VisualizeConfig):
    apply_default_style()

    artifact_store = create_artifact_store(config.artifact_root)

    visualizer = StudyVisualizer(
        study=study,
        artifact_store=artifact_store,
        pagination_config=config.pagination,
        visualization_root=config.visualization_root,
        rr_window_beats=config.rr_window_beats,
    )
    output_path = visualizer.visualize()
    if output_path:
        LOGGER.info(f"Saved visualization to {output_path}")


def _determine_worker_count(max_workers: int | None) -> int:
    if max_workers is not None:
        return max(1, max_workers)

    if ECG_VISUALIZE_WORKERS > 0:
        return ECG_VISUALIZE_WORKERS

    return max(1, os.cpu_count() or 1)


def _run_visualization(
    study: optuna.Study,
    config: VisualizeConfig,
) -> tuple[str, str, Exception | None]:
    dataset_id, entity_id = get_study_identifiers(study)
    try:
        visualize_study(study, config)
        return dataset_id, entity_id, None
    except Exception as exc:  # pragma: no cover - worker error propagation
        return dataset_id, entity_id, exc
