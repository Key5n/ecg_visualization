import logging
from concurrent.futures import as_completed

import optuna
from tqdm import tqdm

from ecg_visualization.logging.config import configure_root_logging
from ecg_visualization.logging.tqdm_multiprocessing import (
    process_pool_logging_context,
)
from ecg_visualization.tasks.visualize_studies.config import VisualizeConfig
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

    if config.max_workers == 1:
        for study in tqdm(studies, desc="visualizations"):
            visualize_study(study, config)
        return

    with process_pool_logging_context(config.max_workers) as executor:
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
