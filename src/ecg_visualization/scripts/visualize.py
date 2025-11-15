import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from ecg_visualization.datasets.dataset import (
    AFDB,
    AFPDB,
    CUDB,
    LTAFDB,
    MITDB,
    SDDB,
    SHDBAF,
    ECG_Dataset,
    ECG_Entity,
)
from ecg_visualization.logging import configure_optuna_logging
from ecg_visualization.utils.optuna_record import (
    build_storage_name,
    create_artifact_store,
    load_study_for_entity,
)
from ecg_visualization.visualization.layouts import PaginationConfig
from ecg_visualization.visualization.styles import apply_default_style
from ecg_visualization.visualization.study_visualizer import StudyVisualizer

RR_WINDOW_BEATS = 100
PAGINATION_CONFIG = PaginationConfig()
ARTIFACT_ROOT = Path("result") / "artifacts"
VISUALIZATION_ROOT = Path("result") / "visualize"

load_dotenv()


def visualize_all_entities(max_workers: int | None = None):
    data_sources: list[ECG_Dataset] = [
        CUDB(),
        AFPDB(),
        MITDB(),
        AFDB(),
        LTAFDB(),
        SHDBAF(),
        SDDB(),
    ]

    entities: list[ECG_Entity] = []
    for data_source in data_sources:
        entities.extend(data_source.data_entities)

    worker_count = _determine_worker_count(max_workers)
    if worker_count == 1:
        for entity in tqdm(entities, desc="visualizations"):
            visualize_entity(entity)
        return

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        results = executor.map(_run_visualization, entities, chunksize=1)
        for dataset_id, entity_id, error in tqdm(
            results,
            total=len(entities),
            desc="visualizations",
        ):
            if error:
                tqdm.write(
                    f"Visualization failed for {dataset_id}/{entity_id}: {error}"
                )


def visualize_entity(entity: ECG_Entity):
    apply_default_style()
    configure_optuna_logging()

    storage_name = build_storage_name()

    artifact_store = create_artifact_store(ARTIFACT_ROOT)
    study = load_study_for_entity(
        entity,
        storage_name=storage_name,
        log_fn=tqdm.write,
    )
    if study is None:
        return

    visualizer = StudyVisualizer(
        entity=entity,
        study=study,
        artifact_store=artifact_store,
        pagination_config=PAGINATION_CONFIG,
        visualization_root=VISUALIZATION_ROOT,
        rr_window_beats=RR_WINDOW_BEATS,
    )
    output_path = visualizer.visualize()
    if output_path:
        tqdm.write(f"Saved visualization to {output_path}")


def _determine_worker_count(max_workers: int | None) -> int:
    if max_workers is not None:
        return max(1, max_workers)

    WORKER_ENV_VAR = "ECG_VISUALIZE_WORKERS"
    env_value = os.getenv(WORKER_ENV_VAR)
    if env_value and env_value.isdigit():
        resolved = int(env_value)
        if resolved > 0:
            return resolved

    return max(1, os.cpu_count() or 1)


def _run_visualization(entity: ECG_Entity) -> tuple[str, str, Exception | None]:
    try:
        visualize_entity(entity)
        return entity.dataset_id, entity.entity_id, None
    except Exception as exc:  # pragma: no cover - worker error propagation
        return entity.dataset_id, entity.entity_id, exc
