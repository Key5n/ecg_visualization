import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import optuna

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
    StudyLoader,
    build_storage_name,
    create_artifact_store,
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


@dataclass(slots=True)
class VisualizationJob:
    entity: ECG_Entity
    study: optuna.Study


def visualize_all_studies(max_workers: int | None = None):
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

    storage_name = build_storage_name()
    jobs = _prepare_visualization_jobs(entities, storage_name)
    if not jobs:
        return

    worker_count = _determine_worker_count(max_workers)
    if worker_count == 1:
        for job in tqdm(jobs, desc="visualizations"):
            visualize_study(job)
        return

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        results = executor.map(_run_visualization, jobs, chunksize=1)
        for dataset_id, entity_id, error in tqdm(
            results,
            total=len(jobs),
            desc="visualizations",
        ):
            if error:
                tqdm.write(
                    f"Visualization failed for {dataset_id}/{entity_id}: {error}"
                )


def visualize_study(job: VisualizationJob):
    apply_default_style()
    configure_optuna_logging()

    artifact_store = create_artifact_store(ARTIFACT_ROOT)

    visualizer = StudyVisualizer(
        entity=job.entity,
        study=job.study,
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

    env_value = os.getenv(WORKER_ENV_VAR)
    if env_value and env_value.isdigit():
        resolved = int(env_value)
        if resolved > 0:
            return resolved

    return max(1, os.cpu_count() or 1)


def _run_visualization(job: VisualizationJob) -> tuple[str, str, Exception | None]:
    try:
        visualize_study(job)
        return job.entity.dataset_id, job.entity.entity_id, None
    except Exception as exc:  # pragma: no cover - worker error propagation
        return job.entity.dataset_id, job.entity.entity_id, exc


def _prepare_visualization_jobs(
    entities: Iterable[ECG_Entity],
    storage_name: str,
) -> list[VisualizationJob]:
    jobs: list[VisualizationJob] = []
    loader = StudyLoader(storage_name)
    for entity in entities:
        study = loader.load(entity, log_fn=tqdm.write)
        if study is None:
            continue
        jobs.append(
            VisualizationJob(
                entity=entity,
                study=study,
            )
        )
    return jobs
