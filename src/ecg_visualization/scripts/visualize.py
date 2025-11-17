import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import optuna

from dotenv import load_dotenv
from tqdm import tqdm

from ecg_visualization.logging import configure_optuna_logging
from ecg_visualization.utils.optuna_record import (
    StudyLoader,
    build_storage_name,
    create_artifact_store,
    get_study_identifiers,
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

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        results = executor.map(_run_visualization, studies, chunksize=1)
        for dataset_id, entity_id, error in tqdm(
            results,
            total=len(studies),
            desc="visualizations",
        ):
            if error:
                tqdm.write(
                    f"Visualization failed for {dataset_id}/{entity_id}: {error}"
                )


def visualize_study(study: optuna.Study):
    apply_default_style()
    configure_optuna_logging()

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


def _run_visualization(study: optuna.Study) -> tuple[str, str, Exception | None]:
    dataset_id, entity_id = get_study_identifiers(study)
    try:
        visualize_study(study)
        return dataset_id, entity_id, None
    except Exception as exc:  # pragma: no cover - worker error propagation
        return dataset_id, entity_id, exc


def load_studies(storage_name: str) -> list[optuna.Study]:
    study_names = optuna.study.get_all_study_names(storage_name)
    loader = StudyLoader(storage_name)
    studies: list[optuna.Study] = []
    for study_name in study_names:
        study = loader.load_by_name(study_name, log_fn=tqdm.write)
        if study is None:
            continue
        studies.append(study)
    return studies
