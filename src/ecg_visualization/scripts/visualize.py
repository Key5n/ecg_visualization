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


def visualize_all_entities():
    data_sources: list[ECG_Dataset] = [
        CUDB(),
        AFPDB(),
        MITDB(),
        AFDB(),
        LTAFDB(),
        SHDBAF(),
        SDDB(),
    ]

    for data_source in tqdm(data_sources):
        for entity in tqdm(data_source.data_entities):
            visualize_entity(entity)


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
