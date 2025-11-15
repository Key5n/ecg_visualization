from tqdm import tqdm
from ecg_visualization.logging.optuna import configure_optuna_logging
import optuna
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
from ecg_visualization.visualization.styles import apply_default_style


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

    storage_name = "mysql+pymysql://root:foo@localhost:3306/optuna"

    study_name = f"{entity.dataset_name} {entity.data_id}"

    storage_name = "mysql+pymysql://root:foo@localhost:3306/optuna"
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    print(study)
