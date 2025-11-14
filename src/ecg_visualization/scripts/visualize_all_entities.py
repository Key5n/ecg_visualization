from tqdm import tqdm
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
)
from ecg_visualization.visualization.styles import apply_default_style


def visualize_all_entities():
    apply_default_style()
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
        dataset_result_dir = os.path.join(root_result_dir, data_source.dataset_id)
        os.makedirs(dataset_result_dir, exist_ok=True)

        for entity in tqdm(data_source.data_entities):
            storage_name = "mysql+pymysql://root:foo@localhost:3306/optuna"

            optuna.logging.get_logger("optuna").addHandler(
                logging.StreamHandler(sys.stdout)
            )
            study_name = f"{entity.dataset_name} {entity.data_id}"

            storage_name = "mysql+pymysql://root:foo@localhost:3306/optuna"
            study = optuna.load_study(study_name=study_name, storage=storage_name)
            print(study)
