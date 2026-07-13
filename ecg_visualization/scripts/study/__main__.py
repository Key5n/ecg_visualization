from ecg_visualization.scripts.study.config import load_config
from ecg_visualization.scripts.study.study import study_all_entities

if __name__ == "__main__":
    study_all_entities(load_config())
