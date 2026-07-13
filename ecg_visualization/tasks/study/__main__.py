from ecg_visualization.tasks.study.config import load_study_config
from ecg_visualization.tasks.study.study import study_all_entities

if __name__ == "__main__":
    study_all_entities(load_study_config())
