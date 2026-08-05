from ecg_visualization.tasks.all_anomaly_detection_scores.config import (
    load_all_anomaly_detection_scores_config,
)
from ecg_visualization.tasks.all_anomaly_detection_scores.run import (
    all_anomaly_detection_scores,
)

if __name__ == "__main__":
    all_anomaly_detection_scores(load_all_anomaly_detection_scores_config())
