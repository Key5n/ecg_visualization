from ecg_visualization.scripts.cudb_anomaly_scores.config import (
    load_cudb_anomaly_scores_config,
)
from ecg_visualization.scripts.cudb_anomaly_scores.cudb_anomaly_scores import (
    cudb_anomaly_scores,
)

if __name__ == "__main__":
    cudb_anomaly_scores(load_cudb_anomaly_scores_config())
