from ecg_visualization.tasks.cudb_anomaly_scores.config import (
    load_cudb_anomaly_scores_config,
)
from ecg_visualization.tasks.cudb_anomaly_scores.run import (
    cudb_anomaly_scores,
)

if __name__ == "__main__":
    cudb_anomaly_scores(load_cudb_anomaly_scores_config())
