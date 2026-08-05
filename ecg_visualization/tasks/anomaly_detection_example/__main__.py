from ecg_visualization.tasks.anomaly_detection_example.config import (
    load_anomaly_detection_example_config,
)
from ecg_visualization.tasks.anomaly_detection_example.run import (
    anomaly_detection_example,
)

if __name__ == "__main__":
    anomaly_detection_example(load_anomaly_detection_example_config())
