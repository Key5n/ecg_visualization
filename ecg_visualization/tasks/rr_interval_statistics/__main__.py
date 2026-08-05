from ecg_visualization.tasks.rr_interval_statistics.config import (
    load_rr_interval_statistics_config,
)
from ecg_visualization.tasks.rr_interval_statistics.run import rr_interval_statistics

if __name__ == "__main__":
    rr_interval_statistics(load_rr_interval_statistics_config())
