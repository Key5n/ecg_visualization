from ecg_visualization.tasks.sddb_rri_histograms.config import (
    load_sddb_rr_histograms_config,
)
from ecg_visualization.tasks.sddb_rri_histograms.run import sddb_rr_histograms

if __name__ == "__main__":
    sddb_rr_histograms(load_sddb_rr_histograms_config())
