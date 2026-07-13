from ecg_visualization.tasks.sddb_concat.config import load_sddb_concat_config
from ecg_visualization.tasks.sddb_concat.visualize.visualize import (
    sddb_concat_visualize,
)

if __name__ == "__main__":
    sddb_concat_visualize(load_sddb_concat_config())
