from ecg_visualization.tasks.sddb_concat.config import load_sddb_concat_config
from ecg_visualization.tasks.sddb_concat.score.score import sddb_concat_scores
from ecg_visualization.tasks.sddb_concat.visualize.visualize import (
    sddb_concat_visualize,
)

if __name__ == "__main__":
    config = load_sddb_concat_config()
    sddb_concat_scores(config)
    sddb_concat_visualize(config)
