from ecg_visualization.tasks.sddb_concat.config import load_sddb_concat_config
from ecg_visualization.tasks.sddb_concat.score.score import sddb_concat_scores

if __name__ == "__main__":
    sddb_concat_scores(load_sddb_concat_config())
