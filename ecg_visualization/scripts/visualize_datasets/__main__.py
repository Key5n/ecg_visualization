from ecg_visualization.scripts.visualize_datasets.config import (
    load_visualize_datasets_config,
)
from ecg_visualization.scripts.visualize_datasets.visualize_datasets import (
    visualize_datasets,
)

if __name__ == "__main__":
    visualize_datasets(load_visualize_datasets_config())
