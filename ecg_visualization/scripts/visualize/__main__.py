from ecg_visualization.scripts.visualize.config import load_config
from ecg_visualization.scripts.visualize.visualize import visualize_all_studies

if __name__ == "__main__":
    visualize_all_studies(load_config())
