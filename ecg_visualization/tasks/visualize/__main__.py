from ecg_visualization.tasks.visualize.config import load_visualize_config
from ecg_visualization.tasks.visualize.visualize import visualize_all_studies

if __name__ == "__main__":
    visualize_all_studies(load_visualize_config())
