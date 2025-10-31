import os
from typing import Final

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from ecg_visualization.datasets.dataset import (
    AFDB,
    AFPDB,
    CUDB,
    LTAFDB,
    MITDB,
    SDDB,
    SHDBAF,
    ECG_Dataset,
)


def RRI_histogram() -> None:
    """
    Generate histograms describing how long it takes to observe several R-peak
    windows.

    The duration of consecutive beats is calculated from the cumulative time
    difference between the first and the kth beat index sourced from
    ``ECG_Entity.beats``. Results are saved under
    ``result/RRI_histogram/<dataset_id>/<record>.pdf`` with one page per window
    size to keep them separate from the main visualization outputs.
    """
    window_sizes: Final[tuple[int, ...]] = (10, 50, 100)
    data_sources: list[ECG_Dataset] = [
        CUDB(),
        AFPDB(),
        MITDB(),
        AFDB(),
        LTAFDB(),
        SHDBAF(),
        SDDB(),
    ]

    task_name: Final[str] = "RRI_histogram"
    task_result_root = os.path.join("result", task_name)
    os.makedirs(task_result_root, exist_ok=True)

    for data_source in data_sources:
        dataset_dir = os.path.join(task_result_root, data_source.dataset_id)
        os.makedirs(dataset_dir, exist_ok=True)
        dataset_durations: dict[int, list[np.ndarray]] = {
            window_size: [] for window_size in window_sizes
        }

        for entity in data_source.data_entities:
            entity_durations: list[tuple[int, np.ndarray]] = []
            beat_times = entity.beats.astype(np.float64) / entity.sr

            for window_size in window_sizes:
                if entity.beats.size < window_size:
                    # Not enough beats to form this window.
                    continue
                # Time between the first and kth peak for each window.
                end_times = beat_times[window_size - 1 :]
                start_times = beat_times[: beat_times.size - window_size + 1]
                durations = end_times - start_times

                if durations.size == 0:
                    continue

                dataset_durations[window_size].append(durations)
                entity_durations.append((window_size, durations))

            if not entity_durations:
                continue

            result_path = os.path.join(dataset_dir, f"{entity.data_id}.pdf")
            with PdfPages(result_path) as pdf:
                for window_size, durations in entity_durations:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.hist(durations, bins="auto", color="tab:blue", alpha=0.7)
                    ax.set_title(
                        f"{entity.dataset_name} / {entity.data_id} (k={window_size})"
                    )
                    ax.set_xlabel("Time for R-peak window (sec)")
                    ax.set_ylabel("Count")
                    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
                    fig.tight_layout()

                    pdf.savefig(fig)
                    plt.close(fig)
