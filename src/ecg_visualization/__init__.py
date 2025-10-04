import os
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
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from ecg_visualization.utils.utils import padding_reshape
import seaborn as sns

custom_params = {
    "lines.linewidth": 0.5,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "axes.linewidth": 0.5,
    "xtick.labelsize": 6,
    "axes.xmargin": 0,
    "axes.ymargin": 0,
}
sns.set_theme(context="paper", style="whitegrid", rc=custom_params)


def main() -> None:
    data_sources: list[ECG_Dataset] = [
        CUDB(),
        AFPDB(),
        MITDB(),
        AFDB(),
        LTAFDB(),
        SHDBAF(),
        SDDB(),
    ]

    root_result_dir = os.path.join("result")

    for data_source in data_sources:
        dataset_result_dir = os.path.join(root_result_dir, data_source.dataset_id)
        os.makedirs(dataset_result_dir, exist_ok=True)

        for entity in data_source.data_entities:
            result_file_path = os.path.join(dataset_result_dir, f"{entity.data_id}.pdf")
            with PdfPages(result_file_path) as pdf:

                n_steps = entity.sr * 10
                n_rows = 6
                n_pages = math.ceil(len(entity.signals) / n_rows / n_steps)
                signals_paged = padding_reshape(
                    entity.signals, (n_pages, n_rows, n_steps)
                )

                length = n_pages * n_rows * n_steps
                ts_paged = np.linspace(0, length / entity.sr, length).reshape(
                    (n_pages, n_rows, n_steps)
                )

                ylim_upper = np.max(entity.signals) * 1.1
                ylim_lower = np.min(entity.signals) * 1.1

                for page_idx, (signals, ts_row) in enumerate(
                    zip(signals_paged, ts_paged)
                ):
                    fig, axs = plt.subplots(
                        nrows=n_rows,
                        # a4
                        figsize=(8.27, 11.69),
                    )
                    for signal, ts, ax in zip(signals, ts_row, axs):
                        ax.plot(ts, signal, "-")
                        ax.set_ylim(ylim_lower, ylim_upper)
                        ax.set_ylabel("mV")
                        ax.set_xlim(ts[0], ts[-1])

                    fig.suptitle(
                        f"{entity.data_kind}: {entity.data_id} Page {page_idx + 1} / {n_pages}",
                    )
                    fig.supxlabel("Time (sec)")
                    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.95)
                    pdf.savefig(pad_inches=0)
                    plt.close()
