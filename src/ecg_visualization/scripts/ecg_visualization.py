import os
from tqdm import tqdm
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
from ecg_visualization.utils.utils import (
    merge_overlapping_windows,
    omit_nan,
    padding_reshape,
)
import seaborn as sns

MIN_RR_INTERVAL_SEC = 0.6
MAX_RR_INTERVAL_SEC = 1.0
ABNORMAL_INTERVAL_COLOR = "#f4a261"
MIN_PR_INTERVAL_SEC = MIN_RR_INTERVAL_SEC
RR_WINDOW_BEATS = 10

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


def ecg_visualization() -> None:
    data_sources: list[ECG_Dataset] = [
        CUDB(),
        # AFPDB(),
        # MITDB(),
        # AFDB(),
        # LTAFDB(),
        # SHDBAF(),
        # SDDB(),
    ]

    root_result_dir = os.path.join("result")

    for data_source in tqdm(data_sources):
        dataset_result_dir = os.path.join(root_result_dir, data_source.dataset_id)
        os.makedirs(dataset_result_dir, exist_ok=True)

        for entity in tqdm(data_source.data_entities):
            result_file_path = os.path.join(dataset_result_dir, f"{entity.data_id}.pdf")
            with PdfPages(result_file_path) as pdf:
                beat_times = (
                    entity.beats / entity.sr if entity.beats.size > 0 else np.array([])
                )
                window_size = RR_WINDOW_BEATS
                abnormal_windows: set[tuple[float, float]] = set()
                if beat_times.size >= window_size:
                    end_times = beat_times[window_size - 1 :]
                    start_times = beat_times[: beat_times.size - window_size + 1]
                    durations = end_times - start_times
                    min_duration = MIN_PR_INTERVAL_SEC * window_size
                    max_duration = MAX_RR_INTERVAL_SEC * window_size
                    for start_time, end_time, duration in zip(
                        start_times, end_times, durations
                    ):
                        if duration < min_duration or duration > max_duration:
                            abnormal_windows.add((start_time, end_time))

                abnormal_windows = merge_overlapping_windows(abnormal_windows)

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

                ylim_upper = np.max(omit_nan(entity.signals)) * 1.1
                ylim_lower = np.min(omit_nan(entity.signals)) * 1.1

                symbol_list = list(set(entity.annotation.symbol))
                symbol_list.sort()
                tqdm.write(
                    f"{entity.data_id}, {entity.dataset_name}, {ylim_lower:.2f}, {ylim_upper:.2f} {"".join(symbol_list)}"
                )

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
                        ax.set_ylabel("mν")
                        ax.set_xlim(ts[0], ts[-1])

                        symbols = [
                            (sample / entity.sr, symbol)
                            for sample, symbol in zip(
                                entity.annotation.sample, entity.annotation.symbol
                            )
                            if sample / entity.sr >= ts[0]
                            and sample / entity.sr <= ts[-1]
                        ]

                        for sample, symbol in symbols:
                            if symbol != "N":
                                ax.axvline(sample, color="red", alpha=0.5)
                                ax.text(
                                    sample,
                                    ylim_lower,
                                    symbol,
                                    fontsize=8,
                                    horizontalalignment="center",
                                    c="red",
                                )

                        beats = [
                            (beat_index / entity.sr)
                            for beat_index in entity.beats
                            if beat_index / entity.sr >= ts[0]
                            and beat_index / entity.sr <= ts[-1]
                        ]

                        for beat_time in beats:
                            ax.text(
                                beat_time,
                                ylim_lower,
                                "N",
                                fontsize=4,
                                horizontalalignment="center",
                            )

                        for window_start, window_end in sorted(abnormal_windows):
                            if window_end <= ts[0] or window_start >= ts[-1]:
                                continue
                            highlight_start = max(window_start, ts[0])
                            highlight_end = min(window_end, ts[-1])

                            if highlight_end > highlight_start:
                                ax.axvspan(
                                    highlight_start,
                                    highlight_end,
                                    color=ABNORMAL_INTERVAL_COLOR,
                                    alpha=0.2,
                                )
                                ax.text(
                                    (highlight_start + highlight_end) / 2,
                                    ylim_upper,
                                    f"Average Time: ({(highlight_end - highlight_start) / window_size} )",
                                    fontsize=6,
                                    horizontalalignment="center",
                                    verticalalignment="bottom",
                                    c=ABNORMAL_INTERVAL_COLOR,
                                )

                    if page_idx == 0:
                        fig.suptitle(
                            f"{entity.dataset_name}: {entity.data_id} {"".join(symbol_list)}"
                        )
                    fig.supxlabel("Time (sec)")
                    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.95)
                    pdf.savefig(pad_inches=0)
                    plt.close()
