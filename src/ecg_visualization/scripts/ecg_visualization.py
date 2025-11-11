import os
import matplotlib.pyplot as plt
import numpy as np
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
from ecg_visualization.visualization.export import pdf_exporter
from ecg_visualization.visualization.layouts import (
    PaginationConfig,
    create_page_layout,
    paginate_signals,
)
from ecg_visualization.visualization.plotters import (
    highlight_windows,
    plot_histogram,
    plot_normal_beats,
    plot_signal,
    plot_symbols,
)
from ecg_visualization.visualization.styles import (
    apply_default_style,
    EXTREME_INTERVAL_COLOR,
)
from ecg_visualization.visualization.limits import compute_ylim, compute_ylim

MIN_RR_INTERVAL_SEC = 0.6
MAX_RR_INTERVAL_SEC = 1.0
RR_WINDOW_BEATS = 100
HISTOGRAM_WINDOW_SIZES = (10, 50, 100)
PAGINATION_CONFIG = PaginationConfig()


def ecg_visualization() -> None:
    apply_default_style()
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
            with pdf_exporter(result_file_path) as exporter:
                for window_size in HISTOGRAM_WINDOW_SIZES:
                    windows = entity.get_window_durations(window_size)
                    if not windows:
                        continue

                    durations = np.array(
                        [(end - start) / entity.sr for start, end in windows],
                        dtype=float,
                    )
                    if durations.size == 0:
                        continue

                    histogram_fig, histogram_ax = plt.subplots(figsize=(8, 4))
                    plot_histogram(
                        histogram_ax,
                        durations,
                        bins="auto",
                        title=f"{entity.dataset_name}: {entity.data_id} "
                        f"(RR window={window_size})",
                        xlabel="Window duration (sec)",
                        ylabel="Count",
                        percentile_lines=(5.0, 95.0),
                    )
                    histogram_fig.tight_layout()
                    exporter.add_page(histogram_fig, pad_inches=0)
                    plt.close(histogram_fig)

                extreme_windows = entity.get_extreme_rr_windows(
                    RR_WINDOW_BEATS,
                    lower_percentile=5.0,
                    upper_percentile=95.0,
                )

                (
                    signals_paged,
                    ts_paged,
                    _,
                    n_rows,
                    _,
                ) = paginate_signals(entity.signals, entity.sr, PAGINATION_CONFIG)

                ylim_lower, ylim_upper = compute_ylim(
                    entity.signals, lower_bound=-5.0, upper_bound=5.0
                )

                symbol_list = sorted(set(entity.annotation.symbol))
                tqdm.write(
                    f"{entity.data_id}, {entity.dataset_name} {"".join(symbol_list)} "
                    f"The number of extreme window: {len(extreme_windows)}"
                )

                annotation_events = [
                    (sample / entity.sr, symbol)
                    for sample, symbol in zip(
                        entity.annotation.sample, entity.annotation.symbol
                    )
                ]
                beat_times = [beat_index / entity.sr for beat_index in entity.beats]

                for page_idx, (signals, ts_row) in enumerate(
                    zip(signals_paged, ts_paged)
                ):
                    fig, axs = create_page_layout(n_rows)

                    for signal, ts, ax in zip(signals, ts_row, axs):
                        window_start, window_end = ts[0], ts[-1]
                        beats_in_window = [
                            beat_time
                            for beat_time in beat_times
                            if window_start <= beat_time <= window_end
                        ]
                        plot_signal(
                            ax,
                            ts,
                            signal,
                            ylim_lower=ylim_lower,
                            ylim_upper=ylim_upper,
                        )
                        plot_normal_beats(
                            ax,
                            beats_in_window,
                            ylim_lower=ylim_lower,
                        )
                        plot_symbols(
                            ax,
                            annotation_events,
                            window_start=window_start,
                            window_end=window_end,
                            ylim_lower=ylim_lower,
                        )
                        highlight_windows(
                            ax,
                            extreme_windows,
                            window_start=window_start,
                            window_end=window_end,
                            ylim_upper=ylim_upper,
                            color=EXTREME_INTERVAL_COLOR,
                        )

                    if page_idx == 0:
                        fig.suptitle(
                            f"{entity.dataset_name}: {entity.data_id} {"".join(symbol_list)} {RR_WINDOW_BEATS}"
                        )
                    fig.supxlabel("Time (sec)")
                    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.95)
                    exporter.add_page(fig, pad_inches=0)
                    plt.close()
