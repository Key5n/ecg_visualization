import os
import matplotlib.pyplot as plt
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
    plot_abnormal_windows,
    plot_signal,
    plot_symbols,
)
from ecg_visualization.visualization.styles import apply_default_style
from ecg_visualization.visualization.limits import compute_signal_ylim

MIN_RR_INTERVAL_SEC = 0.6
MAX_RR_INTERVAL_SEC = 1.0
MIN_PR_INTERVAL_SEC = MIN_RR_INTERVAL_SEC
RR_WINDOW_BEATS = 100
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
                abnormal_windows = entity.get_abnormal_windows(
                    RR_WINDOW_BEATS,
                    MIN_RR_INTERVAL_SEC * RR_WINDOW_BEATS,
                    MAX_RR_INTERVAL_SEC * RR_WINDOW_BEATS,
                )

                (
                    signals_paged,
                    ts_paged,
                    _,
                    n_rows,
                    _,
                ) = paginate_signals(entity.signals, entity.sr, PAGINATION_CONFIG)

                ylim_lower, ylim_upper = compute_signal_ylim(entity.signals)

                symbol_list = list(set(entity.annotation.symbol))
                symbol_list.sort()
                tqdm.write(
                    f"{entity.data_id}, {entity.dataset_name}, {ylim_lower:.2f}, {ylim_upper:.2f} {"".join(symbol_list)}"
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
                            beat_times=beats_in_window,
                        )
                        plot_symbols(
                            ax,
                            annotation_events,
                            window_start=window_start,
                            window_end=window_end,
                            ylim_lower=ylim_lower,
                        )
                        plot_abnormal_windows(
                            ax,
                            abnormal_windows,
                            window_start=window_start,
                            window_end=window_end,
                            ylim_upper=ylim_upper,
                        )

                    if page_idx == 0:
                        fig.suptitle(
                            f"{entity.dataset_name}: {entity.data_id} {"".join(symbol_list)}"
                        )
                    fig.supxlabel("Time (sec)")
                    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.95)
                    exporter.add_page(fig, pad_inches=0)
                    plt.close()
