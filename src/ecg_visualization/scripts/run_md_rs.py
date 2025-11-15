import os
from typing import Any, Final

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
from ecg_visualization.models.md_rs.md_rs import MDRS
from ecg_visualization.utils.timed_sequence import TimedSequence
from ecg_visualization.utils.utils import (
    padding_reshape,
    prepare_sequences,
    sliding_window_sequences,
)
from ecg_visualization.visualization.export import pdf_exporter
from ecg_visualization.visualization.layouts import (
    PaginationConfig,
    create_page_layout,
    paginate_signals,
)
from ecg_visualization.visualization.plotters import (
    highlight_windows,
    plot_anomaly_score,
    plot_histogram,
    plot_normal_beats,
    plot_signal,
    plot_symbols,
)
from ecg_visualization.visualization.styles import (
    apply_default_style,
    EXTREME_INTERVAL_COLOR,
)
from ecg_visualization.visualization.limits import compute_ylim

MIN_RR_INTERVAL_SEC = 0.6
MAX_RR_INTERVAL_SEC = 1.0
RR_WINDOW_BEATS = 100
WINDOW_SIZE = 10
HISTOGRAM_WINDOW_SIZES = (10, 50, 100)
PAGINATION_CONFIG = PaginationConfig()

MD_RS_CONFIG: Final[dict[str, Any]] = {
    "N_x": 256,
    "input_scale": 0.5,
    "rho": 0.9,
    "leaking_rate": 0.3,
    "delta": 1e-3,
    "trans_length": 10,
    "N_x_tilde": 128,
    "seed": 0,
}


def run_md_rs() -> None:
    apply_default_style()
    data_sources: list[ECG_Dataset] = [
        CUDB(),
        AFPDB(),
        MITDB(),
        AFDB(),
        LTAFDB(),
        SHDBAF(),
        SDDB(),
    ]

    task_name: Final[str] = "run_md_rs"
    root_result_dir = os.path.join("result", task_name)

    for data_source in tqdm(data_sources):
        dataset_result_dir = os.path.join(root_result_dir, data_source.dataset_id)
        os.makedirs(dataset_result_dir, exist_ok=True)

        for entity in tqdm(data_source.data_entities):
            result_file_path = os.path.join(
                dataset_result_dir, f"{entity.entity_id}.pdf"
            )
            with pdf_exporter(result_file_path) as exporter:
                for window_size in HISTOGRAM_WINDOW_SIZES:
                    windows = entity.get_window_durations(window_size)
                    if not windows:
                        continue

                    durations = np.array(
                        [(end - start) / entity.sr for start, end in windows],
                        dtype=float,
                    )

                    histogram_fig, histogram_ax = plt.subplots(figsize=(8, 4))
                    plot_histogram(
                        histogram_ax,
                        durations,
                        bins="auto",
                        title=f"{entity.dataset_name}: {entity.entity_id} "
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

                ts_paged = paginate_signals(
                    entity.signals, entity.sr, PAGINATION_CONFIG
                )
                shape = ts_paged.shape
                signals_paged = padding_reshape(entity.signals, shape)
                n_rows = shape[1] if len(shape) >= 2 else 0

                ylim_lower, ylim_upper = compute_ylim(
                    entity.signals, lower_bound=-5.0, upper_bound=5.0
                )

                symbol_list = sorted(set(entity.annotation.symbol))
                tqdm.write(
                    f"{entity.entity_id}, {entity.dataset_name} {"".join(symbol_list)} "
                    f"The number of extreme window: {len(extreme_windows)}"
                )

                annotation_events = TimedSequence.from_time_axis(
                    values=entity.annotation.symbol,
                    time_axis=np.asarray(entity.annotation.sample, dtype=float)
                    / entity.sr,
                )
                beat_times = entity.beats / entity.sr
                beat_sequence = TimedSequence.from_time_axis(
                    values=np.zeros_like(
                        beat_times
                    ),  # placeholder values; only times are used
                    time_axis=beat_times,
                )

                try:
                    normal_window = entity.extract_normal_segment()
                except ValueError:
                    tqdm.write(f"Skipping {entity.entity_id}: no normal segment found.")
                    continue
                rr_intervals = entity.compute_rr_intervals()

                train_windows = sliding_window_sequences(normal_window, WINDOW_SIZE)
                test_windows = sliding_window_sequences(rr_intervals, WINDOW_SIZE)

                train_sequence, test_sequence = prepare_sequences(
                    train_windows, test_windows
                )

                config = {**MD_RS_CONFIG, "N_u": train_sequence.shape[1]}

                model = MDRS(**config)
                model.train(train_sequence)

                model.reset_states()

                scores = model.predict(test_sequence)
                score_times = beat_times[WINDOW_SIZE:]
                score_sequence = TimedSequence.from_time_axis(
                    values=scores,
                    time_axis=score_times,
                )

                score_ylim_lower, score_ylim_upper = compute_ylim(scores)

                for page_idx, (signals, ts_row) in enumerate(
                    zip(signals_paged, ts_paged, strict=False)
                ):
                    fig, axs = create_page_layout(n_rows)

                    for signal, ts, ax in zip(signals, ts_row, axs, strict=False):
                        window_start, window_end = ts[0], ts[-1]
                        beats_in_window = beat_sequence.slice_between(
                            window_start, window_end
                        )
                        scores_in_window = score_sequence.slice_between(
                            window_start, window_end
                        )
                        plot_signal(
                            ax,
                            ts,
                            signal,
                            ylim_lower=ylim_lower,
                            ylim_upper=ylim_upper,
                        )
                        score_ax = ax.twinx()
                        plot_anomaly_score(
                            score_ax,
                            scores_in_window.times.tolist(),
                            scores_in_window.values.tolist(),
                            ylim_lower=score_ylim_lower,
                            ylim_upper=score_ylim_upper,
                            label="Anomaly Score",
                        )

                        plot_normal_beats(
                            ax,
                            beats_in_window.times.tolist(),
                            ylim_lower=ylim_lower,
                        )
                        symbols_in_window = annotation_events.slice_between(
                            window_start, window_end
                        )
                        plot_symbols(
                            ax,
                            symbols_in_window.samples,
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
                            f"{entity.dataset_name}: {entity.entity_id} {"".join(symbol_list)} {RR_WINDOW_BEATS}"
                        )
                    fig.supxlabel("Time (sec)")
                    fig.subplots_adjust(left=0.08, right=0.94, bottom=0.05, top=0.95)
                    exporter.add_page(fig, pad_inches=0)
                    plt.close()
