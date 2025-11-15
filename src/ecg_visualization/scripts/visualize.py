from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.artifacts import FileSystemArtifactStore
from optuna.exceptions import OptunaError
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
    ECG_Entity,
)
from ecg_visualization.logging import configure_optuna_logging
from ecg_visualization.utils.optuna_record import VisualizationRecord
from ecg_visualization.utils.utils import padding_reshape
from ecg_visualization.visualization.export import pdf_exporter
from ecg_visualization.visualization.layouts import (
    PaginationConfig,
    create_page_layout,
    paginate_signals,
)
from ecg_visualization.visualization.limits import compute_ylim
from ecg_visualization.visualization.plotters import (
    highlight_windows,
    plot_anomaly_score,
    plot_normal_beats,
    plot_signal,
    plot_symbols,
)
from ecg_visualization.visualization.styles import (
    apply_default_style,
    EXTREME_INTERVAL_COLOR,
)

RR_WINDOW_BEATS = 100
PAGINATION_CONFIG = PaginationConfig()
ARTIFACT_ROOT = Path("result") / "artifacts"
VISUALIZATION_ROOT = Path("result") / "visualize"


def visualize_all_entities():
    data_sources: list[ECG_Dataset] = [
        CUDB(),
        AFPDB(),
        MITDB(),
        AFDB(),
        LTAFDB(),
        SHDBAF(),
        SDDB(),
    ]

    for data_source in tqdm(data_sources):
        for entity in tqdm(data_source.data_entities):
            visualize_entity(entity)


def visualize_entity(entity: ECG_Entity):
    apply_default_style()
    configure_optuna_logging()

    storage_name = "mysql+pymysql://root:foo@localhost:3306/optuna"
    study_name = f"{entity.dataset_name} {entity.data_id}"

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    artifact_store = FileSystemArtifactStore(base_path=str(ARTIFACT_ROOT))
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
    except OptunaError as exc:
        tqdm.write(f"Skipping {study_name}: failed to load study ({exc})")
        return

    if not study.trials:
        tqdm.write(f"Skipping {study_name}: no trials available.")
        return

    try:
        vis_record = VisualizationRecord.from_trial(
            study.best_trial,
            study_name=study.study_name,
            artifact_store=artifact_store,
        )
    except ValueError as exc:
        tqdm.write(f"Skipping {study_name}: {exc}")
        return

    signals_sequence = vis_record.signal_sequence
    score_sequence = vis_record.score_sequence
    annotation_sequence = vis_record.annotation_sequence
    beat_sequence = vis_record.beat_sequence

    ts_paged = paginate_signals(
        len(signals_sequence.values), entity.sr, PAGINATION_CONFIG
    )
    n_rows = PAGINATION_CONFIG.rows_per_page

    ylim_lower, ylim_upper = compute_ylim(
        signals_sequence.values,
        lower_bound=-5.0,
        upper_bound=5.0,
    )
    score_ylim_lower, score_ylim_upper = compute_ylim(score_sequence.values)

    extreme_windows = entity.get_extreme_rr_windows(
        RR_WINDOW_BEATS,
        lower_percentile=5.0,
        upper_percentile=95.0,
    )
    symbol_list = sorted(set(annotation_sequence.values.tolist()))

    dataset_dir = VISUALIZATION_ROOT / entity.dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_path = dataset_dir / f"{entity.data_id}.pdf"

    with pdf_exporter(str(output_path)) as exporter:
        for page_idx, ts_row in enumerate(ts_paged):
            fig, axs = create_page_layout(n_rows)
            for ts, ax in zip(ts_row, axs):
                window_start, window_end = ts[0], ts[-1]
                signals_in_window = signals_sequence.slice_between(
                    window_start, window_end
                )
                beats_in_window = beat_sequence.slice_between(window_start, window_end)
                scores_in_window = score_sequence.slice_between(
                    window_start, window_end
                )
                symbols_in_window = annotation_sequence.slice_between(
                    window_start, window_end
                )

                plot_signal(
                    ax,
                    ts,
                    signals_in_window.values,
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
                    f"{entity.dataset_name}: {entity.data_id} {"".join(symbol_list)} "
                    f"{RR_WINDOW_BEATS}"
                )
            fig.supxlabel("Time (sec)")
            fig.subplots_adjust(left=0.08, right=0.94, bottom=0.05, top=0.95)
            exporter.add_page(fig, pad_inches=0)
            plt.close(fig)

    tqdm.write(f"Saved visualization to {output_path}")
