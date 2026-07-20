from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.physionet import _load_data_sources
from ecg_visualization.logging.config import configure_root_logging
from ecg_visualization.tasks.visualize_datasets.config import (
    VisualizeDatasetsConfig,
)
from ecg_visualization.tasks.visualize_datasets.rr_histogram_page import (
    render_rr_interval_histogram_page,
)
from ecg_visualization.tasks.visualize_datasets.signal_page import (
    decorate_signal_page,
    render_signal_row,
)
from ecg_visualization.tasks.visualize_datasets.summary_page import (
    render_entity_summary_page,
)
from ecg_visualization.visualization.export import pdf_exporter
from ecg_visualization.visualization.layouts import (
    PaginationConfig,
    create_page_layout,
    paginate_signals,
)
from ecg_visualization.visualization.limits import compute_ylim
from ecg_visualization.visualization.styles import apply_default_style

LOGGER = logging.getLogger(__name__)


def visualize_datasets(config: VisualizeDatasetsConfig) -> None:
    configure_root_logging()
    apply_default_style()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = _load_data_sources(config.dataset_ids)
    pagination_config = config.pagination

    total_processed = 0
    for dataset in datasets:
        dataset_output_dir = config.output_dir / dataset.dataset_id
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info(
            "Visualizing dataset %s (%d entities)",
            dataset.dataset_id,
            len(dataset.entity_ids),
        )
        for entity in tqdm(dataset.get_entities(), desc=dataset.dataset_id):
            output_path = dataset_output_dir / f"{entity.entity_id}.pdf"
            _export_entity_pdf(
                entity,
                output_path=output_path,
                pagination_config=pagination_config,
                signal_ylim_bounds=(
                    config.signal_ylim_lower,
                    config.signal_ylim_upper,
                ),
            )
            total_processed += 1

    LOGGER.info(
        "Finished dataset visualization. processed=%d output_dir=%s",
        total_processed,
        config.output_dir,
    )


def _export_entity_pdf(
    entity: ECGEntity,
    *,
    output_path: Path,
    pagination_config: PaginationConfig,
    signal_ylim_bounds: tuple[float, float],
) -> None:
    ts_paged = paginate_signals(
        entity.signals.size,
        int(entity.sr),
        pagination_config,
    )
    signal_ylim = compute_ylim(
        entity.signals,
        lower_bound=signal_ylim_bounds[0],
        upper_bound=signal_ylim_bounds[1],
    )

    with pdf_exporter(str(output_path)) as exporter:
        render_entity_summary_page(entity, exporter)
        render_rr_interval_histogram_page(entity, exporter)
        for page_idx, ts_row in enumerate(ts_paged):
            fig, axs = create_page_layout(pagination_config.rows_per_page)
            for ts, ax in zip(ts_row, np.atleast_1d(axs), strict=True):
                render_signal_row(
                    ax=ax,
                    ts=ts,
                    entity=entity,
                    signal_ylim=signal_ylim,
                )
            decorate_signal_page(fig=fig, entity=entity, page_idx=page_idx)
            exporter.add_page(fig, pad_inches=0)
            plt.close(fig)
