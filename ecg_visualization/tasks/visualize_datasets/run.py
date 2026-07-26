from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ecg_visualization.core.analysis import NormalSegmentConfig
from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.physionet import load_data_sources
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
    datasets = load_data_sources(config.dataset_ids)
    pagination_config = config.pagination

    total_entities = sum(len(dataset.entity_ids) for dataset in datasets)
    total_processed = 0
    for dataset in datasets:
        (config.output_dir / dataset.dataset_id).mkdir(parents=True, exist_ok=True)
        LOGGER.info(
            "Visualizing dataset %s (%d entities)",
            dataset.dataset_id,
            len(dataset.entity_ids),
        )

    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        futures = [
            executor.submit(
                _export_entity_pdf,
                entity,
                output_path=(
                    config.output_dir / dataset.dataset_id / f"{entity.entity_id}.pdf"
                ),
                pagination_config=pagination_config,
                normal_segment_config=config.normal_segment,
                signal_ylim_bounds=(
                    config.signal_ylim_lower,
                    config.signal_ylim_upper,
                ),
            )
            for dataset in datasets
            for entity in dataset.get_entities()
        ]

        for future in as_completed(futures):
            try:
                output_path = future.result()
            except Exception:
                LOGGER.exception("Failed to export entity PDF")
                continue
            LOGGER.info("Saved entity PDF to %s", output_path)
            total_processed += 1

    LOGGER.info(
        "Finished dataset visualization. processed=%d failed=%d output_dir=%s",
        total_processed,
        total_entities - total_processed,
        config.output_dir,
    )


def _export_entity_pdf(
    entity: ECGEntity,
    *,
    output_path: Path,
    pagination_config: PaginationConfig,
    normal_segment_config: NormalSegmentConfig,
    signal_ylim_bounds: tuple[float, float],
) -> Path:
    LOGGER.info("Starting PDF export of entity=%s", entity)
    apply_default_style()
    ts_paged = paginate_signals(
        entity.signals.size,
        int(entity.dataset.sampling_rate_hz),
        pagination_config,
    )
    signal_ylim = compute_ylim(
        entity.signals,
        lower_bound=signal_ylim_bounds[0],
        upper_bound=signal_ylim_bounds[1],
    )

    with pdf_exporter(str(output_path)) as exporter:
        render_entity_summary_page(entity, exporter, normal_segment_config)
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
    return output_path
