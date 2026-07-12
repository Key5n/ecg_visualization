from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.physionet import DATASET_CLASSES
from ecg_visualization.logging.config import configure_root_logging
from ecg_visualization.scripts.visualize_datasets.rr_histogram_page import (
    render_rr_interval_histogram_page,
)
from ecg_visualization.scripts.visualize_datasets.signal_page import (
    decorate_signal_page,
    render_signal_row,
)
from ecg_visualization.scripts.visualize_datasets.summary_page import (
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

DEFAULT_OUTPUT_DIR = Path("result") / "visualize-datasets"
PAGINATION_CONFIG = PaginationConfig(seconds_per_row=10, rows_per_page=6)


def visualize_datasets() -> None:
    configure_root_logging()
    apply_default_style()

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    for dataset_cls in DATASET_CLASSES:
        dataset = dataset_cls()
        dataset_output_dir = DEFAULT_OUTPUT_DIR / dataset_cls.dataset_id
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info(
            "Visualizing dataset %s (%d entities)",
            dataset_cls.dataset_id,
            len(dataset.data_entities),
        )
        for entity in tqdm(dataset.data_entities, desc=dataset_cls.dataset_id):
            output_path = dataset_output_dir / f"{entity.entity_id}.pdf"
            _export_entity_pdf(entity, output_path=output_path)
            total_processed += 1

    LOGGER.info(
        "Finished dataset visualization. processed=%d output_dir=%s",
        total_processed,
        DEFAULT_OUTPUT_DIR,
    )


def _export_entity_pdf(
    entity: ECGEntity,
    *,
    output_path: Path,
) -> None:
    ts_paged = paginate_signals(
        entity.signals.size,
        int(entity.sr),
        PAGINATION_CONFIG,
    )
    signal_ylim = compute_ylim(
        entity.signals,
        lower_bound=-5.0,
        upper_bound=5.0,
    )

    with pdf_exporter(str(output_path)) as exporter:
        render_entity_summary_page(entity, exporter)
        render_rr_interval_histogram_page(entity, exporter)
        for page_idx, ts_row in enumerate(ts_paged):
            fig, axs = create_page_layout(PAGINATION_CONFIG.rows_per_page)
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
