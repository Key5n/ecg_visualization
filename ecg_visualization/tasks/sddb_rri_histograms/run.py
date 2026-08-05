from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import structlog
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.sddb import SDDB
from ecg_visualization.logging.config import configure_root_logging
from ecg_visualization.tasks.sddb_rri_histograms.config import (
    SDDBRRHistogramsConfig,
)
from ecg_visualization.tasks.visualize_datasets.rr_histogram_page import (
    RR_HISTOGRAM_BINS,
    RR_HISTOGRAM_XMAX_SEC,
    RR_HISTOGRAM_XMIN_SEC,
)
from ecg_visualization.visualization.export import pdf_exporter
from ecg_visualization.visualization.plotters import plot_histogram
from ecg_visualization.visualization.styles import apply_default_style

LOGGER = structlog.get_logger(__name__)

# Positions are specified in (row, column) order.
SUBPLOT_POSITIONS = ((0, 0), (1, 0), (0, 1), (1, 1))
ENTITY_TITLES = {
    "38": "Small median, small variance",
    "51": "Large median, small variance",
    "46": "Small median, large variance",
    "33": "Large median, large variance",
}


def sddb_rr_histograms(config: SDDBRRHistogramsConfig) -> None:
    """Export the four configured SDDB R-peak interval histograms to one PDF."""
    configure_root_logging()
    apply_default_style()

    if len(config.entity_ids) != len(SUBPLOT_POSITIONS):
        raise ValueError("Exactly four SDDB entity IDs are required")

    entities = tuple(
        SDDB.get_entity(entity_id=entity_id) for entity_id in config.entity_ids
    )
    figure = create_rr_histograms_figure(entities)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with pdf_exporter(str(config.output_path)) as exporter:
        exporter.add_page(figure, pad_inches=0)
    plt.close(figure)
    LOGGER.info("sddb_rr_histograms_saved", output_path=str(config.output_path))


def create_rr_histograms_figure(entities: tuple[ECGEntity, ...]) -> Figure:
    if len(entities) != len(SUBPLOT_POSITIONS):
        raise ValueError("Exactly four SDDB entities are required")

    figure, axes = plt.subplots(2, 2, figsize=(6, 4), sharex=True, sharey=True)
    for entity, position in zip(entities, SUBPLOT_POSITIONS, strict=True):
        _plot_rr_histogram(
            axes[position],
            entity,
            annotation_on_right=position[1] == 0,
        )

    figure.tight_layout()
    return figure


def _plot_rr_histogram(
    ax: Axes,
    entity: ECGEntity,
    *,
    annotation_on_right: bool = False,
) -> None:
    title = ENTITY_TITLES.get(entity.entity_id, f"SDDB entity {entity.entity_id}")
    intervals = entity.rr_intervals
    intervals = intervals[
        (intervals >= RR_HISTOGRAM_XMIN_SEC) & (intervals <= RR_HISTOGRAM_XMAX_SEC)
    ]

    if intervals.size:
        plot_histogram(
            ax,
            intervals,
            bins=RR_HISTOGRAM_BINS,
            weights=np.full(intervals.shape, 1.0 / intervals.size),
            title=title,
            xlabel="R-peak interval (sec)",
            ylabel="Proportion",
        )
        median = float(np.median(intervals))
        average = float(np.mean(intervals))
        standard_deviation = float(np.std(intervals))
        ax.axvline(median, color="tab:red", linestyle="--", linewidth=1.0)
        ax.text(
            0.98 if annotation_on_right else 0.02,
            0.95,
            (
                f"Med: {median:.2f} s\n"
                f"Ave: {average:.2f} s\n"
                f"SD: {standard_deviation:.2f} s"
            ),
            transform=ax.transAxes,
            fontsize=7,
            horizontalalignment="right" if annotation_on_right else "left",
            verticalalignment="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    else:
        ax.set_title(title)
        ax.set_xlabel("R-peak interval (sec)")
        ax.set_ylabel("Count")
        ax.text(
            0.5,
            0.5,
            "No intervals in range",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_xlim(RR_HISTOGRAM_XMIN_SEC, RR_HISTOGRAM_XMAX_SEC)
