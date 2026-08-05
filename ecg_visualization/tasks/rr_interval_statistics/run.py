from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import structlog
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ecg_visualization.core.dataset import ECGDataset
from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.physionet import load_data_sources
from ecg_visualization.logging.config import configure_root_logging
from ecg_visualization.tasks.rr_interval_statistics.config import (
    RRIntervalStatisticsConfig,
)
from ecg_visualization.visualization.export import pdf_exporter
from ecg_visualization.visualization.styles import apply_default_style

LOGGER = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EntityRRStatistics:
    entity_id: str
    average_sec: float
    median_sec: float
    standard_deviation_sec: float


def compute_entity_rr_statistics(entity: ECGEntity) -> EntityRRStatistics:
    """Compute population statistics over all RR intervals for one entity."""
    intervals = entity.rr_intervals
    return EntityRRStatistics(
        entity_id=entity.entity_id,
        average_sec=float(np.mean(intervals)),
        median_sec=float(np.median(intervals)),
        standard_deviation_sec=float(np.std(intervals)),
    )


def collect_dataset_statistics(
    dataset: ECGDataset,
) -> tuple[EntityRRStatistics, ...]:
    return tuple(
        compute_entity_rr_statistics(entity) for entity in dataset.get_entities()
    )


def rr_interval_statistics(config: RRIntervalStatisticsConfig) -> None:
    """Export distributions of per-entity RR statistics for each dataset."""
    configure_root_logging()
    apply_default_style()

    datasets = load_data_sources(config.dataset_ids)
    statistics_by_dataset: list[tuple[EntityRRStatistics, ...]] = []
    for dataset in datasets:
        LOGGER.info("rr_statistics_dataset_started", dataset_id=dataset.dataset_id)
        statistics = collect_dataset_statistics(dataset)
        statistics_by_dataset.append(statistics)
        LOGGER.info(
            "rr_statistics_dataset_finished",
            dataset_id=dataset.dataset_id,
            entity_count=len(statistics),
        )

    figure = create_rr_statistics_figure(datasets, statistics_by_dataset)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with pdf_exporter(str(config.output_path)) as exporter:
        exporter.add_page(figure, pad_inches=0)
    plt.close(figure)
    LOGGER.info("rr_interval_statistics_saved", output_path=str(config.output_path))


def create_rr_statistics_figure(
    datasets: list[ECGDataset],
    statistics_by_dataset: list[tuple[EntityRRStatistics, ...]],
) -> Figure:
    if len(datasets) != len(statistics_by_dataset):
        raise ValueError("Each dataset must have one statistics collection")
    if not datasets:
        raise ValueError("At least one dataset is required")
    if any(not statistics for statistics in statistics_by_dataset):
        raise ValueError("Every dataset must contain at least one entity")

    figure_height = max(4.0, 2.7 * len(datasets))
    figure, axes = plt.subplots(
        len(datasets),
        1,
        figsize=(8.27, figure_height),
        sharex=True,
        squeeze=False,
    )
    fields = (
        ("average_sec", "Ave"),
        ("median_sec", "Med"),
        ("standard_deviation_sec", "SD"),
    )
    flat_axes = axes.ravel()
    for dataset_index, (ax, dataset, statistics) in enumerate(
        zip(flat_axes, datasets, statistics_by_dataset, strict=True)
    ):
        values = [
            np.asarray([getattr(statistic, field) for statistic in statistics])
            for field, _ in fields
        ]
        panel_label = chr(ord("a") + dataset_index)
        _plot_dataset_distributions(
            ax,
            values,
            labels=[label for _, label in fields],
            title=f"({panel_label}) {dataset.dataset_id.upper()}",
        )

    flat_axes[-1].set_xlabel("sec")
    figure.suptitle("Distribution of per-entity RR interval statistics")
    figure.tight_layout()
    return figure


def _plot_dataset_distributions(
    ax: Axes,
    values: list[np.ndarray],
    *,
    labels: list[str],
    title: str,
) -> None:
    positions = np.arange(1, len(values) + 1)
    boxplot = ax.boxplot(
        values,
        positions=positions,
        widths=0.55,
        orientation="horizontal",
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "tab:red", "linewidth": 1.5},
    )
    for box in boxplot["boxes"]:
        box.set_facecolor("tab:blue")
        box.set_alpha(0.25)

    rng = np.random.default_rng(0)
    for position, dataset_values in zip(positions, values, strict=True):
        jitter = rng.uniform(-0.14, 0.14, size=dataset_values.size)
        ax.scatter(
            dataset_values,
            position + jitter,
            s=12,
            alpha=0.55,
            color="tab:blue",
            edgecolors="none",
            zorder=3,
        )

    ax.set_title(title)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
