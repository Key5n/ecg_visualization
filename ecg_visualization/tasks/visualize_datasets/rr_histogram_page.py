from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.visualization.export import PdfExporter
from ecg_visualization.visualization.plotters import plot_histogram

RR_HISTOGRAM_XMIN_SEC = 0.0
RR_HISTOGRAM_XMAX_SEC = 2.0
RR_HISTOGRAM_BIN_WIDTH_SEC = 0.025
RR_HISTOGRAM_BINS = np.arange(
    RR_HISTOGRAM_XMIN_SEC,
    RR_HISTOGRAM_XMAX_SEC + RR_HISTOGRAM_BIN_WIDTH_SEC,
    RR_HISTOGRAM_BIN_WIDTH_SEC,
)


def render_rr_interval_histogram_page(
    entity: ECGEntity,
    exporter: PdfExporter,
) -> None:
    rr_intervals_sec = entity.rr_intervals

    fig, ax = plt.subplots(figsize=(8.27, 5.0))
    if rr_intervals_sec.size > 0:
        rr_intervals_in_range = rr_intervals_sec[
            (rr_intervals_sec >= RR_HISTOGRAM_XMIN_SEC)
            & (rr_intervals_sec <= RR_HISTOGRAM_XMAX_SEC)
        ]
        if rr_intervals_in_range.size > 0:
            plot_histogram(
                ax,
                rr_intervals_in_range,
                bins=RR_HISTOGRAM_BINS,
                title=f"{entity.dataset.name} / {entity.entity_id} RR intervals",
                xlabel="R-peak interval (sec)",
                ylabel="Count",
            )
        else:
            ax.set_title(f"{entity.dataset.name} / {entity.entity_id} RR intervals")
            ax.set_xlabel("R-peak interval (sec)")
            ax.set_ylabel("Count")
        ax.set_xlim(RR_HISTOGRAM_XMIN_SEC, RR_HISTOGRAM_XMAX_SEC)

        median_rr_interval = float(np.median(rr_intervals_sec))
        ax.axvline(
            median_rr_interval,
            color="tab:red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
        )
        ylim_upper = ax.get_ylim()[1]
        ax.text(
            median_rr_interval,
            ylim_upper * 0.95,
            f"Median: {median_rr_interval:.2f}s",
            rotation=90,
            fontsize=8,
            color="tab:red",
            horizontalalignment="right",
            verticalalignment="top",
        )
    else:
        ax.set_title(f"{entity.dataset.name} / {entity.entity_id} RR intervals")
        ax.set_xlabel("R-peak interval (sec)")
        ax.set_ylabel("Count")
        ax.text(
            0.5,
            0.5,
            "Not enough R-peaks to compute intervals.",
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
        )

    fig.tight_layout()
    exporter.add_page(fig, pad_inches=0)
    plt.close(fig)
