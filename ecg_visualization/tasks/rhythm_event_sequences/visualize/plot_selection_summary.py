from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ecg_visualization.tasks.rhythm_event_sequences.config import (
    SegmentWindow,
)
from ecg_visualization.tasks.rhythm_event_sequences.utils import (
    SequenceSelectionResult,
    SequenceSelectionSuccess,
)
from ecg_visualization.visualization.export import pdf_exporter


@dataclass(frozen=True, slots=True)
class SelectionSummaryRow:
    entity_id: str
    status: str
    failure_reason: str
    train: str
    pre_ar: str
    ar: str
    test: str
    train_duration_sec: float | None


@dataclass(frozen=True, slots=True)
class SelectionSummaryStats:
    total: int
    succeeded: int
    failed: int
    train_count: int
    train_min_sec: float | None
    train_median_sec: float | None
    train_mean_sec: float | None
    train_max_sec: float | None
    train_std_sec: float | None


def prepare_selection_summary(
    results: Iterable[SequenceSelectionResult],
) -> tuple[list[SelectionSummaryRow], SelectionSummaryStats, list[float]]:
    rows: list[SelectionSummaryRow] = []
    train_durations_sec: list[float] = []

    for result in results:
        if isinstance(result, SequenceSelectionSuccess):
            segments_info = result.segments_info
            train_duration_sec = _duration_sec(segments_info.train)
            train_durations_sec.append(train_duration_sec)
            rows.append(
                SelectionSummaryRow(
                    entity_id=result.entity.entity_id,
                    status="succeeded",
                    failure_reason="",
                    train=_format_window(segments_info.train),
                    pre_ar=_format_window(segments_info.pre_ar),
                    ar=_format_window(segments_info.ar),
                    test=_format_window(segments_info.test),
                    train_duration_sec=train_duration_sec,
                )
            )
        else:
            rows.append(
                SelectionSummaryRow(
                    entity_id=result.entity.entity_id,
                    status="failed",
                    failure_reason=result.failure_reason,
                    train="",
                    pre_ar="",
                    ar="",
                    test="",
                    train_duration_sec=None,
                )
            )

    stats = _compute_stats(
        total=len(rows),
        succeeded=len(train_durations_sec),
        train_durations_sec=train_durations_sec,
    )
    return rows, stats, train_durations_sec


def plot_selection_summary(
    results: Iterable[SequenceSelectionResult],
    *,
    output_path: Path,
) -> None:
    results = list(results)
    rows, _, train_durations_sec = prepare_selection_summary(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pdf_exporter(str(output_path)) as exporter:
        for fig in _create_summary_figures(rows, train_durations_sec, results):
            exporter.add_page(fig, pad_inches=0.2)
            plt.close(fig)


def _compute_stats(
    *,
    total: int,
    succeeded: int,
    train_durations_sec: list[float],
) -> SelectionSummaryStats:
    durations = np.asarray(train_durations_sec, dtype=np.float64)
    failed = total - succeeded
    if durations.size == 0:
        return SelectionSummaryStats(
            total=total,
            succeeded=succeeded,
            failed=failed,
            train_count=0,
            train_min_sec=None,
            train_median_sec=None,
            train_mean_sec=None,
            train_max_sec=None,
            train_std_sec=None,
        )

    return SelectionSummaryStats(
        total=total,
        succeeded=succeeded,
        failed=failed,
        train_count=int(durations.size),
        train_min_sec=float(np.min(durations)),
        train_median_sec=float(np.median(durations)),
        train_mean_sec=float(np.mean(durations)),
        train_max_sec=float(np.max(durations)),
        train_std_sec=float(np.std(durations, ddof=1)) if durations.size > 1 else None,
    )


def _create_summary_figures(
    rows: list[SelectionSummaryRow],
    train_durations_sec: list[float],
    results: list[SequenceSelectionResult],
) -> list[Figure]:
    dataset_name = _get_dataset_name(results)
    figures = [
        _create_train_histogram_figure(train_durations_sec, dataset_name),
        _create_segment_timeline_figure(results, dataset_name),
        *_create_entity_table_figures(rows, dataset_name),
    ]
    return figures


def _get_dataset_name(results: list[SequenceSelectionResult]) -> str:
    dataset_names = sorted({result.entity.dataset.name for result in results})
    return ", ".join(dataset_names) if dataset_names else "Unknown dataset"


def _create_segment_timeline_figure(
    results: list[SequenceSelectionResult],
    dataset_name: str,
) -> Figure:
    minimum_display_width_pct = 0.5
    succeeded_results = sorted(
        (result for result in results if isinstance(result, SequenceSelectionSuccess)),
        key=lambda result: result.entity.entity_id,
    )
    row_count = max(1, len(succeeded_results))
    fig_height = max(5.5, min(18.0, 1.5 + row_count * 0.28))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.set_title(f"{dataset_name}: Selected Segment Timeline")
    ax.set_xlabel("Position within record (%)")
    ax.set_ylabel("Entity")

    if not succeeded_results:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No successful selections.",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return fig

    segment_colors = {
        "sinus_train": "#2a9d8f",
        "pre_ar": "#f4a261",
        "ar": "#e63946",
        "sinus_test": "#264653",
    }
    segment_labels = {
        "sinus_train": "sinus train",
        "pre_ar": "pre-AR",
        "ar": "AR",
        "sinus_test": "sinus test",
    }
    y_positions = np.arange(len(succeeded_results))
    for y_position, result in zip(y_positions, succeeded_results, strict=True):
        segments_info = result.segments_info
        total_duration_sec = (
            result.entity.signals.size / result.entity.dataset.sampling_rate_hz
        )
        for name, window in (
            ("sinus_train", segments_info.train),
            ("pre_ar", segments_info.pre_ar),
            ("ar", segments_info.ar),
            ("sinus_test", segments_info.test),
        ):
            if window is None:
                continue
            start_pct = 100 * window.start_sec / total_duration_sec
            end_pct = 100 * window.end_sec / total_duration_sec
            ax.barh(
                y_position,
                max(end_pct - start_pct, minimum_display_width_pct),
                left=start_pct,
                height=0.65,
                color=segment_colors[name],
                label=segment_labels[name],
            )

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles, strict=True))
    ax.legend(
        unique_labels.values(),
        unique_labels.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        frameon=False,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels([result.entity.entity_id for result in succeeded_results])
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#d9dee3", linewidth=0.6)
    fig.tight_layout()
    return fig


def _create_train_histogram_figure(
    train_durations_sec: list[float],
    dataset_name: str,
) -> Figure:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    _plot_train_histogram(ax, train_durations_sec)
    fig.suptitle(
        f"{dataset_name}: Train Sinus Duration Distribution",
        fontsize=16,
        y=0.96,
    )
    return fig


def _create_entity_table_figures(
    rows: list[SelectionSummaryRow],
    dataset_name: str,
) -> list[Figure]:
    rows_per_page = 28
    rows = sorted(rows, key=lambda row: row.entity_id)
    if not rows:
        row_pages = [[]]
    else:
        row_pages = [
            rows[start_idx : start_idx + rows_per_page]
            for start_idx in range(0, len(rows), rows_per_page)
        ]

    figures = []
    for page_idx, page_rows in enumerate(row_pages, start=1):
        fig, ax = plt.subplots(figsize=(16, 10))
        _plot_entity_table(ax, page_rows)
        fig.suptitle(
            f"{dataset_name}: Entity Selection Details ({page_idx}/{len(row_pages)})",
            fontsize=16,
            y=0.96,
        )
        figures.append(fig)
    return figures


def _plot_train_histogram(ax, train_durations_sec: list[float]) -> None:
    ax.set_title("Train sinus duration distribution")
    ax.set_xlabel("Duration (sec)")
    ax.set_ylabel("Entities")
    if not train_durations_sec:
        ax.text(
            0.5,
            0.5,
            "No successful selections.",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return

    bins = min(20, max(1, len(train_durations_sec)))
    ax.hist(train_durations_sec, bins=bins, color="#457b9d", edgecolor="white")


def _plot_entity_table(ax, rows: list[SelectionSummaryRow]) -> None:
    ax.axis("off")
    columns = [
        "entity_id",
        "status",
        "failure reason",
        "train start-end (dur)",
        "pre-ar start-end (dur)",
        "ar start-end (dur)",
        "test start-end (dur)",
    ]
    cell_text = [
        [
            row.entity_id,
            row.status,
            row.failure_reason,
            row.train,
            row.pre_ar,
            row.ar,
            row.test,
        ]
        for row in rows
    ]
    if not cell_text:
        cell_text = [["", "", "No entities found.", "", "", "", ""]]

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        loc="upper left",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.05, 0.09, 0.30, 0.14, 0.14, 0.14, 0.14],
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1, 1.15)

    for (row_idx, _col_idx), cell in table.get_celld().items():
        cell.set_linewidth(0.3)
        if row_idx == 0:
            cell.set_facecolor("#f1f3f5")
            cell.set_text_props(weight="bold")
        elif rows and rows[row_idx - 1].status == "failed":
            cell.set_facecolor("#fff5f5")


def _duration_sec(window: SegmentWindow) -> float:
    return window.end_sec - window.start_sec


def _format_window(window: SegmentWindow | None) -> str:
    if window is None:
        return ""
    duration_sec = _duration_sec(window)
    return f"{window.start_sec:.1f}-{window.end_sec:.1f} ({duration_sec:.1f})"
