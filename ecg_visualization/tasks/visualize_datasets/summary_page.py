from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from ecg_visualization.core.analysis import NormalSegmentConfig, extract_normal_segment
from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.visualization.export import PdfExporter


def render_entity_summary_page(
    entity: ECGEntity,
    exporter: PdfExporter,
    normal_segment_config: NormalSegmentConfig,
) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")

    rows = _entity_summary_rows(entity, normal_segment_config)
    title = f"{entity.dataset.name}: {entity.entity_id}"
    ax.text(
        0.05,
        0.96,
        title,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        verticalalignment="top",
    )

    y = 0.90
    label_x = 0.05
    value_x = 0.32
    for label, value in rows:
        ax.text(
            label_x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            verticalalignment="top",
        )
        ax.text(
            value_x,
            y,
            value,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            wrap=True,
        )
        y -= 0.045

    exporter.add_page(fig, pad_inches=0)
    plt.close(fig)


def _entity_summary_rows(
    entity: ECGEntity,
    normal_segment_config: NormalSegmentConfig,
) -> list[tuple[str, str]]:
    signal_samples = int(entity.signals.size)
    duration_sec = (
        signal_samples / entity.dataset.sampling_rate_hz
        if entity.dataset.sampling_rate_hz
        else 0.0
    )
    annotation_symbols = ", ".join(sorted(set(entity.annotation.symbol))) or "-"

    return [
        ("Entity ID", entity.entity_id),
        ("Dataset", f"{entity.dataset.name} ({entity.dataset.dataset_id})"),
        ("Sampling rate", f"{entity.dataset.sampling_rate_hz} Hz"),
        (
            "Signal length",
            f"{signal_samples:,} samples (~{duration_sec / 60:.2f} min)",
        ),
        ("Beats", f"{entity.beats.size:,}"),
        ("Annotations", f"{len(entity.annotation.sample):,} symbols"),
        ("Annotation types", annotation_symbols),
        ("Aux notes", _format_aux_note_summary(entity)),
        ("RR intervals", _format_rr_statistics(entity)),
        (
            "Normal segment",
            _format_normal_segment_summary(entity, normal_segment_config),
        ),
    ]


def _format_rr_statistics(entity: ECGEntity) -> str:
    rr_intervals = entity.rr_intervals
    if rr_intervals.size == 0:
        return "Unavailable"

    rr_min = float(np.min(rr_intervals))
    rr_max = float(np.max(rr_intervals))
    rr_mean = float(np.mean(rr_intervals))
    rr_median = float(np.median(rr_intervals))
    return (
        f"{rr_intervals.size:,} intervals | mean={rr_mean:.3f}s "
        f"median={rr_median:.3f}s [{rr_min:.3f}s, {rr_max:.3f}s]"
    )


def _format_normal_segment_summary(
    entity: ECGEntity,
    normal_segment_config: NormalSegmentConfig,
) -> str:
    try:
        normal_segment = extract_normal_segment(entity, normal_segment_config)
    except ValueError as exc:
        return f"Unavailable ({exc})"

    duration_min = normal_segment.duration / 60
    return (
        f"{normal_segment.length:,} beats spanning {duration_min:.2f} min "
        f"(start={normal_segment.start_time:.1f}s, "
        f"end={normal_segment.end_time:.1f}s)"
    )


def _format_aux_note_summary(entity: ECGEntity) -> str:
    notes = [note.strip() for note in entity.aux_notes if note.strip()]
    if not notes:
        return "None recorded"

    note_counts = Counter(notes)
    summary_parts = [
        f"{note} ({note_counts[note]})"
        for note in sorted(note_counts, key=lambda n: (-note_counts[n], n))
    ]
    summary = ", ".join(summary_parts)
    return f"{len(notes):,} entries | {summary}"
