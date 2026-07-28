from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from ecg_visualization.core.analysis import NormalSegmentConfig, extract_normal_segment
from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.visualization.export import PdfExporter
from ecg_visualization.visualization.text import sanitize_annotation_text

# WFDB beat annotation codes:
# https://physionet.org/physiotools/wpg/wpg_36.htm
BEAT_ANNOTATION_DESCRIPTIONS = {
    "N": "Normal beat",
    "L": "Left bundle branch block beat",
    "R": "Right bundle branch block beat",
    "B": "Unspecified bundle branch block beat",
    "A": "Atrial premature beat",
    "a": "Aberrated atrial premature beat",
    "J": "Junctional premature beat",
    "S": "Supraventricular premature or ectopic beat",
    "V": "Premature ventricular contraction",
    "r": "R-on-T premature ventricular contraction",
    "F": "Fusion of ventricular and normal beat",
    "e": "Atrial escape beat",
    "j": "Junctional escape beat",
    "n": "Supraventricular escape beat",
    "E": "Ventricular escape beat",
    "/": "Paced beat",
    "f": "Fusion of paced and normal beat",
    "Q": "Unclassifiable beat",
    "?": "Beat not classified during learning",
}

NON_BEAT_ANNOTATION_DESCRIPTIONS = {
    "[": "Start of ventricular flutter or fibrillation",
    "!": "Ventricular flutter wave",
    "]": "End of ventricular flutter or fibrillation",
    "x": "Non-conducted P-wave",
    "(": "Waveform onset",
    ")": "Waveform end",
    "p": "P-wave peak",
    "t": "T-wave peak",
    "u": "U-wave peak",
    "^": "Non-captured pacemaker artifact",
    "|": "Isolated QRS-like artifact",
    "~": "Change in signal quality",
    "+": "Rhythm change",
    "s": "ST-segment change",
    "T": "T-wave change",
    "*": "Systole",
    "D": "Diastole",
    "=": "Measurement annotation",
    '"': "Comment annotation",
    "@": "Link to external data",
}


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
    value_x = 0.40
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
        y -= 0.045 * max(1, value.count("\n") + 1)

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
        (
            "Beat annotation codes",
            _format_annotation_codes(entity, BEAT_ANNOTATION_DESCRIPTIONS),
        ),
        (
            "Non-beat annotation codes",
            _format_annotation_codes(entity, NON_BEAT_ANNOTATION_DESCRIPTIONS),
        ),
        ("Aux notes", _format_aux_note_summary(entity)),
        ("RR intervals", _format_rr_statistics(entity)),
        (
            "Normal segment",
            _format_normal_segment_summary(entity, normal_segment_config),
        ),
    ]


def _format_annotation_codes(
    entity: ECGEntity,
    descriptions: dict[str, str],
) -> str:
    symbols = sorted(set(entity.annotation.symbol) & descriptions.keys())
    if not symbols:
        return "None recorded"

    return "\n".join(f"{symbol}: {descriptions[symbol]}" for symbol in symbols)


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
    notes = [
        cleaned_note
        for note in entity.aux_notes
        if (cleaned_note := sanitize_annotation_text(note))
    ]
    if not notes:
        return "None recorded"

    note_counts = Counter(notes)
    summary_parts = [
        f"{note} ({note_counts[note]})"
        for note in sorted(note_counts, key=lambda n: (-note_counts[n], n))
    ]
    summary = ", ".join(summary_parts)
    return f"{len(notes):,} entries | {summary}"
