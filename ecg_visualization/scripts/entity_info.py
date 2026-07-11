from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ecg_visualization.datasets.dataset import DATASET_REGISTRY, ECGEntity


def entity_info(argv: Sequence[str] | None = None) -> None:
    """
    Entry point to display a concise summary of a single ECG entity.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_datasets:
        _print_supported_datasets()
        return

    if not args.dataset_id or not args.entity_id:
        parser.error(
            "--dataset-id and --entity-id are required unless --list-datasets is used."
        )

    dataset_id = args.dataset_id.lower()
    dataset_cls = DATASET_REGISTRY.get(dataset_id)
    if dataset_cls is None:
        parser.error(
            f"Unknown dataset id '{args.dataset_id}'. "
            f"Use --list-datasets to see available options."
        )

    try:
        entity = dataset_cls._load_entity(args.entity_id)
    except FileNotFoundError as exc:  # pragma: no cover - depends on local data
        raise SystemExit(str(exc)) from exc

    record_path = Path(dataset_cls.dir_path) / args.entity_id
    _print_entity_summary(entity, record_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show basic information for a PhysioNet ECG record."
    )
    parser.add_argument(
        "--dataset-id",
        help="Dataset identifier (e.g., cudb, mitdb). Use --list-datasets to inspect all options.",
    )
    parser.add_argument(
        "--entity-id",
        help="Entity/record identifier inside the selected dataset (e.g., 00001).",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List dataset identifiers and exit.",
    )
    return parser


def _print_supported_datasets() -> None:
    print("Available datasets:")
    for dataset_id, dataset_cls in DATASET_REGISTRY.items():
        print(f"  - {dataset_id}: {dataset_cls.name}")


def _print_entity_summary(entity: ECGEntity, record_path: Path) -> None:
    signal_samples = int(entity.signals.size)
    duration_sec = signal_samples / entity.sr if entity.sr else 0.0
    beat_count = int(entity.beats.size)
    annotation_count = len(entity.annotation.sample)
    annotation_symbols = ", ".join(sorted(set(entity.annotation.symbol))) or "-"

    rr_stats_str = _format_rr_statistics(entity)
    normal_segment_summary = _format_normal_segment_summary(entity)

    rows: list[tuple[str, str]] = [
        ("Entity ID", entity.entity_id),
        ("Dataset", f"{entity.dataset_name} ({entity.dataset_id})"),
        ("Record path", str(record_path)),
        ("Sampling rate", f"{entity.sr} Hz"),
        (
            "Signal length",
            f"{signal_samples:,} samples (~{duration_sec / 60:.2f} min)",
        ),
        ("Beats", f"{beat_count:,}"),
        ("Annotations", f"{annotation_count:,} symbols"),
        ("Annotation types", annotation_symbols),
        ("Aux notes", _format_aux_note_summary(entity)),
        ("RR intervals", rr_stats_str),
        ("Normal segment", normal_segment_summary),
    ]
    _print_rows(rows)


def _format_rr_statistics(entity: ECGEntity) -> str:
    try:
        rr_intervals = entity.compute_rr_intervals()
    except ValueError as exc:
        return f"Unavailable ({exc})"

    rr_min = float(np.min(rr_intervals))
    rr_max = float(np.max(rr_intervals))
    rr_mean = float(np.mean(rr_intervals))
    return (
        f"{rr_intervals.size} intervals | mean={rr_mean:.3f}s "
        f"[{rr_min:.3f}s, {rr_max:.3f}s]"
    )


def _format_normal_segment_summary(entity: ECGEntity) -> str:
    try:
        normal_segment = entity.extract_normal_segment()
    except ValueError as exc:
        return f"Unavailable ({exc})"

    duration_min = normal_segment.duration / 60
    return (
        f"{normal_segment.length} beats spanning {duration_min:.2f} min "
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
    return f"{len(notes)} entries | {summary}"


def _print_rows(rows: Iterable[tuple[str, str]]) -> None:
    label_width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label:<{label_width}} : {value}")
