from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import structlog

from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.logging.config import configure_root_logging
from ecg_visualization.tasks.config import save_config_text
from ecg_visualization.tasks.rhythm_event_sequences.config import (
    RhythmEventSequencesConfig,
    RRHistogramConfig,
    SinusExtractionConfig,
)
from ecg_visualization.tasks.rhythm_event_sequences.utils import (
    ConcatenatedSequence,
    SequenceSelectionSuccess,
    iter_sequence_selection_results,
)
from ecg_visualization.tasks.rhythm_event_sequences.visualize.export_concatenated_pdf import (
    export_concatenated_pdf,
)
from ecg_visualization.tasks.rhythm_event_sequences.visualize.plot_selection_summary import (
    plot_selection_summary,
)
from ecg_visualization.visualization.layouts import PaginationConfig
from ecg_visualization.visualization.styles import apply_default_style

LOGGER = structlog.get_logger(__name__)


def rhythm_event_sequence_visualize(config: RhythmEventSequencesConfig) -> None:
    configure_root_logging()
    apply_default_style()
    config.visualize_output_dir.mkdir(parents=True, exist_ok=True)
    save_config_text(config, config.config_path)
    pagination_config = config.pagination

    processed = 0
    results = list(iter_sequence_selection_results(config))
    successes = [
        result for result in results if isinstance(result, SequenceSelectionSuccess)
    ]

    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        futures = [
            executor.submit(
                _export_pdf,
                result.entity,
                result.concat,
                output_dir=config.visualize_output_dir,
                pagination_config=pagination_config,
                rr_histogram_config=config.rr_histogram,
                sinus_extraction_config=config.sinus_extraction,
                segment_colors=config.segment_colors,
                segment_labels=config.segment_labels,
            )
            for result in successes
        ]
        for future in as_completed(futures):
            try:
                output_path = future.result()
            except Exception:
                LOGGER.exception("Failed to export concatenated signal PDF")
                continue
            LOGGER.info("concatenated_signal_pdf_saved", output_path=str(output_path))
            processed += 1

    plot_selection_summary(results, output_path=config.summary_path)
    LOGGER.info(
        "sequence_selection_summary_saved", output_path=str(config.summary_path)
    )

    LOGGER.info(
        "visualization_finished",
        processed=processed,
        failed=len(results) - processed,
        output_dir=str(config.visualize_output_dir),
    )


def _export_pdf(
    entity: ECGEntity,
    concat: ConcatenatedSequence,
    *,
    output_dir: Path,
    pagination_config: PaginationConfig,
    rr_histogram_config: RRHistogramConfig,
    sinus_extraction_config: SinusExtractionConfig,
    segment_colors: dict[str, str],
    segment_labels: dict[str, str],
) -> Path:
    LOGGER.info("entity_pdf_export_started", entity=entity)
    apply_default_style()
    output_path = output_dir / f"{entity.entity_id}.pdf"
    export_concatenated_pdf(
        entity,
        concat,
        output_path=output_path,
        pagination_config=pagination_config,
        rr_histogram_config=rr_histogram_config,
        sinus_extraction_config=sinus_extraction_config,
        segment_colors=segment_colors,
        segment_labels=segment_labels,
    )
    return output_path
