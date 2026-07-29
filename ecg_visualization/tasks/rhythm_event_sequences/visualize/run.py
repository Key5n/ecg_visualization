from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

import structlog

from ecg_visualization.datasets.physionet import load_data_sources
from ecg_visualization.logging.config import configure_root_logging
from ecg_visualization.tasks.config import save_config_text
from ecg_visualization.tasks.rhythm_event_sequences.config import (
    RhythmEventSequencesConfig,
)
from ecg_visualization.tasks.rhythm_event_sequences.utils import (
    SequenceSelectionResult,
    SequenceSelectionSuccess,
    select_sequence_result,
)
from ecg_visualization.tasks.rhythm_event_sequences.visualize.export_concatenated_pdf import (
    export_concatenated_pdf,
)
from ecg_visualization.tasks.rhythm_event_sequences.visualize.plot_selection_summary import (
    plot_selection_summary,
)
from ecg_visualization.visualization.styles import apply_default_style

LOGGER = structlog.get_logger(__name__)


def rhythm_event_sequence_visualize(config: RhythmEventSequencesConfig) -> None:
    configure_root_logging()
    apply_default_style()
    config.visualize_output_dir.mkdir(parents=True, exist_ok=True)
    save_config_text(config, config.config_path)
    results: list[SequenceSelectionResult] = []
    dataset = load_data_sources((config.dataset_id,))[0]

    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        futures = [
            executor.submit(
                _export_pdf,
                entity_id,
                config,
            )
            for entity_id in dataset.entity_ids
        ]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception:
                LOGGER.exception("Failed to export concatenated signal PDF")
                continue
            results.append(result)
            if not isinstance(result, SequenceSelectionSuccess):
                continue
            output_path = config.visualize_output_dir / f"{result.entity.entity_id}.pdf"
            LOGGER.info("concatenated_signal_pdf_saved", output_path=str(output_path))

    plot_selection_summary(results, output_path=config.summary_path)
    LOGGER.info(
        "sequence_selection_summary_saved", output_path=str(config.summary_path)
    )

    LOGGER.info(
        "visualization_finished",
        processed=sum(
            isinstance(result, SequenceSelectionSuccess) for result in results
        ),
        failed=len(dataset.entity_ids)
        - sum(isinstance(result, SequenceSelectionSuccess) for result in results),
        output_dir=str(config.visualize_output_dir),
    )


def _export_pdf(
    entity_id: str,
    config: RhythmEventSequencesConfig,
) -> SequenceSelectionResult:
    dataset = load_data_sources((config.dataset_id,))[0]
    entity = dataset.get_entity(entity_id=entity_id)
    result = select_sequence_result(
        entity,
        pre_ar_duration_sec=config.pre_ar_duration_sec,
        sinus_extraction_config=config.sinus_extraction,
    )
    if not isinstance(result, SequenceSelectionSuccess):
        return result

    LOGGER.info("entity_pdf_export_started", entity=entity)
    apply_default_style()
    output_path = config.visualize_output_dir / f"{entity.entity_id}.pdf"
    export_concatenated_pdf(
        entity,
        result.concat,
        output_path=output_path,
        pagination_config=config.pagination,
        rr_histogram_config=config.rr_histogram,
        sinus_extraction_config=config.sinus_extraction,
        segment_colors=config.segment_colors,
        segment_labels=config.segment_labels,
    )
    return result
