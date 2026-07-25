from __future__ import annotations

import logging

from ecg_visualization.logging.config import configure_root_logging
from ecg_visualization.tasks.config import save_config_text
from ecg_visualization.tasks.rhythm_event_sequences.config import (
    RhythmEventSequencesConfig,
)
from ecg_visualization.tasks.rhythm_event_sequences.utils import (
    SequenceSelectionSuccess,
    iter_sequence_selection_results,
)
from ecg_visualization.tasks.rhythm_event_sequences.visualize.export_concatenated_pdf import (
    export_concatenated_pdf,
)
from ecg_visualization.tasks.rhythm_event_sequences.visualize.plot_selection_summary import (
    plot_selection_summary,
)
from ecg_visualization.visualization.styles import apply_default_style

LOGGER = logging.getLogger(__name__)


def rhythm_event_sequence_visualize(config: RhythmEventSequencesConfig) -> None:
    configure_root_logging()
    apply_default_style()
    config.visualize_output_dir.mkdir(parents=True, exist_ok=True)
    save_config_text(config, config.config_path)
    pagination_config = config.pagination

    processed = 0
    results = list(iter_sequence_selection_results(config))
    for result in results:
        if not isinstance(result, SequenceSelectionSuccess):
            continue

        entity = result.entity
        concat = result.concat
        output_path = config.visualize_output_dir / f"{entity.entity_id}.pdf"
        export_concatenated_pdf(
            entity,
            concat,
            output_path=output_path,
            pagination_config=pagination_config,
            rr_histogram_config=config.rr_histogram,
            segment_colors=config.segment_colors,
            segment_labels=config.segment_labels,
        )
        LOGGER.info("Saved concatenated signal PDF to %s", output_path)
        processed += 1

    plot_selection_summary(results, output_path=config.summary_path)
    LOGGER.info("Saved sequence selection summary figure to %s", config.summary_path)

    LOGGER.info(
        "Finished ecg_visualization.visualization. processed=%d failed=%d output_dir=%s",
        processed,
        len(results) - processed,
        config.visualize_output_dir,
    )
