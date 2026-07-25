from __future__ import annotations

import logging

from ecg_visualization.logging.config import configure_root_logging
from ecg_visualization.tasks.config import save_config_text
from ecg_visualization.tasks.rhythm_event_sequences.config import (
    RhythmEventSequencesConfig,
)
from ecg_visualization.tasks.rhythm_event_sequences.utils import (
    iter_concatenated_sequences,
)
from ecg_visualization.tasks.rhythm_event_sequences.visualize.export_concatenated_pdf import (
    _export_concatenated_pdf,
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
    for entity, concat in iter_concatenated_sequences(config):
        output_path = config.visualize_output_dir / f"{entity.entity_id}.pdf"
        _export_concatenated_pdf(
            entity,
            concat,
            output_path=output_path,
            pagination_config=pagination_config,
            config=config,
        )
        LOGGER.info("Saved concatenated signal PDF to %s", output_path)
        processed += 1

    LOGGER.info(
        "Finished ecg_visualization.visualization. processed=%d output_dir=%s",
        processed,
        config.visualize_output_dir,
    )
