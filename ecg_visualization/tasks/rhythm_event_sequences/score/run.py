from __future__ import annotations

import structlog

from ecg_visualization.logging.config import configure_root_logging
from ecg_visualization.tasks.config import save_config_text
from ecg_visualization.tasks.rhythm_event_sequences.config import (
    RhythmEventSequencesConfig,
)
from ecg_visualization.tasks.rhythm_event_sequences.score.helpers import (
    score_concatenated_sequence,
)
from ecg_visualization.tasks.rhythm_event_sequences.score.plot_concat_scores import (
    export_concat_scores_pdf,
)
from ecg_visualization.tasks.rhythm_event_sequences.utils import (
    iter_concatenated_sequences,
)
from ecg_visualization.visualization.styles import apply_default_style

LOGGER = structlog.get_logger(__name__)


def rhythm_event_sequence_scores(config: RhythmEventSequencesConfig) -> None:
    configure_root_logging()
    apply_default_style()
    config.score_output_dir.mkdir(parents=True, exist_ok=True)
    save_config_text(config, config.config_path)

    processed = 0
    for entity, concat in iter_concatenated_sequences(
        config.dataset_id,
        pre_ar_duration_sec=config.pre_ar_duration_sec,
        sinus_extraction_config=config.sinus_extraction,
    ):
        try:
            score_result = score_concatenated_sequence(
                concat,
                window_size=config.window_size,
                model_config=config.model,
            )
        except ValueError as exc:
            LOGGER.warning(
                "entity_skipped", entity_id=entity.entity_id, reason=str(exc)
            )
            continue

        output_path = config.score_output_dir / f"{entity.entity_id}.pdf"
        export_concat_scores_pdf(
            entity,
            concat,
            score_result,
            output_path=str(output_path),
            config=config,
        )
        LOGGER.info("mdrs_scores_saved", output_path=str(output_path))
        processed += 1

    LOGGER.info(
        "scoring_finished",
        processed=processed,
        output_dir=str(config.score_output_dir),
    )
