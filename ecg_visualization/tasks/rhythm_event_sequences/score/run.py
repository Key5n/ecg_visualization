from __future__ import annotations

import logging

import matplotlib.pyplot as plt

from ecg_visualization.logging.config import configure_root_logging
from ecg_visualization.tasks.config import save_config_text
from ecg_visualization.tasks.rhythm_event_sequences.config import (
    RhythmEventSequencesConfig,
)
from ecg_visualization.tasks.rhythm_event_sequences.score.helpers import (
    score_concatenated_sequence,
)
from ecg_visualization.tasks.rhythm_event_sequences.score.plot_concat_scores import (
    _plot_concat_scores,
)
from ecg_visualization.tasks.rhythm_event_sequences.utils import (
    iter_concatenated_sequences,
)
from ecg_visualization.visualization.styles import apply_default_style

LOGGER = logging.getLogger(__name__)


def rhythm_event_sequence_scores(config: RhythmEventSequencesConfig) -> None:
    configure_root_logging()
    apply_default_style()
    config.score_output_dir.mkdir(parents=True, exist_ok=True)
    save_config_text(config, config.config_path)

    processed = 0
    for entity, concat in iter_concatenated_sequences(config):
        try:
            score_result = score_concatenated_sequence(
                concat,
                window_size=config.window_size,
                model_config=config.model,
            )
        except ValueError as exc:
            LOGGER.warning("Skipping %s: %s", entity.entity_id, exc)
            continue

        fig = _plot_concat_scores(entity, concat, score_result, config=config)
        output_path = config.score_output_dir / f"{entity.entity_id}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        LOGGER.info("Saved MD-RS scores to %s", output_path)
        processed += 1

    LOGGER.info(
        "Finished scoring. processed=%d output_dir=%s",
        processed,
        config.score_output_dir,
    )
