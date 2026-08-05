from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import structlog
from tqdm import tqdm

from ecg_visualization.datasets.physionet import load_data_sources
from ecg_visualization.logging.config import configure_root_logging
from ecg_visualization.tasks.all_anomaly_detection_scores.config import (
    AllAnomalyDetectionScoresConfig,
)
from ecg_visualization.tasks.anomaly_detection_example.run import (
    SCORE_METHOD_NAMES,
    build_example_data,
    plot_entity_scores,
)
from ecg_visualization.visualization.export import pdf_exporter
from ecg_visualization.visualization.styles import apply_default_style

LOGGER = structlog.get_logger(__name__)


def all_anomaly_detection_scores(config: AllAnomalyDetectionScoresConfig) -> None:
    """Write one ECG plus seven anomaly-score rows for every usable entity."""
    configure_root_logging()
    apply_default_style()
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    skipped = 0

    with pdf_exporter(str(config.output_path)) as exporter:
        for dataset_config in config.datasets:
            dataset = load_data_sources((dataset_config.dataset_id,))[0]
            for entity in tqdm(
                dataset.get_entities(),
                total=len(dataset.get_entity_ids()),
                desc=dataset_config.dataset_id.upper(),
            ):
                try:
                    example = build_example_data(
                        entity,
                        dataset_config.pre_ar_duration_sec,
                        config.model,
                    )
                    figure = plot_entity_scores(example, config.model)
                except (ValueError, np.linalg.LinAlgError) as exc:
                    LOGGER.warning(
                        "entity_skipped",
                        dataset_id=dataset_config.dataset_id,
                        entity_id=entity.entity_id,
                        reason=str(exc),
                    )
                    skipped += 1
                    continue
                exporter.add_page(figure)
                plt.close(figure)
                processed += 1

    LOGGER.info(
        "all_anomaly_detection_scores_saved",
        processed=processed,
        skipped=skipped,
        score_methods=SCORE_METHOD_NAMES,
        output_path=str(config.output_path),
    )
