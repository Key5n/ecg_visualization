import logging
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import optuna
from optuna.artifacts import FileSystemArtifactStore, upload_artifact
from tqdm import tqdm

from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.physionet import _load_data_sources
from ecg_visualization.logging.config import configure_root_logging
from ecg_visualization.logging.optuna import configure_optuna_logging
from ecg_visualization.models.md_rs.md_rs import MDRS, MDRSConfig
from ecg_visualization.tasks.study.config import StudyConfig
from ecg_visualization.utils.optuna_record import (
    build_storage_name,
    create_artifact_store,
    create_study_for_entity,
)
from ecg_visualization.utils.timed_sequence import TimedSequence
from ecg_visualization.utils.utils import prepare_sequences, sliding_window_sequences

LOGGER = logging.getLogger(__name__)


def study_all_entities(config: StudyConfig):
    configure_root_logging()
    configure_optuna_logging()

    data_sources = _load_data_sources(config.dataset_ids)
    artifact_store = create_artifact_store(config.artifact_root)
    storage_name = build_storage_name()
    model_config = config.model

    for data_source in tqdm(data_sources):
        for entity in tqdm(data_source.get_entities()):
            study = create_study_for_entity(entity=entity, storage_name=storage_name)
            study.optimize(
                Objective(
                    entity=entity,
                    artifact_store=artifact_store,
                    MD_RS_CONFIG=model_config,
                    WINDOW_SIZE=config.window_size,
                ),
                n_trials=config.n_trials,
            )


class Objective:
    def __init__(
        self,
        entity: ECGEntity,
        artifact_store: FileSystemArtifactStore,
        MD_RS_CONFIG: MDRSConfig,
        WINDOW_SIZE=10,
    ) -> None:
        self.entity = entity
        self._artifact_store = artifact_store
        self.MD_RS_CONFIG = MD_RS_CONFIG
        self.WINDOW_SIZE = WINDOW_SIZE

    def __call__(self, trial: optuna.Trial) -> float:

        input_scale = trial.suggest_float("input_scale", 0.1, 1.0)
        leaking_rate = trial.suggest_float("leaking_rate", 0.1, 0.9)
        rho = trial.suggest_float("rho", 0.5, 1.2)
        delta = trial.suggest_float("delta", 1e-5, 1e-2, log=True)

        try:
            normal_window = self.entity.extract_normal_segment()
        except ValueError:
            LOGGER.warning(
                f"Skipping {self.entity.entity_id}: no normal segment found."
            )
            return 0

        rr_intervals = self.entity.rr_intervals

        train_windows = sliding_window_sequences(normal_window.values, self.WINDOW_SIZE)
        test_windows = sliding_window_sequences(rr_intervals, self.WINDOW_SIZE)

        train_sequence, test_sequence = prepare_sequences(train_windows, test_windows)

        model_config = replace(
            self.MD_RS_CONFIG,
            input_scale=input_scale,
            leaking_rate=leaking_rate,
            rho=rho,
            delta=delta,
        )
        model = MDRS(model_config)
        model.train(train_sequence)

        model.reset_states()

        scores = model.predict(test_sequence)

        beat_times = self.entity.beats / self.entity.sr
        score_times = beat_times[self.WINDOW_SIZE :]
        score_sequence = TimedSequence(
            values=scores,
            times=score_times,
        )

        score_artifact = self._store_sequence_artifact(
            name="score_sequence",
            sequence=score_sequence,
            trial=trial,
        )

        trial.set_user_attr("score_sequence_artifact_id", score_artifact)
        trial.set_user_attr("entity_id", self.entity.entity_id)
        trial.set_user_attr("dataset_id", self.entity.dataset_id)
        trial.set_user_attr("normal_window_start_time", float(normal_window.start_time))
        trial.set_user_attr("normal_window_end_time", float(normal_window.end_time))

        return 0

    def _store_sequence_artifact(
        self,
        *,
        name: str,
        sequence: TimedSequence[np.float64],
        trial: optuna.Trial,
    ) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / f"{name}.npz"
            np.savez_compressed(
                tmp_path,
                values=sequence.values,
                times=sequence.times,
            )
            artifact_id = upload_artifact(
                artifact_store=self._artifact_store,
                file_path=str(tmp_path),
                study_or_trial=trial,
            )
        return artifact_id
