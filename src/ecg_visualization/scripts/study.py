import tempfile
from pathlib import Path
from typing import Any, Final

import numpy as np

from ecg_visualization.datasets.dataset import ECG_Entity
from ecg_visualization.logging import configure_optuna_logging
from ecg_visualization.models.md_rs.md_rs import MDRS
from ecg_visualization.utils.timed_sequence import TimedSequence
from ecg_visualization.utils.utils import prepare_sequences, sliding_window_sequences
from ecg_visualization.utils.optuna_record import (
    build_storage_name,
    create_artifact_store,
    create_study_for_entity,
)
import optuna
from tqdm import tqdm
from optuna.artifacts import FileSystemArtifactStore
from optuna.artifacts import upload_artifact


from ecg_visualization.datasets.dataset import (
    AFDB,
    AFPDB,
    CUDB,
    LTAFDB,
    MITDB,
    SDDB,
    SHDBAF,
    ECG_Dataset,
)


DEFAULT_MD_RS_CONFIG: Final[dict[str, Any]] = {
    "N_x": 256,
    "input_scale": 0.5,
    "rho": 0.9,
    "leaking_rate": 0.3,
    "delta": 1e-3,
    "trans_length": 10,
    "N_x_tilde": 128,
    "seed": 0,
}


def study_all_entities():
    configure_optuna_logging()

    data_sources: list[ECG_Dataset] = [
        CUDB(),
        AFPDB(),
        MITDB(),
        AFDB(),
        LTAFDB(),
        SHDBAF(),
        SDDB(),
    ]
    artifact_root = Path("result") / "artifacts"
    artifact_store = create_artifact_store(artifact_root)
    storage_name = build_storage_name()

    for data_source in tqdm(data_sources):
        for entity in tqdm(data_source.data_entities):
            study = create_study_for_entity(entity=entity, storage_name=storage_name)
            study.optimize(
                Objective(
                    entity=entity,
                    artifact_store=artifact_store,
                    MD_RS_CONFIG=DEFAULT_MD_RS_CONFIG,
                ),
                n_trials=1,
            )


class Objective:
    def __init__(
        self,
        entity: ECG_Entity,
        artifact_store: FileSystemArtifactStore,
        MD_RS_CONFIG: dict[str, Any],
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
            tqdm.write(f"Skipping {self.entity.entity_id}: no normal segment found.")
            return 0

        rr_intervals = self.entity.compute_rr_intervals()

        train_windows = sliding_window_sequences(normal_window.values, self.WINDOW_SIZE)
        test_windows = sliding_window_sequences(rr_intervals, self.WINDOW_SIZE)

        train_sequence, test_sequence = prepare_sequences(train_windows, test_windows)

        tuned_config = {
            **self.MD_RS_CONFIG,
            "input_scale": input_scale,
            "leaking_rate": leaking_rate,
            "rho": rho,
            "delta": delta,
            "N_u": train_sequence.shape[1],
        }

        model = MDRS(**tuned_config)
        model.train(train_sequence)

        model.reset_states()

        scores = model.predict(test_sequence)

        beat_times = self.entity.beats / self.entity.sr
        score_times = beat_times[self.WINDOW_SIZE :]
        score_sequence = TimedSequence.from_time_axis(
            values=scores,
            time_axis=score_times,
        )

        score_artifact = self._store_sequence_artifact(
            name="score_sequence",
            sequence=score_sequence,
            trial=trial,
        )

        trial.set_user_attr("score_sequence_artifact_id", score_artifact)
        trial.set_user_attr("entity_id", self.entity.entity_id)
        trial.set_user_attr("dataset_id", self.entity.dataset_id)

        return 0

    def _store_sequence_artifact(
        self,
        *,
        name: str,
        sequence: TimedSequence,
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
