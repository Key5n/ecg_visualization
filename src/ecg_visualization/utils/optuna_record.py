import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from optuna.artifacts import FileSystemArtifactStore, download_artifact
from optuna.trial import FrozenTrial

from ecg_visualization.utils.timed_sequence import TimedSequence


def _load_sequence_from_artifact(
    *,
    artifact_store: FileSystemArtifactStore,
    artifact_id: str | None,
    artifact_label: str,
) -> TimedSequence:
    if not artifact_id:
        raise ValueError(f"Missing artifact id for {artifact_label}.")

    with tempfile.TemporaryDirectory() as tmpdir:
        destination = Path(tmpdir) / f"{artifact_label}.npz"
        download_artifact(artifact_store, artifact_id, str(destination))
        payload = np.load(destination, allow_pickle=False)
        values = payload["values"]
        times = payload["times"]
        return TimedSequence(values=values, times=times)


@dataclass(slots=True)
class Record:
    """
    Lightweight mirror of the Optuna RDB trial payload plus ECG-specific attrs.
    """

    study_name: str
    trial_id: int
    trial_number: int
    state: optuna.trial.TrialState
    value: float | None
    params: dict[str, Any]
    user_attrs: dict[str, Any]
    system_attrs: dict[str, Any]
    datetime_start: datetime | None
    datetime_complete: datetime | None

    # Domain-specific shortcuts
    entity_id: str
    dataset_name: str
    score_sequence_artifact_id: str
    annotation_sequence_artifact_id: str
    beat_sequence_artifact_id: str
    signal_sequence_artifact_id: str

    @classmethod
    def from_trial(cls, trial: FrozenTrial, *, study_name: str) -> "Record":
        """
        Build a record from Optuna's FrozenTrial plus study metadata.
        """

        entity_id = trial.user_attrs.get("entity_id")
        dataset_name = trial.user_attrs.get("dataset_name")
        score_artifact_id = trial.user_attrs.get("score_sequence_artifact_id")
        annotation_artifact_id = trial.user_attrs.get("annotation_sequence_artifact_id")
        beat_artifact_id = trial.user_attrs.get("beat_sequence_artifact_id")
        signal_artifact_id = trial.user_attrs.get("signal_sequence_artifact_id")
        trial_id = getattr(trial, "_trial_id", trial.number)

        return cls(
            study_name=study_name,
            trial_id=trial_id,
            trial_number=trial.number,
            state=trial.state,
            value=trial.value,
            params=dict(trial.params),
            user_attrs=dict(trial.user_attrs),
            system_attrs=dict(trial.system_attrs),
            datetime_start=trial.datetime_start,
            datetime_complete=trial.datetime_complete,
            entity_id=entity_id,
            dataset_name=dataset_name,
            score_sequence_artifact_id=score_artifact_id,
            annotation_sequence_artifact_id=annotation_artifact_id,
            beat_sequence_artifact_id=beat_artifact_id,
            signal_sequence_artifact_id=signal_artifact_id,
        )


@dataclass(slots=True)
class VisualizationRecord:
    """
    Bundles all assets required to render a single Optuna trial.
    """

    record: Record
    score_sequence: TimedSequence
    annotation_sequence: TimedSequence
    beat_sequence: TimedSequence
    signal_sequence: TimedSequence

    @classmethod
    def from_trial(
        cls,
        trial: FrozenTrial,
        *,
        study_name: str,
        artifact_store: FileSystemArtifactStore,
    ) -> "VisualizationRecord":
        """
        Convenience constructor mirroring Record.from_trial then hydrating assets.
        """

        record = Record.from_trial(trial, study_name=study_name)
        return cls.from_record(record, artifact_store=artifact_store)

    @classmethod
    def from_record(
        cls,
        record: Record,
        *,
        artifact_store: FileSystemArtifactStore,
    ) -> "VisualizationRecord":
        """
        Hydrate visualization artifacts from Optuna storage.
        """

        score_sequence = _load_sequence_from_artifact(
            artifact_store=artifact_store,
            artifact_id=record.score_sequence_artifact_id,
            artifact_label="score_sequence",
        )
        annotation_sequence = _load_sequence_from_artifact(
            artifact_store=artifact_store,
            artifact_id=record.annotation_sequence_artifact_id,
            artifact_label="annotation_sequence",
        )
        beat_sequence = _load_sequence_from_artifact(
            artifact_store=artifact_store,
            artifact_id=record.beat_sequence_artifact_id,
            artifact_label="beat_sequence",
        )
        signal_sequence = _load_sequence_from_artifact(
            artifact_store=artifact_store,
            artifact_id=record.user_attrs.get("signal_sequence_artifact_id"),
            artifact_label="signal_sequence",
        )

        return cls(
            record=record,
            score_sequence=score_sequence,
            annotation_sequence=annotation_sequence,
            beat_sequence=beat_sequence,
            signal_sequence=signal_sequence,
        )
