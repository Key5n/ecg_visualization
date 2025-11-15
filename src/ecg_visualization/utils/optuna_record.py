import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

import numpy as np
import optuna
from optuna.artifacts import FileSystemArtifactStore, download_artifact
from optuna.exceptions import OptunaError
from optuna.trial import FrozenTrial
from tqdm import tqdm

from ecg_visualization.utils.timed_sequence import TimedSequence

if TYPE_CHECKING:
    from ecg_visualization.datasets.dataset import ECG_Entity


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
        download_artifact(
            artifact_store=artifact_store,
            artifact_id=artifact_id,
            file_path=str(destination),
        )
        payload = np.load(destination, allow_pickle=False)
        values = payload["values"]
        times = payload["times"]
        return TimedSequence(values=values, times=times)


def create_artifact_store(base_dir: str | Path) -> FileSystemArtifactStore:
    """Build a FileSystemArtifactStore rooted at base_dir, ensuring the directory exists."""

    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    return FileSystemArtifactStore(base_path=str(base_path))


def build_storage_name(
    *,
    driver_env: str = "OPTUNA_DB_DRIVER",
    user_env: str = "OPTUNA_DB_USER",
    password_env: str = "OPTUNA_DB_PASSWORD",
    host_env: str = "OPTUNA_DB_HOST",
    port_env: str = "OPTUNA_DB_PORT",
    database_env: str = "OPTUNA_DB_NAME",
    default_driver: str = "mysql+pymysql",
    default_user: str = "root",
    default_password: str = "foo",
    default_host: str = "localhost",
    default_port: str = "3306",
    default_database: str = "optuna",
) -> str:
    """Construct an Optuna storage URL using environment variables with defaults."""

    driver = os.getenv(driver_env, default_driver)
    user = os.getenv(user_env, default_user)
    password = os.getenv(password_env, default_password)
    host = os.getenv(host_env, default_host)
    port = os.getenv(port_env, default_port)
    database = os.getenv(database_env, default_database)
    return f"{driver}://{user}:{password}@{host}:{port}/{database}"


def load_study_for_entity(
    entity: "ECG_Entity",
    *,
    storage_name: str,
    log_fn: Callable[[str], None] | None = None,
) -> optuna.Study | None:
    """Load the Optuna study for an entity with shared logging behavior."""

    study_name = f"{entity.dataset_name} {entity.data_id}"
    log = log_fn or tqdm.write
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
    except OptunaError as exc:
        log(f"Skipping {study_name}: failed to load study ({exc})")
        return None

    if not study.trials:
        log(f"Skipping {study_name}: no trials available.")
        return None

    return study


def create_study_for_entity(
    entity: "ECG_Entity",
    *,
    storage_name: str,
    **kwargs: Any,
) -> optuna.Study:
    """Create (or reuse) an Optuna study for a specific entity."""

    study_name = f"{entity.dataset_name} {entity.data_id}"
    kwargs.setdefault("load_if_exists", True)
    return optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        **kwargs,
    )


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

    @classmethod
    def from_trial(cls, trial: FrozenTrial, *, study_name: str) -> "Record":
        """
        Build a record from Optuna's FrozenTrial plus study metadata.
        """

        entity_id = trial.user_attrs.get("entity_id")
        dataset_name = trial.user_attrs.get("dataset_name")
        score_artifact_id = trial.user_attrs.get("score_sequence_artifact_id")
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
        )


@dataclass(slots=True)
class VisualizationRecord:
    """
    Bundles all assets required to render a single Optuna trial.
    """

    record: Record
    score_sequence: TimedSequence

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
        return cls(
            record=record,
            score_sequence=score_sequence,
        )
