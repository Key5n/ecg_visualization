import logging
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import optuna
from optuna.artifacts import FileSystemArtifactStore, download_artifact
from optuna.exceptions import OptunaError
from optuna.storages import RDBStorage
from optuna.trial import FrozenTrial

from ecg_visualization.config.settings import (
    MYSQL_DATABASE,
    MYSQL_ROOT_PASSWORD,
    OPTUNA_DB_DRIVER,
    OPTUNA_DB_HOST,
    OPTUNA_DB_PORT,
    OPTUNA_DB_USER,
)
from ecg_visualization.utils.timed_sequence import TimedSequence

if TYPE_CHECKING:
    from ecg_visualization.core.entity import ECGEntity

LOGGER = logging.getLogger(__name__)


def _load_sequence_from_artifact(
    *,
    artifact_store: FileSystemArtifactStore,
    artifact_id: str | None,
    artifact_label: str,
) -> TimedSequence[np.float64]:
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


def build_storage_name() -> str:
    """Construct an Optuna storage URL from configured settings."""

    return f"{OPTUNA_DB_DRIVER}://{OPTUNA_DB_USER}:{MYSQL_ROOT_PASSWORD}@{OPTUNA_DB_HOST}:{OPTUNA_DB_PORT}/{MYSQL_DATABASE}"


def get_study_identifiers(study: optuna.Study) -> tuple[str, str]:
    dataset_id = study.user_attrs.get("dataset_id")
    entity_id = study.user_attrs.get("entity_id")
    if dataset_id is None or entity_id is None:
        raise ValueError(
            f"Study '{study.study_name}' is missing dataset_id/entity_id user attrs"
        )
    return str(dataset_id), str(entity_id)


class StudyLoader:
    """Singleton per storage URL to reuse Optuna RDB storage connections."""

    _instances: ClassVar[dict[str, "StudyLoader"]] = {}
    _lock: ClassVar[Lock] = Lock()

    def __new__(cls, storage_name: str):
        with cls._lock:
            instance = cls._instances.get(storage_name)
            if instance is None:
                instance = super().__new__(cls)
                instance._initialize(storage_name)
                cls._instances[storage_name] = instance
            return instance

    def _initialize(self, storage_name: str) -> None:
        self._storage_name = storage_name
        self._storage = RDBStorage(
            storage_name,
        )

    def load(
        self,
        entity: "ECGEntity",
    ) -> optuna.Study | None:
        """Load the Optuna study for the provided entity."""

        study_name = f"{entity.dataset_id} {entity.entity_id}"
        return self.load_by_name(study_name)

    def load_by_name(
        self,
        study_name: str,
    ) -> optuna.Study | None:
        """Load a study by name, logging errors and filtering empty trials."""

        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=self._storage,
            )
        except OptunaError as exc:
            LOGGER.warning(f"Skipping {study_name}: failed to load study ({exc})")
            return None

        if not study.trials:
            LOGGER.info(f"Skipping {study_name}: no trials available.")
            return None

        return study


def load_studies(storage_name: str) -> list[optuna.Study]:
    """Load every study in the storage, skipping ones that fail or lack trials."""

    study_names = optuna.study.get_all_study_names(storage_name)
    loader = StudyLoader(storage_name)
    studies: list[optuna.Study] = []
    for study_name in study_names:
        study = loader.load_by_name(study_name)
        if study is None:
            continue
        studies.append(study)
    return studies


def create_study_for_entity(
    entity: "ECGEntity",
    *,
    storage_name: str,
    **kwargs: Any,
) -> optuna.Study:
    """Create (or reuse) an Optuna study for a specific entity."""

    study_name = f"{entity.dataset_id} {entity.entity_id}"
    kwargs.setdefault("load_if_exists", True)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        **kwargs,
    )
    if study.user_attrs.get("dataset_id") != entity.dataset_id:
        study.set_user_attr("dataset_id", entity.dataset_id)
    if study.user_attrs.get("entity_id") != entity.entity_id:
        study.set_user_attr("entity_id", entity.entity_id)
    return study


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
    score_sequence: TimedSequence[np.float64]

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
