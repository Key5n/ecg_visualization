from dataclasses import dataclass, field
from ecg_visualization.utils.utils import merge_overlapping_windows
import numpy as np
import numpy.typing as npt
import os

import wfdb
from wfdb.io import Annotation

dataset_root_dir = os.path.join("physionet.org", "files")

MIN_NORMAL_RR_INTERVAL_SEC = 0.6
MAX_NORMAL_RR_INTERVAL_SEC = 1.0
NORMAL_SEGMENT_DURATION_SEC = 10 * 60  # 10 minutes


@dataclass
class ECG_Entity:
    """
    Class representing a single ECG record/entity

    Attributes:
        data_id (str): Identifier for the ECG record
        dataset_name (str): Name of the dataset the record belongs to
        sr (int): Sampling rate of the ECG signal
        signals (npt.NDArray[np.float64]): ECG signal data
        annotation (Annotation): Annotation object containing metadata about the ECG record
        beats (npt.NDArray[np.int_]): Array of beat sample indices, each element divided by its sampling rate representing the times of beats in seconds
    """

    data_id: str
    dataset_name: str
    sr: int
    signals: npt.NDArray[np.float64]
    annotation: Annotation
    beats: npt.NDArray[np.int_]

    def __str__(self):
        return self.data_id

    def get_window_durations(
        self,
        window_size: int,
    ) -> list[tuple[int, int]]:
        """
        Build contiguous windows of RR intervals.

        Args:
            window_size (int): Number of RR intervals per window.

        Returns:
            list[tuple[int, int]]: Sample index windows for the requested size.
            Returns an empty list when insufficient beats are available.
        """

        if window_size < 2:
            raise ValueError("window_size must be at least 2 beats")

        if self.beats.size < window_size + 1:
            return []

        windows: list[tuple[int, int]] = []
        for start_idx in range(self.beats.size - window_size):
            start_sample = int(self.beats[start_idx])
            end_sample = int(self.beats[start_idx + window_size])
            windows.append((start_sample, end_sample))

        return windows

    def extract_normal_segment(self) -> tuple[int, int]:
        """
        Extract a 10-minute normal beat segment for this entity.

        Returns:
            tuple containing (
                start sample index (int),
                end sample index (int)
            ) where every RR interval stays within the normal range.

        Raises:
            ValueError: If the entity does not contain enough information to
            determine such a segment.
        """

        if self.beats.size < 2:
            raise ValueError(f"{self.data_id} does not contain enough beats to analyze")

        target_samples = int(self.sr * NORMAL_SEGMENT_DURATION_SEC)
        beat_times = self.beats / self.sr
        rr_intervals = np.diff(beat_times)

        start_idx = 0
        for interval_idx, rr_interval in enumerate(rr_intervals):
            if (
                rr_interval < MIN_NORMAL_RR_INTERVAL_SEC
                or rr_interval > MAX_NORMAL_RR_INTERVAL_SEC
            ):
                start_idx = interval_idx + 1
                continue

            current_duration = beat_times[interval_idx + 1] - beat_times[start_idx]
            if current_duration >= NORMAL_SEGMENT_DURATION_SEC:
                start_sample = int(self.beats[start_idx])
                end_sample = start_sample + target_samples
                if end_sample > self.signals.shape[0]:
                    raise ValueError(
                        f"{self.data_id} does not have sufficient samples for a 10-minute segment"
                    )

                return start_sample, end_sample

        raise ValueError(
            f"No 10-minute normal beat segment found for {self.data_id} ({self.dataset_name})"
        )

    def get_abnormal_windows(
        self,
        window_size: int,
        min_duration: float,
        max_duration: float,
    ) -> set[tuple[float, float]]:
        """
        Identify abnormal windows based on RR intervals.

        Args:
            window_size (int): Number of beats in each window.
            min_duration (float): Minimum duration for a normal window in seconds.
            max_duration (float): Maximum duration for a normal window in seconds.

        Returns:
            set[tuple[float, float]]: Set of tuples representing start and end times
            of abnormal windows.
        """

        windows = self.get_window_durations(window_size)
        if not windows:
            return set()

        abnormal_windows: set[tuple[float, float]] = set()
        for start_sample, end_sample in windows:
            start_time = start_sample / self.sr
            end_time = end_sample / self.sr
            duration = end_time - start_time
            if duration < min_duration or duration > max_duration:
                abnormal_windows.add((start_time, end_time))

        abnormal_windows = merge_overlapping_windows(abnormal_windows)
        return abnormal_windows

    def get_extreme_rr_windows(
        self,
        window_size: int,
        *,
        lower_percentile: float = 5.0,
        upper_percentile: float = 95.0,
    ) -> set[tuple[float, float]]:
        """
        Collect start/end times for 10-R-peak windows in the lowest or
        highest percentile range of durations across all such windows.
        """

        windows = self.get_window_durations(window_size)
        if not windows:
            return set()

        if not 0 <= lower_percentile < upper_percentile <= 100:
            raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100")

        durations_arr = np.array([(end - start) for start, end in windows], dtype=float)
        lower_bound = np.percentile(durations_arr, lower_percentile)
        upper_bound = np.percentile(durations_arr, upper_percentile)

        extreme_windows = set(
            (
                start_sample / self.sr,
                end_sample / self.sr,
            )
            for start_sample, end_sample in windows
            if (end_sample - start_sample) < lower_bound
            or (end_sample - start_sample) > upper_bound
        )
        return merge_overlapping_windows(extreme_windows)


@dataclass
class ECG_Dataset:
    """
    Base class for ECG Datasets

    Attributes:
        dir_path (str): Path to the dataset directory
        name (str): Name of the dataset
        dataset_id (str): Identifier for the dataset
        annotation_extention_priority (list[str]): List of annotation file extensions in order of priority
        beat_extention_priority (list[str]): List of beat annotation file extensions in order of priority
        data_ids (list[str]): List of data record identifiers
        data_entities (list[ECG_Entity]): List of ECG entities in the dataset
    """

    dir_path: str
    name: str
    dataset_id: str
    annotation_extention_priority: list[str] = field(
        default_factory=lambda: ["atr", "qrs", "ari"]
    )
    beat_extention_priority: list[str] = field(
        default_factory=lambda: ["atr", "qrs", "ari"]
    )
    data_ids: list[str] = field(init=False)
    data_entities: list[ECG_Entity] = field(default_factory=list)

    def __post_init__(self):
        record_path = os.path.join(self.dir_path, "RECORDS")
        with open(record_path, "r") as f:
            self.data_ids = f.read().splitlines()

        for data_id in self.data_ids:
            self.data_entities.append(self._load_entity(data_id))

    def _read_annotation(self, data_path: str) -> Annotation:
        for ext in self.annotation_extention_priority:
            annotation_file = f"{data_path}.{ext}"
            if os.path.isfile(annotation_file):
                annotation = wfdb.rdann(data_path, ext)
                return annotation

    def _read_normal_beats(self, data_path: str) -> npt.NDArray[np.int_]:
        for ext in self.beat_extention_priority:
            annotation_file = f"{data_path}.{ext}"
            if os.path.isfile(annotation_file):
                annotation = self._read_annotation(data_path)
                if ext == "atr":
                    beats = np.array(
                        [
                            sample
                            for sample, symbol in zip(
                                annotation.sample, annotation.symbol
                            )
                            if symbol == "N"
                        ],
                        dtype=np.int_,
                    )

                    return beats

                return np.asarray(annotation.sample, dtype=np.int_)

        raise FileNotFoundError(f"No annotation file found for {data_path}")

    def _load_entity(self, data_id: str) -> ECG_Entity:
        data_path = os.path.join(self.dir_path, data_id)
        signals, _ = wfdb.rdsamp(
            data_path,
            channels=[0],
        )
        squeezed = np.squeeze(signals)

        annotation = self._read_annotation(data_path)
        beats = self._read_normal_beats(data_path)

        record = wfdb.rdheader(data_path)
        sr = record.fs
        return ECG_Entity(data_id, self.name, sr, squeezed, annotation, beats)

    def extract_normal_segments(
        self,
    ) -> dict[str, tuple[int, int]]:
        """
        Extract 10-minute normal beat segments for all records in the dataset.

        Returns:
            dict[str, tuple[start_index, end_index]].
        """

        segments: dict[str, tuple[int, int]] = {}
        for entity in self.data_entities:
            segment = entity.extract_normal_segment()
            segments[entity.data_id] = segment

        return segments

    def __str__(self):
        return self.name


# https://physionet.org/content/cudb/1.0.0/
@dataclass
class CUDB(ECG_Dataset):
    dir_path: str = os.path.join(dataset_root_dir, "cudb", "1.0.0")
    name: str = "Tachyarrythmia"
    dataset_id: str = "cudb"
    sr: int = 250


# https://physionet.org/content/afpdb/1.0.0/
@dataclass
class AFPDB(ECG_Dataset):
    dir_path: str = os.path.join(dataset_root_dir, "afpdb", "1.0.0")
    name: str = "PAF Prediction Challenge Database"
    dataset_id: str = "afpdb"
    sr: int = 128


# https://physionet.org/content/mitdb/1.0.0/
@dataclass
class MITDB(ECG_Dataset):
    dir_path: str = os.path.join(dataset_root_dir, "mitdb", "1.0.0")
    name: str = "MIT-BIH Arrhythmia Database"
    dataset_id: str = "mitdb"
    sr: int = 360


# https://physionet.org/content/afdb/1.0.0/
@dataclass
class AFDB(ECG_Dataset):
    dir_path: str = os.path.join(dataset_root_dir, "afdb", "1.0.0")
    name: str = "MIT-BIH Atrial Fibrillation Database"
    dataset_id: str = "afdb"
    sr: int = 250

    def __post_init__(self):
        record_path = os.path.join(self.dir_path, "RECORDS")
        with open(record_path, "r") as f:
            self.data_ids = f.read().splitlines()

        self.data_ids = list(
            filter(lambda data_id: not data_id in ["00735", "03665"], self.data_ids)
        )

        for data_id in self.data_ids:
            self.data_entities.append(self._load_entity(data_id))


# https://physionet.org/content/ltafdb/1.0.0/
@dataclass
class LTAFDB(ECG_Dataset):
    dir_path: str = os.path.join(dataset_root_dir, "ltafdb", "1.0.0")
    name: str = "Long Term AF Database"
    dataset_id: str = "ltafdb"
    sr: int = 128


# https://physionet.org/content/shdb-af/1.0.1/
@dataclass
class SHDBAF(ECG_Dataset):
    dir_path: str = os.path.join(dataset_root_dir, "shdb-af", "1.0.1")
    name: str = "SHDB-AF: a Japanese Holter ECG database of atrial fibrillation"
    dataset_id: str = "shdb-af"
    sr: int = 200
    beat_extention_priority: list[str] = field(default_factory=lambda: ["qrs"])

    def __post_init__(self):
        record_path = os.path.join(self.dir_path, "RECORDS.txt")
        with open(record_path, "r") as f:
            self.data_ids = f.read().splitlines()

        for data_id in self.data_ids:
            try:
                self.data_entities.append(self._load_entity(data_id))
            except FileNotFoundError:
                continue


# https://physionet.org/content/sddb/1.0.0/
@dataclass
class SDDB(ECG_Dataset):
    dir_path: str = os.path.join(dataset_root_dir, "sddb", "1.0.0")
    name: str = "Sudden Cardiac Death Holter Database"
    dataset_id: str = "sddb"
    sr: int = 250
