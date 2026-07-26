from __future__ import annotations

import os

import numpy as np
import numpy.typing as npt
import wfdb
from wfdb.io import Annotation

from ecg_visualization.utils.signal_processing.rpeak_detection import detect_rpeaks

AnnotationExtention = str | tuple[str, ...]


def _iter_annotation_extentions(
    annotation_extention: AnnotationExtention,
) -> tuple[str, ...]:
    if isinstance(annotation_extention, str):
        return (annotation_extention,)

    return annotation_extention


def read_annotation(
    annotation_extention: AnnotationExtention,
    data_path: str,
) -> Annotation:
    attempted_files = []
    for extention in _iter_annotation_extentions(annotation_extention):
        annotation_file = f"{data_path}.{extention}"
        attempted_files.append(annotation_file)
        if os.path.isfile(annotation_file):
            annotation = wfdb.rdann(data_path, extention)
            return annotation

    raise FileNotFoundError(
        f"No annotation file found for {data_path}; tried {attempted_files}"
    )


def read_normal_beats(
    signals: npt.NDArray[np.float64],
    sampling_rate_hz: int | float,
) -> npt.NDArray[np.int_]:
    return detect_rpeaks(signals, sampling_rate_hz)


def read_annotated_normal_beats(
    beat_extention: AnnotationExtention,
    data_path: str,
) -> npt.NDArray[np.int_]:
    attempted_files = []
    for extention in _iter_annotation_extentions(beat_extention):
        annotation_file = f"{data_path}.{extention}"
        attempted_files.append(annotation_file)
        if os.path.isfile(annotation_file):
            annotation = wfdb.rdann(data_path, extention)
            if extention == "atr":
                return np.asarray(
                    [
                        sample
                        for sample, symbol in zip(annotation.sample, annotation.symbol)
                        if symbol == "N"
                    ],
                    dtype=np.int_,
                )

            return np.asarray(annotation.sample, dtype=np.int_)

    raise FileNotFoundError(
        f"No annotation file found for {data_path}; tried {attempted_files}"
    )
