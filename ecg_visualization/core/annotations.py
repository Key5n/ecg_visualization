from __future__ import annotations

import os

import numpy as np
import numpy.typing as npt
import wfdb
from wfdb.io import Annotation


def read_annotation(
    annotation_extention_priority: tuple[str, ...], data_path: str
) -> Annotation:
    for ext in annotation_extention_priority:
        annotation_file = f"{data_path}.{ext}"
        if os.path.isfile(annotation_file):
            annotation = wfdb.rdann(data_path, ext)
            return annotation

    raise FileNotFoundError(f"No annotation file found for {data_path}")


def read_normal_beats(
    beat_extention_priority: tuple[str, ...], data_path: str
) -> npt.NDArray[np.int_]:
    for ext in beat_extention_priority:
        annotation_file = f"{data_path}.{ext}"
        if os.path.isfile(annotation_file):
            annotation = wfdb.rdann(data_path, ext)
            if ext == "atr":
                beats = np.array(
                    [
                        sample
                        for sample, symbol in zip(annotation.sample, annotation.symbol)
                        if symbol == "N"
                    ],
                    dtype=np.int_,
                )

                return beats

            return np.asarray(annotation.sample, dtype=np.int_)

    raise FileNotFoundError(f"No annotation file found for {data_path}")
