from __future__ import annotations

import os

import numpy as np
import numpy.typing as npt
import wfdb
from wfdb.io import Annotation


def read_annotation(annotation_extention: str, data_path: str) -> Annotation:
    annotation_file = f"{data_path}.{annotation_extention}"
    if os.path.isfile(annotation_file):
        annotation = wfdb.rdann(data_path, annotation_extention)
        return annotation

    raise FileNotFoundError(f"No annotation file found for {data_path}")


def read_normal_beats(beat_extention: str, data_path: str) -> npt.NDArray[np.int_]:
    annotation_file = f"{data_path}.{beat_extention}"
    if os.path.isfile(annotation_file):
        annotation = wfdb.rdann(data_path, beat_extention)
        if beat_extention == "atr":
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
