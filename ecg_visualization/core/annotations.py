from __future__ import annotations

import os
from functools import singledispatch
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import wfdb
from wfdb.io import Annotation

if TYPE_CHECKING:
    from ecg_visualization.core.dataset import ECGDataset


@singledispatch
def read_annotation(dataset_cls, data_path: str) -> Annotation:
    raise TypeError(f"Unsupported dataset type: {type(dataset_cls)!r}")


@read_annotation.register(type)
def _(dataset_cls: type[ECGDataset], data_path: str) -> Annotation:
    for ext in dataset_cls.annotation_extention_priority:
        annotation_file = f"{data_path}.{ext}"
        if os.path.isfile(annotation_file):
            annotation = wfdb.rdann(data_path, ext)
            return annotation

    raise FileNotFoundError(f"No annotation file found for {data_path}")


@singledispatch
def read_normal_beats(
    dataset_cls: type[ECGDataset], data_path: str
) -> npt.NDArray[np.int_]:
    raise TypeError(f"Unsupported dataset type: {type(dataset_cls)!r}")


@read_normal_beats.register(type)
def _(dataset_cls: type[ECGDataset], data_path: str) -> npt.NDArray[np.int_]:
    for ext in dataset_cls.beat_extention_priority:
        annotation_file = f"{data_path}.{ext}"
        if os.path.isfile(annotation_file):
            annotation = read_annotation(dataset_cls, data_path)
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
