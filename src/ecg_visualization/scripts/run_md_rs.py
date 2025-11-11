from typing import Any, Final

from ecg_visualization.utils.utils import prepare_sequences
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

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
from ecg_visualization.models.md_rs.md_rs import MDRS

MD_RS_CONFIG: Final[dict[str, Any]] = {
    "N_x": 256,
    "input_scale": 0.5,
    "rho": 0.9,
    "leaking_rate": 0.3,
    "delta": 1e-3,
    "trans_length": 10,
    "N_x_tilde": 128,
    "seed": 0,
}


def run_md_rs() -> None:
    """
    Train MD-RS on each entity's 10-minute normal window, then score the full
    signal after scaling both sets with a StandardScaler fitted on the normal
    portion.
    """

    data_sources: list[ECG_Dataset] = [
        CUDB(),
        AFPDB(),
        MITDB(),
        AFDB(),
        LTAFDB(),
        SHDBAF(),
        SDDB(),
    ]

    for data_source in tqdm(data_sources):
        for entity in tqdm(data_source.data_entities):
            normal_window = entity.extract_normal_segment()

            train_sequence, test_sequence = prepare_sequences(
                normal_window, entity.signals
            )

            MD_RS_CONFIG.update({"N_u": train_sequence.shape[1]})

            model = MDRS(**MD_RS_CONFIG)
            model.train(train_sequence)

            model.reset_states()

            scores = model.predict(test_sequence)
