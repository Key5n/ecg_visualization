from typing import Any, Final

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
from ecg_visualization.utils.utils import prepare_sequences, sliding_window_sequences

WINDOW_SIZE: Final[int] = 10
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
    RR-interval series after scaling both sets with a StandardScaler fitted on
    the normal portion.
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
            rr_intervals = entity.compute_rr_intervals()

            if normal_window.size < WINDOW_SIZE or rr_intervals.size < WINDOW_SIZE:
                tqdm.write(
                    f"Skipping {entity.data_id}: insufficient RR intervals for windowing."
                )
                continue

            train_windows = sliding_window_sequences(normal_window, WINDOW_SIZE)
            test_windows = sliding_window_sequences(rr_intervals, WINDOW_SIZE)

            train_sequence, test_sequence = prepare_sequences(
                train_windows, test_windows
            )

            config = {**MD_RS_CONFIG, "N_u": train_sequence.shape[1]}

            model = MDRS(**config)
            model.train(train_sequence)

            model.reset_states()

            scores = model.predict(test_sequence)
