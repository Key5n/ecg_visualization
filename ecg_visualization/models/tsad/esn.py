import time

import numpy as np
from sklearn.linear_model import Ridge

from .reservoir import EchoStateReservoir


def tsad_esn(
    rri_segments,
    n_reservoir=100,
    threshold_scale=10.0,
    seed=0,
    spectral_radius=0.95,
    input_scale=1.0,
    leak_rate=1.0,
    density=0.1,
):
    """Return ESN reconstruction-error anomaly scores for TS/RS/PA/AR."""
    if rri_segments.get("TS") is None:
        raise ValueError("TS segment is required.")

    scores = {"TS": None, "RS": None, "PA": None, "AR": None}
    reservoir = EchoStateReservoir(
        n_input=rri_segments["TS"].shape[1],
        n_reservoir=n_reservoir,
        spectral_radius=spectral_radius,
        input_scale=input_scale,
        leak_rate=leak_rate,
        density=density,
        seed=seed,
    )
    time_train_start = time.time()
    reservoir.reset()
    state_ts = reservoir.transform(rri_segments["TS"])
    readout = Ridge(alpha=1e-5, fit_intercept=False)
    readout.fit(state_ts, rri_segments["TS"])
    reconstruction_ts = readout.predict(state_ts)
    scores["TS"] = np.linalg.norm(reconstruction_ts - rri_segments["TS"], axis=1)
    threshold = np.max(scores["TS"]) * threshold_scale
    time_train = time.time() - time_train_start

    time_test_start = time.time()
    # The plotted segments are cut from disjoint portions of the source ECG.
    # Never carry reservoir state across one of those artificial boundaries.
    for segment in ("PA", "AR", "RS"):
        values = rri_segments.get(segment)
        if values is None:
            continue
        reservoir.reset()
        states = reservoir.transform(values)
        scores[segment] = np.linalg.norm(readout.predict(states) - values, axis=1)
    time_test = time.time() - time_test_start
    n_window_test = sum(
        len(rri_segments[name])
        for name in ("RS", "PA", "AR")
        if rri_segments.get(name) is not None
    )
    return {
        "scores": scores,
        "threshold": threshold,
        "time_train": time_train,
        "time_test": time_test,
        "n_window_test": n_window_test,
        "n_memory": reservoir.n_memory + readout.coef_.size,
        "n_trainable": readout.coef_.size,
    }
