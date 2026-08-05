import time

import numpy as np


def tsad_cv(rri_segments, threshold_scale=10.0):
    """Coefficient of Variation (CV)-based anomaly detection."""

    def cv(x):
        return np.std(x, ddof=1) / np.mean(x)

    scores = {"TS": None, "RS": None, "PA": None, "AR": None}
    if rri_segments.get("TS") is None:
        raise ValueError("TS segment is required.")

    start = time.time()
    scores["TS"] = np.array([cv(x) for x in rri_segments["TS"]])
    time_train = time.time() - start
    threshold = np.max(scores["TS"]) * threshold_scale

    start = time.time()
    if rri_segments.get("RS") is not None:
        scores["RS"] = np.array([cv(x) for x in rri_segments["RS"]])
    if rri_segments.get("PA") is not None:
        if rri_segments.get("AR") is None:
            scores["PA"] = np.array([cv(x) for x in rri_segments["PA"]])
        else:
            joined = np.vstack((rri_segments["PA"], rri_segments["AR"]))
            joined_scores = np.array([cv(x) for x in joined])
            n_pa = len(rri_segments["PA"])
            scores["PA"] = joined_scores[:n_pa]
            scores["AR"] = joined_scores[n_pa:]
    time_test = time.time() - start

    n_window_test = sum(
        len(rri_segments[name])
        for name in ("RS", "PA", "AR")
        if rri_segments.get(name) is not None
    )
    n_feature = rri_segments["TS"].shape[1]
    return {
        "scores": scores,
        "threshold": threshold,
        "time_train": time_train,
        "time_test": time_test,
        "n_window_test": n_window_test,
        "n_memory": 2 * n_feature,
        "n_trainable": 2 * n_feature,
    }
