import time

import numpy as np

from .mahalanobis import OnlineMahalanobis


def tsad_md(rri_segments, reg=1e-5, threshold_scale=10.0):
    """Return Mahalanobis-distance anomaly scores for TS/RS/PA/AR."""
    if rri_segments.get("TS") is None:
        raise ValueError("TS segment is required.")
    scores = {"TS": None, "RS": None, "PA": None, "AR": None}
    maha = OnlineMahalanobis(reg=reg)
    start = time.time()
    scores["TS"] = maha.fit(rri_segments["TS"])
    time_train = time.time() - start
    threshold = np.nanmax(scores["TS"]) * threshold_scale
    start = time.time()
    if rri_segments.get("PA") is not None:
        if rri_segments.get("AR") is None:
            joined = rri_segments["PA"]
            n_pa = len(joined)
        else:
            joined = np.vstack((rri_segments["PA"], rri_segments["AR"]))
            n_pa = len(rri_segments["PA"])
        joined_scores = maha.score(joined)
        scores["PA"] = joined_scores[:n_pa]
        if rri_segments.get("AR") is not None:
            scores["AR"] = joined_scores[n_pa:]
    if rri_segments.get("RS") is not None:
        scores["RS"] = maha.score(rri_segments["RS"])
    time_test = time.time() - start
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
        "n_memory": maha.n_memory,
        "n_trainable": maha.n_trainable,
    }
