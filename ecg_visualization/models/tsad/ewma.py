import time

import numpy as np


class OnlineEWMAAnomaly:
    def __init__(self, alpha=0.05, eps=1e-8):
        self.alpha = alpha
        self.eps = eps
        self.mu = None
        self.var = None
        self.frozen = False

    def _ready(self):
        return self.mu is not None and self.var is not None

    def freeze(self):
        self.frozen = True
        return self

    def score_one(self, x):
        if not self._ready():
            return np.nan
        x = np.asarray(x, dtype=float)
        return float(np.sum(((x - self.mu) ** 2) / (self.var + self.eps)))

    def update(self, x):
        if self.frozen:
            return self
        x = np.asarray(x, dtype=float)
        if self.mu is None:
            self.mu = x.copy()
            self.var = np.full_like(x, self.eps, dtype=float)
            return self
        mu_prev = self.mu.copy()
        self.mu = (1.0 - self.alpha) * self.mu + self.alpha * x
        self.var = (1.0 - self.alpha) * self.var + self.alpha * (x - mu_prev) ** 2
        self.var = np.maximum(self.var, self.eps)
        return self

    def fit(self, X):
        self.reset()
        scores = []
        for x in X:
            scores.append(self.score_one(x))
            self.update(x)
        self.freeze()
        return np.asarray(scores, dtype=float)

    def score(self, X):
        return np.asarray([self.score_one(x) for x in X], dtype=float)

    def reset(self):
        self.mu = None
        self.var = None
        self.frozen = False
        return self

    @property
    def n_memory(self):
        return self.mu.size + self.var.size

    @property
    def n_trainable(self):
        return self.mu.size + self.var.size


def tsad_ewma(rri_segments, alpha=0.05, threshold_scale=10.0):
    if rri_segments.get("TS") is None:
        raise ValueError("TS segment is required.")
    scores = {"TS": None, "RS": None, "PA": None, "AR": None}
    ewma = OnlineEWMAAnomaly(alpha=alpha)
    start = time.time()
    ewma.reset()
    scores["TS"] = ewma.fit(rri_segments["TS"])
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
        joined_scores = ewma.score(joined)
        scores["PA"] = joined_scores[:n_pa]
        if rri_segments.get("AR") is not None:
            scores["AR"] = joined_scores[n_pa:]
    if rri_segments.get("RS") is not None:
        scores["RS"] = ewma.score(rri_segments["RS"])
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
        "n_memory": ewma.n_memory,
        "n_trainable": ewma.n_trainable,
    }
