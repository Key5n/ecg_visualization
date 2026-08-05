import time

import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler


class HybridIncrementalPCA:
    def __init__(self, n_components=5, batch_size=32):
        self.n_components = int(n_components)
        self.batch_size = int(batch_size)
        self.reset()

    def reset(self):
        self.scaler = StandardScaler()
        self.pca = IncrementalPCA(n_components=self.n_components)
        self._init_buffer = []
        self._update_buffer = []
        self._fitted = False
        self.frozen = False
        return self

    def ready(self):
        return self._fitted

    def freeze(self):
        self.frozen = True
        return self

    def _initial_fit(self):
        X0 = np.vstack(self._init_buffer)
        if X0.shape[0] < self.n_components:
            return
        if np.all(np.var(X0, axis=0) < 1e-12):
            return
        self.scaler.fit(X0)
        X0z = self.scaler.transform(X0)
        self.pca.partial_fit(X0z)
        self._fitted = True
        self._init_buffer = []
        self._update_buffer = []

    def score_one(self, x):
        if not self.ready():
            return np.nan
        x = np.asarray(x, dtype=float).reshape(1, -1)
        xz = self.scaler.transform(x)
        z = self.pca.transform(xz)
        reconstruction = self.pca.inverse_transform(z)
        return float(np.mean((xz - reconstruction) ** 2))

    def update(self, x):
        if self.frozen:
            return self
        x = np.asarray(x, dtype=float).ravel()
        if not self._fitted:
            self._init_buffer.append(x)
            if len(self._init_buffer) >= self.n_components:
                self._initial_fit()
            return self
        self._update_buffer.append(x)
        if len(self._update_buffer) >= self.batch_size:
            X_batch = np.vstack(self._update_buffer)
            X_batch = self.scaler.transform(X_batch)
            self.pca.partial_fit(X_batch)
            self._update_buffer = []
        return self

    def step_train(self, x):
        score = self.score_one(x)
        self.update(x)
        return score

    def fit(self, X):
        self.reset()
        scores = [self.step_train(x) for x in X]
        self.freeze()
        return np.asarray(scores, dtype=float)

    def score(self, X):
        return np.asarray([self.score_one(x) for x in X], dtype=float)

    @property
    def n_memory(self):
        return (
            self.scaler.mean_.size
            + self.scaler.scale_.size
            + self.pca.mean_.size
            + self.pca.components_.size
        )

    @property
    def n_trainable(self):
        return self.pca.mean_.size + self.pca.components_.size


def tsad_ipca(rri_segments, n_components=5, batch_size=8, threshold_scale=10.0):
    """Incremental PCA-based anomaly detection."""
    if rri_segments.get("TS") is None:
        raise ValueError("TS segment is required.")
    scores = {"TS": None, "RS": None, "PA": None, "AR": None}
    ipca = HybridIncrementalPCA(
        n_components=n_components,
        batch_size=batch_size,
    )
    start = time.time()
    scores["TS"] = ipca.fit(rri_segments["TS"])
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
        joined_scores = ipca.score(joined)
        scores["PA"] = joined_scores[:n_pa]
        if rri_segments.get("AR") is not None:
            scores["AR"] = joined_scores[n_pa:]
    if rri_segments.get("RS") is not None:
        scores["RS"] = ipca.score(rri_segments["RS"])
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
        "n_memory": ipca.n_memory,
        "n_trainable": ipca.n_trainable,
    }
