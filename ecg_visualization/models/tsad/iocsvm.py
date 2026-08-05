import time

import numpy as np
from sklearn.linear_model import SGDOneClassSVM
from sklearn.preprocessing import StandardScaler


class HybridIncrementalOCSVM:
    def __init__(self, nu=0.05, warmup_size=8, random_state=0):
        self.nu = nu
        self.warmup_size = int(warmup_size)
        self.random_state = random_state
        self.reset()

    def reset(self):
        self.scaler = StandardScaler()
        self.model = SGDOneClassSVM(nu=self.nu, random_state=self.random_state)
        self._warmup_buffer = []
        self._fitted = False
        self._frozen = False
        return self

    def ready(self):
        return self._fitted

    def freeze(self):
        self._frozen = True
        return self

    def _initialize_from_warmup(self):
        X0 = np.vstack(self._warmup_buffer)
        self.scaler.fit(X0)
        X0z = self.scaler.transform(X0)
        self.model.partial_fit(X0z)
        self._warmup_buffer = []
        self._fitted = True

    def score_one(self, x):
        if not self._fitted:
            return np.nan
        x = np.asarray(x, dtype=float).reshape(1, -1)
        xz = self.scaler.transform(x)
        return float(-self.model.decision_function(xz)[0])

    def update(self, x):
        if self._frozen:
            return self
        x = np.asarray(x, dtype=float).ravel()
        if not self._fitted:
            self._warmup_buffer.append(x)
            if len(self._warmup_buffer) >= self.warmup_size:
                self._initialize_from_warmup()
            return self
        xz = self.scaler.transform(x.reshape(1, -1))
        self.model.partial_fit(xz)
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
        n = self.scaler.mean_.size + self.scaler.scale_.size
        if self._fitted:
            n += self.model.coef_.size
            if hasattr(self.model, "offset_"):
                n += np.size(self.model.offset_)
            elif hasattr(self.model, "intercept_"):
                n += np.size(self.model.intercept_)
        return n

    @property
    def n_trainable(self):
        if not self._fitted:
            return 0
        n = self.model.coef_.size
        if hasattr(self.model, "offset_"):
            n += np.size(self.model.offset_)
        elif hasattr(self.model, "intercept_"):
            n += np.size(self.model.intercept_)
        return n


def tsad_iocsvm(
    rri_segments, nu=0.05, warmup_size=8, random_state=0, threshold_scale=10.0
):
    """Incremental One-Class SVM-based anomaly detection."""
    if rri_segments.get("TS") is None:
        raise ValueError("TS segment is required.")
    scores = {"TS": None, "RS": None, "PA": None, "AR": None}
    iocsvm = HybridIncrementalOCSVM(
        nu=nu, warmup_size=warmup_size, random_state=random_state
    )
    start = time.time()
    scores["TS"] = iocsvm.fit(rri_segments["TS"])
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
        joined_scores = iocsvm.score(joined)
        scores["PA"] = joined_scores[:n_pa]
        if rri_segments.get("AR") is not None:
            scores["AR"] = joined_scores[n_pa:]
    if rri_segments.get("RS") is not None:
        scores["RS"] = iocsvm.score(rri_segments["RS"])
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
        "n_memory": iocsvm.n_memory,
        "n_trainable": iocsvm.n_trainable,
    }
