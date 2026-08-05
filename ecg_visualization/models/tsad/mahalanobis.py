import numpy as np


class OnlineMahalanobis:
    """
    Online Gaussian estimation using Welford's algorithm.
    Mahalanobis distance is used as the anomaly score.
    """

    def __init__(self, reg=1e-6):
        self.reg = reg
        self.reset()

    def reset(self):
        self.n = 0
        self.mean = None
        self.M2 = None
        self.cov = None
        self.precision = None
        self.frozen = False
        return self

    def ready(self):
        return self.n >= 2

    def freeze(self):
        if self.ready():
            self.cov = self.covariance()
            self.precision = np.linalg.pinv(self.cov)
        self.frozen = True
        return self

    def covariance(self):
        d = self.mean.size
        cov = self.M2 / (self.n - 1)
        trace = np.trace(cov)
        cov += (self.reg * (trace / d + 1.0)) * np.eye(d)
        return cov

    def score_one(self, x):
        if not self.ready():
            return np.nan
        x = np.asarray(x, dtype=float)
        diff = x - self.mean
        return float(diff @ self.precision @ diff)

    def update(self, x):
        if self.frozen:
            return self
        x = np.asarray(x, dtype=float)
        if self.n == 0:
            self.mean = x.copy()
            self.M2 = np.zeros((x.size, x.size), dtype=float)
            self.n = 1
            return self
        self.n += 1
        delta_old = x - self.mean
        self.mean += delta_old / self.n
        delta_new = x - self.mean
        self.M2 += np.outer(delta_old, delta_new)
        return self

    def step_train(self, x):
        score = self.score_one(x)
        self.update(x)
        return score

    def fit(self, X):
        self.reset()
        for x in X:
            self.update(x)
        self.freeze()
        return self.score(X)

    def score(self, X):
        scores = []
        for x in X:
            scores.append(self.score_one(x))
        return np.asarray(scores, dtype=float)

    @property
    def n_memory(self):
        return self.mean.size + self.precision.size

    @property
    def n_trainable(self):
        return self.mean.size + self.precision.size
