from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .input import Input
from .reservoir import Reservoir

ActivationFunc = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True, slots=True, kw_only=True)
class MDRSConfig:
    N_x: int = 256
    input_scale: float = 1.0
    rho: float = 0.9
    leaking_rate: float = 0.9
    delta: float = 1e-3
    trans_length: int = 10
    precision_matrix: NDArray[np.float64] | None = None
    N_x_tilde: int | None = 256
    threshold: float | None = None
    density: float = 0.05
    activation_func: ActivationFunc = np.tanh
    noise_level: float | None = None
    update: int = 1
    lam: float = 1
    seed: int = 0


class MDRS:
    def __init__(
        self,
        config: MDRSConfig,
    ):
        self.seed = config.seed
        self.Input = None
        self.Reservoir = Reservoir(
            config.N_x,
            config.density,
            config.rho,
            config.activation_func,
            config.leaking_rate,
            seed=self.seed,
        )
        self.N_x = config.N_x
        self.input_scale = config.input_scale
        self.trans_length = config.trans_length
        self.threshold = None if config.threshold is None else config.threshold
        self.precision_matrix = None
        if config.noise_level is None:
            self.noise = None
        else:
            np.random.seed(seed=0)
            self.noise = np.random.uniform(
                -config.noise_level, config.noise_level, (self.N_x, 1)
            )
        self.delta = config.delta
        self.lam = config.lam
        self.update = config.update

        N_x_tilde = config.N_x if config.N_x_tilde is None else config.N_x_tilde

        self.N_x_tilde = N_x_tilde

        if config.precision_matrix is None:
            self.precision_matrix = (1.0 / self.delta) * np.eye(N_x_tilde, N_x_tilde)
        else:
            self.precision_matrix = config.precision_matrix

    def train(self, U):
        """
        U: input data
        """
        U = self._ensure_input(U)
        covariance_matrix = self.delta * np.eye(self.N_x_tilde)
        train_length = len(U)

        for n in range(train_length):
            x_in = self.Input(U[n])

            if self.noise is not None:
                x_in += self.noise

            x = self.Reservoir(x_in)

            if n > self.trans_length:
                x = x.reshape((-1, 1))
                x = self.subsample(x, self.N_x_tilde, self.seed)

                covariance_matrix += np.dot(x, x.T)

                # disable comment out below when you perform online learning
                # self.precision_matrix = self.calc_next_precision_matrix(
                #     x, self.precision_matrix
                # )
                #
                # mahalanobis_distance = np.dot(np.dot(x.T, self.precision_matrix), x)
                # self.threshold = (
                #     max(mahalanobis_distance, self.threshold)
                #     if self.threshold is not None
                #     else mahalanobis_distance
                # )

        self.precision_matrix = np.linalg.inv(covariance_matrix)
        return covariance_matrix

    def predict(self, U, threshold=None):
        """
        U: input data
        """
        U = self._ensure_input(U)
        data_length = len(U)
        mahalanobis_distances = []

        if threshold is not None:
            self.threshold = threshold

        for n in range(data_length):
            x_in = self.Input(U[n])

            x = self.Reservoir(x_in)
            x = x.reshape((-1, 1))
            x = self.subsample(x, self.N_x_tilde, self.seed)

            mahalanobis_distance = np.dot(np.dot(x.T, self.precision_matrix), x)
            mahalanobis_distance = np.squeeze(mahalanobis_distance)
            mahalanobis_distances.append(mahalanobis_distance)

        return np.array(mahalanobis_distances, dtype=np.float64)

    def calc_next_precision_matrix(self, x, precision_matrix):
        x = np.reshape(x, (-1, 1))
        next_precision_matrix = precision_matrix
        for _ in np.arange(self.update):
            gain = 1 / self.lam * np.dot(next_precision_matrix, x)
            gain = gain / (
                1 + 1 / self.lam * np.dot(np.dot(x.T, next_precision_matrix), x)
            )
            next_precision_matrix = (
                1
                / self.lam
                * (
                    next_precision_matrix
                    - np.dot(np.dot(gain, x.T), next_precision_matrix)
                )
            )
        return next_precision_matrix

    def reset_states(self) -> None:
        """Reset reservoir dynamics to forget any previous sequence."""
        self.Reservoir.reset_states()

    def _ensure_input(self, U: NDArray[np.float64]) -> NDArray[np.float64]:
        U = np.asarray(U, dtype=np.float64).reshape(len(U), -1)
        input_size = U.shape[1]
        if self.Input is None:
            self.Input = Input(input_size, self.N_x, self.input_scale, seed=self.seed)
            return U

        expected_size = self.Input.Win.shape[1]
        if expected_size != input_size:
            raise ValueError(
                f"Expected input dimension {expected_size}, got {input_size}."
            )
        return U

    def subsample(
        self,
        x: NDArray[np.float64],
        subsampling_size: int | None = None,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        """Randomly subsample the reservoir state for dimensionality reduction."""
        rng = np.random.default_rng(self.seed if seed is None else seed)
        size = self.N_x_tilde if subsampling_size is None else subsampling_size
        return rng.choice(x, size, replace=False)
