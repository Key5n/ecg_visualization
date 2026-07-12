from collections.abc import Callable

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from .layer import Layer

ActivationFunc = Callable[[NDArray[np.float64]], NDArray[np.float64]]


class Reservoir(Layer):
    """Recurrent reservoir layer with leaky state updates."""

    def __init__(
        self,
        N_x: int,
        density: float,
        rho: float,
        activation_func: ActivationFunc,
        leaking_rate: float,
        seed: int = 0,
    ) -> None:
        """Initialize reservoir weights and state.

        Args:
            N_x: Reservoir size.
            density: Connection density for the random graph.
            rho: Target spectral radius for recurrent weights.
            activation_func: Activation function applied to the state update.
            leaking_rate: Leaky integration rate.
            seed: Random seed for graph and weight initialization.
        """
        self.seed = seed
        self.W = self.make_connection(N_x, density, rho)
        self.x = np.zeros(N_x)
        self.activation_func = activation_func
        self.alpha = leaking_rate

    def make_connection(
        self,
        N_x: int,
        density: float,
        rho: float,
    ) -> NDArray[np.float64]:
        """Create a scaled recurrent weight matrix.

        Args:
            N_x: Reservoir size.
            density: Connection density for the random graph.
            rho: Target spectral radius.

        Returns:
            Recurrent weight matrix with shape ``(N_x, N_x)``.
        """
        # Erdos-Renyi random graph
        m = int(N_x * (N_x - 1) * density / 2)
        G = nx.gnm_random_graph(N_x, m, self.seed)
        connection = nx.to_numpy_array(G)
        W = np.array(connection)

        rec_scale = 1.0
        np.random.seed(seed=self.seed)
        W *= np.random.uniform(-rec_scale, rec_scale, (N_x, N_x))

        # rescaling
        eigv_list = np.linalg.eig(W)[0]
        sp_radius = np.max(np.abs(eigv_list))
        W *= rho / sp_radius

        return W

    def forward(self, x_in: NDArray[np.float64]) -> NDArray[np.float64]:
        """Update and return the reservoir state.

        Args:
            x_in: Input vector added to the recurrent state update.

        Returns:
            Updated reservoir state vector.
        """
        self.x = np.multiply(1.0 - self.alpha, self.x) + np.multiply(
            self.alpha, self.activation_func(np.dot(self.W, self.x) + x_in)
        )
        return self.x

    def reset_states(self) -> None:
        """Reset reservoir state vector to zeros."""
        self.x = np.zeros_like(self.x)
