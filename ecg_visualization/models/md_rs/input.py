import numpy as np
from numpy.typing import NDArray

from .layer import Layer


class Input(Layer):
    """Input projection layer for reservoir state updates."""

    def __init__(
        self,
        N_u: int,
        N_x: int,
        input_scale: float,
        seed: int = 0,
    ) -> None:
        """Create a random input weight matrix.

        Args:
            N_u: Input dimension.
            N_x: Reservoir size.
            input_scale: Half-width of the uniform distribution used for weights.
            seed: Random seed for weight initialization.
        """
        # uniform distribution
        np.random.seed(seed=seed)
        self.Win = np.random.uniform(-input_scale, input_scale, (N_x, N_u))

    # weighted sum
    def forward(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project an input vector into reservoir space.

        Args:
            u: Input vector with shape ``(N_u,)``.

        Returns:
            Projected vector with shape ``(N_x,)``.
        """
        return np.dot(self.Win, u)
