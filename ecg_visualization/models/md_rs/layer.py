import numpy as np
from numpy.typing import NDArray


class Layer:
    """Base class for callable model layers."""

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the layer to an input vector."""
        return self.forward(x)

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the layer output for an input vector."""
        raise NotImplementedError
