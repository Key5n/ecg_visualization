import numpy as np


class EchoStateReservoir:
    def __init__(
        self,
        n_input,
        n_reservoir,
        spectral_radius=0.95,
        input_scale=1.0,
        leak_rate=1.0,
        density=0.1,
        seed=None,
    ):
        rng = np.random.default_rng(seed)
        self.n_input = n_input
        self.n_reservoir = n_reservoir
        self.leak_rate = leak_rate
        self.W_in = rng.uniform(
            -input_scale,
            input_scale,
            size=(n_reservoir, n_input),
        )
        W = rng.uniform(
            -1.0,
            1.0,
            size=(n_reservoir, n_reservoir),
        )
        mask = rng.random((n_reservoir, n_reservoir)) < density
        W *= mask
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        if radius > 0:
            W *= spectral_radius / radius
        self.W = W
        self.reset()

    def reset(self):
        self.state = np.zeros(self.n_reservoir, dtype=float)

    def step(self, u):
        pre_activation = self.W @ self.state + self.W_in @ u
        x = np.tanh(pre_activation)
        self.state = (1.0 - self.leak_rate) * self.state + self.leak_rate * x
        return self.state.copy()

    def transform(self, inputs):
        """
        Transform a sequence using the current reservoir state.

        This method does not reset the state. Call ``reset`` before processing
        a new independent sequence.
        """
        states = np.empty((len(inputs), self.n_reservoir), dtype=float)
        for i, u in enumerate(inputs):
            states[i] = self.step(u)
        return states

    @property
    def n_memory(self):
        return self.W_in.size + self.W.size

    @property
    def n_trainable(self):
        return 0
