import numpy as np
import networkx as nx


class Reservoir:
    def __init__(self, N_x, density, rho, activation_func, leaking_rate, seed=0):
        """
        param N_x: reservoir size
        param density: connection density
        param rho: spectral radius
        param activation_func: activation function
        param leaking_rate: leak rates
        param seed
        """
        self.seed = seed
        self.W = self.make_connection(N_x, density, rho)
        self.x = np.zeros(N_x)
        self.activation_func = activation_func
        self.alpha = leaking_rate

    def make_connection(self, N_x, density, rho):
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

    def __call__(self, x_in):
        """
        param x_in: x before update
        return: x after update
        """
        self.x = np.multiply(1.0 - self.alpha, self.x) + np.multiply(
            self.alpha, self.activation_func(np.dot(self.W, self.x) + x_in)
        )
        return self.x

    def reset_states(self) -> None:
        """Reset reservoir state vector to zeros."""
        self.x = np.zeros_like(self.x)
