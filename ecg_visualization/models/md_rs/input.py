import numpy as np


class Input:
    def __init__(self, N_u, N_x, input_scale, seed=0):
        """
        param N_u: input dim
        param N_x: reservoir size
        param input_scale: input scaling
        """
        # uniform distribution
        np.random.seed(seed=seed)
        self.Win = np.random.uniform(-input_scale, input_scale, (N_x, N_u))

    # weighted sum
    def __call__(self, u):
        """
        param u: (N_u)-dim vector
        return: (N_x)-dim vector
        """
        return np.dot(self.Win, u)
