from dataclasses import replace

import numpy as np
import numpy.typing as npt

from ..md_rs.md_rs import MDRS, MDRSConfig


class IncFedMDRS:
    def __init__(
        self,
        N_x: int,
        input_scale: float,
        rho: float,
        leaking_rate: float,
        delta: float,
        trans_length: int,
        P_g: npt.NDArray[np.float64] | None = None,
        N_x_tilde: int | None = None,
        threshold: float | None = None,
        density: float = 0.05,
        update: int = 1,
        lam: int = 1,
        seed: int = 0,
    ):
        self.seed = seed
        self.N_x = N_x
        self.input_scale = input_scale
        self.rho = rho
        self.leaking_rate = leaking_rate
        self.delta = delta
        self.trans_length = trans_length
        self.N_x_tilde = N_x_tilde
        self.P_g = P_g
        self.md_rs_config = MDRSConfig(
            N_x=N_x,
            input_scale=input_scale,
            rho=rho,
            leaking_rate=leaking_rate,
            delta=delta,
            trans_length=trans_length,
            N_x_tilde=N_x_tilde,
            threshold=threshold,
            density=density,
            update=update,
            lam=lam,
            seed=seed,
        )

    def train(self, U_list: list[npt.NDArray[np.float64]]) -> None:
        """
        U_list: input data list from multiple clients
        """

        phi_g = self.delta * np.identity(self.N_x_tilde)
        for U in U_list:
            model = MDRS(self.md_rs_config)
            phi_c = model.train(U)

            phi_g += phi_c

        P_g = np.linalg.inv(phi_g)

        self.P_g = P_g

        return P_g

    def evaluate(
        self,
        test_data: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        U: input data
        """

        model_config = replace(self.md_rs_config, precision_matrix=self.P_g)
        model = MDRS(model_config)

        scores = model.predict(test_data)
        scores = scores[self.trans_length :]

        return scores
