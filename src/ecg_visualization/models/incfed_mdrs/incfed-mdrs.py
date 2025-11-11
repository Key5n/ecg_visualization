import numpy.typing as npt
import numpy as np
from ..md_rs.mdrs import MDRS


class IncFedMDRS:
    def __init__(
        self,
        N_u: int,
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
        self.N_u = N_u
        self.N_x = N_x
        self.input_scale = input_scale
        self.rho = rho
        self.leaking_rate = leaking_rate
        self.delta = delta
        self.trans_length = trans_length
        self.N_x_tilde = N_x_tilde
        self.P_g = P_g

    def train(self, U_list: list[npt.NDArray[np.float64]]) -> None:
        """
        U_list: input data list from multiple clients
        """

        phi_g = self.delta * np.identity(self.N_x_tilde)
        for U in U_list:
            model = MDRS(
                self.N_u,
                self.N_x,
                input_scale=self.input_scale,
                rho=self.rho,
                leaking_rate=self.leaking_rate,
                delta=self.delta,
                trans_length=self.trans_length,
                N_x_tilde=self.N_x_tilde,
                seed=self.seed,
            )
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

        model = MDRS(
            self.N_u,
            self.N_x,
            input_scale=self.input_scale,
            rho=self.rho,
            leaking_rate=self.leaking_rate,
            delta=self.delta,
            trans_length=self.trans_length,
            N_x_tilde=self.N_x_tilde,
            precision_matrix=self.P_g,
            seed=self.seed,
        )

        scores = model.predict(test_data)
        scores = scores[self.trans_length :]

        return scores
