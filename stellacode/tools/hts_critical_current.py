import pandas as pd
from os.path import dirname, join
from typing import Optional
from stellacode import np


class HTSCriticalCurrent:
    """
    Computes the critical current for a number of HTS tapes
    """

    HTS_DATABASE_PATH = join(dirname(dirname(dirname(__file__))), "data", "hts_params_jeroen.xlsx")
    B_MIN = 0.1
    EPS = 1e-15

    def __init__(self, tapeLabel: str = "JeroenThesis", hts_database_path: Optional[str] = None, unit: str = "A/m²"):
        if hts_database_path is None:
            hts_database_path = self.HTS_DATABASE_PATH
        data = pd.read_excel(hts_database_path, index_col="label")
        self.par = data[tapeLabel]
        self.unit = unit

    @classmethod
    def available_tapes(cls) -> str:
        return pd.read_excel(cls.HTS_DATABASE_PATH, index_col="label").columns[1:]

    def __call__(self, *, b_field: np.ndarray, theta: np.ndarray, temperature: np.ndarray) -> np.ndarray:
        """
        This formula is from the paper:
        "Parameterization of the critical surface of REBCO conductors from Bruker."

        Args:
            * b_field: the self field is neglected (in Tesla), therefore the discontinuity
                of the tangent field is negligible.
            * theta: angle (from surface normal to magnetic field) in degrees
                theta=90 (b field parallel to the surface) gives a maximal critical current
            * temperature: in Kelvin

        Returns:
            * Critical current: in A/m² or A/m
        """
        # prevent zero field at which an anomalous behaviour occurs
        assert np.all(b_field >= self.B_MIN)

        # Normalized temperatures
        temperature_n = temperature / self.par.Tc0
        t_n0, t_n1 = np.maximum(1 - temperature_n**self.par.n0, self.EPS), np.maximum(
            1 - temperature_n**self.par.n1, self.EPS
        )

        # irreversibility field  # ab,c-plane:
        Bi_ab, Bi_c = self.par.Bi0_ab * (t_n1**self.par.n2 + self.par.a * t_n0), self.par.Bi0_c * t_n0
        b_ab, b_c = np.where(b_field < Bi_ab, b_field / Bi_ab, 1), np.where(b_field < Bi_c, b_field / Bi_c, 1)

        # critical current density:
        Jc_ab = (
            (self.par.alpha_ab / b_field)
            * b_ab**self.par.p_ab
            * (1 - b_ab) ** self.par.q_ab
            * (t_n1**self.par.n2 + self.par.a * t_n0) ** self.par.gamma_ab
        )  # ab-plane
        Jc_c = (
            (self.par.alpha_c / b_field) * b_c**self.par.p_c * (1 - b_c) ** self.par.q_c * t_n0**self.par.gamma_c
        )  # c-plane

        # temperature below Tcs and field below Bi
        Jc_ab = np.where((temperature_n < 1) & (b_ab < 1), Jc_ab, 0)
        Jc_c = np.where((temperature_n < 1) & (b_c < 1), Jc_c, 0)

        # anisotropic behavior
        g = self.par.g0 + self.par.g1 * np.exp(-self.par.g2 * b_field * np.exp(self.par.g3 * temperature))
        theta_symm = np.abs(theta + self.par.theta_peakOffset) * np.pi / 180
        theta_per = np.remainder(theta_symm, np.pi)
        theta = np.where(theta_per <= np.pi / 2, theta_per, np.pi - theta_per)

        # critical current
        crit_curr = np.minimum(Jc_c, Jc_ab) + np.maximum(0, Jc_ab - Jc_c) / (
            1 + ((np.pi / 2 - theta) / g) ** self.par.nu
        )
        if self.unit == "A/m²":
            return crit_curr
        elif self.unit == "A/m":
            return crit_curr * self.par.tSc  # tSc is the superconductor material width
