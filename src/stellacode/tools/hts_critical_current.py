from os.path import dirname, join
from typing import Optional

import pandas as pd

from stellacode import np


class HTSCriticalCurrent:
    """
    Computes the critical current for a number of HTS tapes
    """

    HTS_DATABASE_PATH = join(
        dirname(dirname(dirname(dirname(__file__)))), "data", "hts_params_jeroen.xlsx")
    B_MIN = 0.1
    EPS = 1e-15

    def __init__(self, tapeLabel: str = "JeroenThesis",
                 hts_database_path: Optional[str] = None,
                 unit: str = "A/m²"):
        """
        Initialize the HTSCriticalCurrent class.

        Args:
            tapeLabel (str): Label of the HTS tape.
            hts_database_path (str, optional): Path to the HTS database.
            unit (str, optional): Unit of the critical current.

        Raises:
            ValueError: If the given tapeLabel is not in the database.
        """
        # Set the default path to the HTS database
        if hts_database_path is None:
            hts_database_path = self.HTS_DATABASE_PATH

        # Read the HTS parameters from the database
        data = pd.read_excel(hts_database_path, index_col="label")

        # Check if the given tapeLabel is in the database
        if tapeLabel not in data.index:
            raise ValueError(f"Invalid tapeLabel: {tapeLabel}")

        # Set the parameters and unit of the HTS tape
        self.par = data.loc[tapeLabel]
        self.unit = unit

    @classmethod
    def available_tapes(cls) -> str:
        """
        Get the list of available HTS tapes.

        Returns:
            str: A string containing the labels of the available HTS tapes,
                 separated by commas.
        """
        # Read the HTS parameters from the database
        data = pd.read_excel(cls.HTS_DATABASE_PATH, index_col="label")

        # Get the list of available tapes
        tapes = data.columns[1:]

        # Return the list of tapes as a string
        return ", ".join(tapes)

    def __call__(self, *, b_field: np.ndarray, theta: np.ndarray, temperature: np.ndarray) -> np.ndarray:
        """
        Compute the critical current density.

        Args:
            * b_field: the self field is neglected (in Tesla), therefore the discontinuity
                of the tangent field is negligible.
            * theta: angle (from surface normal to magnetic field) in degrees
                theta=90 (b field parallel to the surface) gives a maximal critical current
            * temperature: in Kelvin

        Returns:
            * Critical current density: in A/m² or A/m
        """
        # Prevent zero field at which an anomalous behavior occurs
        assert np.all(b_field >= self.B_MIN)

        # Normalize temperatures
        temperature_n = temperature / self.par.Tc0
        t_n0, t_n1 = np.maximum(1 - temperature_n**self.par.n0, self.EPS), np.maximum(
            1 - temperature_n**self.par.n1, self.EPS
        )

        # Compute irreversibility field
        Bi_ab, Bi_c = self.par.Bi0_ab * \
            (t_n1**self.par.n2 + self.par.a * t_n0), self.par.Bi0_c * t_n0
        b_ab, b_c = np.where(b_field < Bi_ab, b_field / Bi_ab,
                             1), np.where(b_field < Bi_c, b_field / Bi_c, 1)

        # Compute critical current density
        Jc_ab = (
            (self.par.alpha_ab / b_field)
            * b_ab**self.par.p_ab
            * (1 - b_ab) ** self.par.q_ab
            * (t_n1**self.par.n2 + self.par.a * t_n0) ** self.par.gamma_ab
        )  # ab-plane
        Jc_c = (
            (self.par.alpha_c / b_field) * b_c**self.par.p_c *
            (1 - b_c) ** self.par.q_c * t_n0**self.par.gamma_c
        )  # c-plane

        # Set critical current density to zero below Tcs and below Bi
        Jc_ab = np.where((temperature_n < 1) & (b_ab < 1), Jc_ab, 0)
        Jc_c = np.where((temperature_n < 1) & (b_c < 1), Jc_c, 0)

        # Compute anisotropic behavior
        g = self.par.g0 + self.par.g1 * \
            np.exp(-self.par.g2 * b_field * np.exp(self.par.g3 * temperature))
        theta_symm = np.abs(theta + self.par.theta_peakOffset) * np.pi / 180
        theta_per = np.remainder(theta_symm, np.pi)
        theta = np.where(theta_per <= np.pi / 2, theta_per, np.pi - theta_per)

        # Compute critical current
        crit_curr = np.minimum(Jc_c, Jc_ab) + np.maximum(0, Jc_ab - Jc_c) / (
            1 + ((np.pi / 2 - theta) / g) ** self.par.nu
        )

        # Convert critical current to the desired unit
        if self.unit == "A/m²":
            return crit_curr
        elif self.unit == "A/m":
            return crit_curr * self.par.tSc  # tSc is the superconductor material width
        else:
            raise NotImplementedError
