import numpy as np
from pydantic import BaseModel, Extra


def phi_coeff_from_nb(k, phisize):
    """the coefficents of Phi are store as 1d array,
    this function make the conversion with a 2d matrix"""
    lm, ln = phisize
    if k < ln:
        return [0, k + 1]
    else:
        kk = k - ln
        return [1 + kk // (2 * ln + 1), kk % (2 * ln + 1) - ln]


def get_coeffs(lc, phisize):
    return np.array([phi_coeff_from_nb(k, phisize) for k in range(lc)])


class CurrentPotential(BaseModel):
    num_pol: int
    num_tor: int

    def get_matrix_from_grid(self, grids):
        """
        Only sine coefficients are implmented
        
        """
        ugrid, vgrid = grids
        lu, lv = ugrid.shape
        lc = self.num_pol * (2 * self.num_tor + 1) + self.num_tor

        coeffs = get_coeffs(lc, (self.num_pol, self.num_tor))
        coeffs
        xm = coeffs[:, None, None, 0]
        xn = coeffs[:, None, None, 1]
        angle = 2 * np.pi * (xm * ugrid[:, :] + xn * vgrid[:, :])
        dphi = np.stack(
            (-2 * np.pi * xn * np.cos(angle), 2 * np.pi * xm * np.cos(angle)),
            axis=-1,
        )
        dphi = np.concatenate((np.zeros((2, lu, lv, 2)), dphi), axis=0)

        dphi[0, :, :, 0] = np.ones((lu, lv))
        dphi[1, :, :, 1] = np.ones((lu, lv))

        return dphi
