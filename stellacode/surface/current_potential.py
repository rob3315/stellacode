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
    sin_basis: bool = True
    cos_basis: bool = False

    def get_matrix_from_grid(self, grids):
        ugrid, vgrid = grids
        lu, lv = ugrid.shape
        lc = self.num_pol * (2 * self.num_tor + 1) + self.num_tor

        coeffs = get_coeffs(lc, (self.num_pol, self.num_tor))
        coeffs
        xm = coeffs[:, None, None, 0]
        xn = coeffs[:, None, None, 1]
        angle = 2 * np.pi * (xm * ugrid[:, :] + xn * vgrid[:, :])
        dphi = []
        assert self.sin_basis or self.cos_basis
        if self.sin_basis:
            dphi.append(
                np.stack(
                    (-2 * np.pi * xn * np.cos(angle), 2 * np.pi * xm * np.cos(angle)),
                    axis=-1,
                )
            )
        if self.cos_basis:
            dphi.append(
                np.stack(
                    (2 * np.pi * xn * np.sin(angle), -2 * np.pi * xm * np.sin(angle)),
                    axis=-1,
                )
            )
        dphi = np.concatenate(dphi, axis=0)
        dphi = np.concatenate((np.zeros((2, lu, lv, 2)), dphi), axis=0)

        dphi[0, :, :, 0] = np.ones((lu, lv))
        dphi[1, :, :, 1] = np.ones((lu, lv))

        return dphi
