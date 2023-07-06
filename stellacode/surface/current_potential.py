import numpy as onp
from pydantic import BaseModel, Extra

from stellacode import np


def phi_coeff_from_nb(k, phisize):
    """the coefficents of Phi are store as 1d array,
    this function make the conversion with a 2d matrix"""
    lm, ln = phisize
    if k < ln:
        return (0, k + 1)
    else:
        kk = k - ln
        return (1 + kk // (2 * ln + 1), kk % (2 * ln + 1) - ln)


class CurrentPotential(BaseModel):
    num_pol: int
    num_tor: int

    def get_matrix_from_grid(self, grids):
        # lm, ln = phisize
        ugrid, vgrid = grids
        lu, lv = ugrid.shape
        lc = self.num_pol * (2 * self.num_tor + 1) + self.num_tor
        matrix_dPhi = onp.zeros((lc + 2, lu, lv, 2))
        # we start with du:
        matrix_dPhi[0, :, :, 0] = onp.ones((lu, lv))
        # dv
        matrix_dPhi[1, :, :, 1] = onp.ones((lu, lv))
        for coeff in range(lc):
            m, n = phi_coeff_from_nb(coeff, (self.num_pol, self.num_tor))
            # Phi=sin(2pi(mu+nv))
            # \nabla\perp Phi = (-2 pi n cos(2pi(mu+nv)),2pi m cos(2pi(mu+nv)))
            matrix_dPhi[coeff + 2, :, :, 0] = -2 * onp.pi * n * onp.cos(2 * onp.pi * (m * ugrid + n * vgrid))
            matrix_dPhi[coeff + 2, :, :, 1] = 2 * onp.pi * m * onp.cos(2 * onp.pi * (m * ugrid + n * vgrid))

        return matrix_dPhi