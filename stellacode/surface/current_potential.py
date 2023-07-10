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


def _stack(a, b):
    return 2 * np.pi * np.stack((a, b), axis=-1)


class CurrentPotential(BaseModel):
    num_pol: int
    num_tor: int
    sin_basis: bool = True
    cos_basis: bool = False
    zero_tor_bc: bool = False

    def get_matrix_from_grid(self, grids):
        ugrid, vgrid = grids  # u -> poloidal, v -> toroidal
        lu, lv = ugrid.shape
        lc = self.num_pol * (2 * self.num_tor + 1) + self.num_tor

        coeffs = get_coeffs(lc, (self.num_pol, self.num_tor))

        xm = coeffs[:, None, None, 0]
        xn = coeffs[:, None, None, 1]

        dphi = []
        assert self.sin_basis or self.cos_basis

        if self.zero_tor_bc:
            cosu = np.cos(2 * np.pi * xm * ugrid)
            cosv = np.cos(2 * np.pi * xn * ugrid)
            sinu = np.sin(2 * np.pi * xm * ugrid)
            sinv = np.sin(2 * np.pi * xn * ugrid)

            if self.sin_basis:
                dphi.append(_stack(-xn * sinu * cosv, xm * cosu * sinv))

            if self.cos_basis:
                dphi.append(_stack(xn * cosu * cosv, -xm * sinu * sinv))

        else:
            angle = 2 * np.pi * (xm * ugrid + xn * vgrid)

            if self.sin_basis:
                dphi.append(_stack(xn * np.cos(angle), -xm * np.cos(angle)))

            if self.cos_basis:
                dphi.append(_stack(xn * np.sin(angle), -xm * np.sin(angle)))

        dphi = np.concatenate(dphi, axis=0)
        dphi = np.concatenate((np.zeros((2, lu, lv, 2)), dphi), axis=0)

        dphi[0, :, :, 0] = np.ones((lu, lv))
        dphi[1, :, :, 1] = np.ones((lu, lv))

        return dphi

    def plot(self, dphi, grids):
        import matplotlib.pyplot as plt

        fig1, ax1 = plt.subplots()

        Q = ax1.quiver(grids[0], grids[1], dphi[..., 0], dphi[..., 1], units="width")
        qk = ax1.quiverkey(Q, 0.9, 0.9, 2, r"$2 \frac{m}{s}$", labelpos="E", coordinates="figure")
