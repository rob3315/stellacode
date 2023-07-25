import numpy as np
from pydantic import BaseModel, Extra
from .abstract_surface import IntegrationParams
import pandas as pd


def _stack(a, b):
    return 2 * np.pi * np.stack((a, b), axis=-1)


class AbstractCurrent(BaseModel):
    def _get_coeffs(self):
        raise NotImplementedError

    # def _get_grad_phi(self, gridu, gridv):
    #     raise NotImplementedError

    def get_matrix_from_grid(self, grids):
        # ugrid, vgrid = grids  # u -> poloidal, v -> toroidal
        # lu, lv = ugrid.shape
        xm, xn = self.get_coeffs()
        xm = xm[:, None, None]
        xn = xn[:, None, None]

        dphi = []
        assert self.sin_basis or self.cos_basis

        dphi = self._get_grad_phi(*grids)

        dphi = np.concatenate(dphi, axis=0)
        dphi = np.concatenate((np.zeros((2, *grids[0].shape, 2)), dphi), axis=0)

        dphi[0, :, :, 0] = np.ones(grids[0].shape)
        dphi[1, :, :, 1] = np.ones(grids[0].shape)

        return dphi


class Current(AbstractCurrent):
    num_pol: int
    num_tor: int
    sin_basis: bool = True
    cos_basis: bool = False

    def get_integration_params(self, factor: float = 4):
        return IntegrationParams(num_points_u=self.num_pol * factor, num_points_v=self.num_tor * factor)

    def _get_coeffs(self):
        grid = np.mgrid[1 : (self.num_pol + 1), -self.num_tor : (self.num_tor + 1)].reshape((2, -1))
        xm = np.concatenate((np.zeros(self.num_tor), grid[0]))
        xn = np.concatenate((-np.arange(1, self.num_tor + 1), -grid[1]))

        return xm, xn

    def get_matrix_from_grid(self, grids):
        ugrid, vgrid = grids  # u -> poloidal, v -> toroidal
        lu, lv = ugrid.shape
        xm, xn = self._get_coeffs()
        xm = xm[:, None, None]
        xn = xn[:, None, None]

        dphi = []
        assert self.sin_basis or self.cos_basis

        angle = 2 * np.pi * (xm * ugrid + xn * vgrid)

        if self.sin_basis:
            # Phi = sin(xm*u+xn*v)
            # dPhi/dv, -dPhi/du
            dphi.append(_stack(xn * np.cos(angle), -xm * np.cos(angle)))

        if self.cos_basis:
            # Phi = cos(xm*u+xn*v)
            dphi.append(-_stack(xn * np.sin(angle), xm * np.sin(angle)))

        dphi = np.concatenate(dphi, axis=0)
        dphi = np.concatenate((np.zeros((2, lu, lv, 2)), dphi), axis=0)

        dphi[0, :, :, 0] = np.ones((lu, lv))
        dphi[1, :, :, 1] = np.ones((lu, lv))

        return dphi

    def plot_phi_mn(self, phi_mn):
        shape = (self.num_pol, self.num_tor * 2 + 1)
        col = pd.Index(np.arange(-self.num_tor, (self.num_tor + 1)), name="toroidal")
        ind = pd.Index(np.arange(1, self.num_pol + 1), name="poloidal")
        ph_sin = pd.DataFrame(
            np.reshape(phi_mn[self.num_tor + 2 : self.num_tor + 2 + np.prod(shape)], shape),
            columns=col,
            index=ind,
        )
        ph_cos = pd.DataFrame(
            np.reshape(phi_mn[self.num_tor * 2 + 2 + np.prod(shape) :], shape),
            columns=col,
            index=ind,
        )

        import matplotlib.pyplot as plt
        import seaborn as sns

        f, axs = plt.subplots(2, 1, figsize=(8, 8))
        sns.heatmap(ph_sin, cmap="seismic", center=0, ax=axs[0])
        axs[0].set_title("sin coefficients")
        sns.heatmap(ph_cos, cmap="seismic", center=0, ax=axs[1])
        axs[0].set_title("cos coefficients")


class CurrentZeroTorBC(AbstractCurrent):
    num_pol: int
    num_tor: int
    sin_basis: bool = True
    cos_basis: bool = True

    def get_coeffs(self):
        grid = np.mgrid[1 : (self.num_pol + 2), 1 : (self.num_tor + 2)].reshape((2, -1))
        return grid[0], grid[1], np.arange(1, (self.num_tor + 2))

    def get_matrix_from_grid(self, grids):
        ugrid, vgrid = grids  # u -> poloidal, v -> toroidal
        vgrid = vgrid + 0.5 / vgrid.shape[1]
        lu, lv = ugrid.shape
        xm, xn, xn0 = self.get_coeffs()
        xm = xm[:, None, None]
        xn = xn[:, None, None]
        xn0 = xn0[:, None, None]

        dphi = []
        assert self.sin_basis or self.cos_basis

        cosu = np.cos(2 * np.pi * xm * ugrid)
        cosv = np.cos(2 * np.pi * xn * vgrid)
        sinu = np.sin(2 * np.pi * xm * ugrid)
        sinv = np.sin(2 * np.pi * xn * vgrid)
        # the current potential is written as:
        # sin(2*np.pi*u)*sin(2*np.pi*v) and cos(2*np.pi*u)*sin(2*np.pi*v)
        # to ensure proper BC
        if self.sin_basis:
            # current for Phi=sin(2*np.pi*xn*v)
            cosu0 = np.cos(2*np.pi * xn0 * vgrid)
            dphi.append(_stack(xn0 * cosu0, np.zeros_like(xn0 * cosu0)))
            # current for Phi=sin(2*np.pi*xm*u)*sin(2*np.pi*xn*v)
            dphi.append(_stack(xn * sinu * cosv, -xm * cosu * sinv))

        if self.cos_basis:
            # current for Phi=cos(2*np.pi*xn*v)
            sinu0 = np.sin(2*np.pi * xn0 * vgrid)
            dphi.append(_stack(-xn0 * sinu0, np.zeros_like(xn0 * sinu0)))
            # current for Phi=cos(2*np.pi*xm*u)*sin(2*np.pi*xn*v)
            dphi.append(_stack(xn * cosu * cosv, xm * sinu * sinv))

        dphi = np.concatenate(dphi, axis=0)
        dphi = np.concatenate((np.zeros((2, lu, lv, 2)), dphi), axis=0)

        dphi[0, :, :, 0] = np.ones((lu, lv))
        dphi[1, :, :, 1] = np.ones((lu, lv))

        return dphi
