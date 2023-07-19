import numpy as np
from pydantic import BaseModel, Extra
from .abstract_surface import IntegrationParams


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
            # dPhi/dv, -dPhi/du
            dphi.append(_stack(xn * np.cos(angle), -xm * np.cos(angle)))

        if self.cos_basis:
            dphi.append(_stack(xn * np.sin(angle), -xm * np.sin(angle)))

        dphi = np.concatenate(dphi, axis=0)
        dphi = np.concatenate((np.zeros((2, lu, lv, 2)), dphi), axis=0)

        dphi[0, :, :, 0] = np.ones((lu, lv))
        dphi[1, :, :, 1] = np.ones((lu, lv))

        return dphi


class CurrentZeroTorBC(AbstractCurrent):
    num_pol: int
    num_tor: int
    sin_basis: bool = True
    cos_basis: bool = False

    def get_coeffs(self):
        grid = np.mgrid[1 : (self.num_pol + 2), 1 : (self.num_tor + 2)].reshape((2, -1))
        return grid[0], grid[1]

    def get_matrix_from_grid(self, grids):
        ugrid, vgrid = grids  # u -> poloidal, v -> toroidal
        vgrid = vgrid + 0.5 / vgrid.shape[1]
        lu, lv = ugrid.shape
        xm, xn = self.get_coeffs()
        xm = xm[:, None, None]
        xn = xn[:, None, None]

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
            # current for Phi=sin(2*np.pi*xm*u)*sin(2*np.pi*xn*v)
            dphi.append(_stack(xn * sinu * cosv, -xm * cosu * sinv))

        if self.cos_basis:
            # current for Phi=cos(2*np.pi*xm*u)*sin(2*np.pi*xn*v)
            dphi.append(_stack(xn * cosu * cosv, xm * sinu * sinv))

        dphi = np.concatenate(dphi, axis=0)
        dphi = np.concatenate((np.zeros((2, lu, lv, 2)), dphi), axis=0)

        dphi[0, :, :, 0] = np.ones((lu, lv))
        dphi[1, :, :, 1] = np.ones((lu, lv))

        return dphi
