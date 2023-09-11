import typing as tp

import numpy as onp
import pandas as pd
from jax.typing import ArrayLike
from pydantic import BaseModel

from stellacode import np
import jax
from .abstract_surface import IntegrationParams


def _stack(a, b):
    return 2 * onp.pi * onp.stack((a, b), axis=-1)


class AbstractCurrent(BaseModel):
    num_pol: int
    num_tor: int
    net_currents: ArrayLike
    phi_mn: ArrayLike = onp.zeros(1)
    sin_basis: bool = True
    cos_basis: bool = False
    trainable_params: tp.List[str] = ["phi_mn"]
    scale_phi_mn: float = 1e8

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        num_dims = (self.num_tor * 2 + 1) * self.num_pol + self.num_tor
        if self.sin_basis and self.cos_basis:
            num_dims *= 2
        self.phi_mn = onp.zeros(num_dims)

    # def phi(self, uv):
    #     raise NotImplementedError

    # def get_phi_on_grid(self, grid):
    #     grid_ = np.reshape(grid, (2, -1))
    #     _, lu, lv = grid.shape
    #     surf = jax.vmap(self.get_phi, in_axes=1, out_axes=0)
    #     surf_res = surf(grid_)
    #     phi = np.reshape(surf_res, (lu, lv))

    #     return phi

    # def get_jac_phi_on_grid(self, grid):
    #     grid_ = np.reshape(grid, (2, -1))
    #     _, lu, lv = grid.shape

    #     jac_surf = jax.jacobian(self.get_phi, argnums=0)
    #     jac_surf_vmap = jax.vmap(jac_surf, in_axes=1, out_axes=0)
    #     jac_surf_res = jac_surf_vmap(grid_)
    #     jac_phi = np.reshape(jac_surf_res, (lu, lv, 2))

    #     return jac_phi

    # def get_hess_phi_on_grid(self, grid):
    #     grid_ = np.reshape(grid, (2, -1))
    #     _, lu, lv = grid.shape

    #     hess_surf = jax.hessian(self.get_phi, argnums=0, holomorphic=False)
    #     hess_surf_vmap = jax.vmap(hess_surf, in_axes=1, out_axes=0)
    #     hess_surf_res = hess_surf_vmap(grid_)

    #     return np.reshape(hess_surf_res, (lu, lv, 2, 2))

    def get_integration_params(self, factor: float = 4):
        return IntegrationParams(num_points_u=self.num_pol * factor, num_points_v=self.num_tor * factor)

    def get_j_surface(self, phi_mn=None):
        if phi_mn is None:
            phi_mn = self.phi_mn
        return np.einsum("oijk,o->ijk", self.current_op, phi_mn)

    def get_phi_mn(self):
        phi_mn = self.phi_mn * self.scale_phi_mn
        if self.net_currents is not None:
            phi_mn = np.concatenate((self.net_currents, phi_mn))
        return phi_mn

    def set_phi_mn(self, phi_mn: float):
        self.phi_mn = phi_mn / self.scale_phi_mn

    def get_trainable_params(self):
        return {k: getattr(self, k) for k in self.trainable_params}

    def _get_coeffs(self):
        raise NotImplementedError

    @classmethod
    def _get_matrix_from_grid(cls, grids, sin_basis, cos_basis, num_pol, num_tor):
        raise NotImplementedError


class Current(AbstractCurrent):
    def _get_coeffs(self):
        grid = onp.mgrid[1 : (self.num_pol + 1), -self.num_tor : (self.num_tor + 1)].reshape((2, -1))
        xm = onp.concatenate((onp.zeros(self.num_tor), grid[0]))
        xn = onp.concatenate((-onp.arange(1, self.num_tor + 1), -grid[1]))

        return xm, xn

    def _get_matrix_from_grid(self, grids):
        ugrid, vgrid = grids  # u -> poloidal, v -> toroidal
        lu, lv = ugrid.shape
        xm, xn = self._get_coeffs()
        xm = xm[:, None, None]
        xn = xn[:, None, None]

        dphi = []
        assert self.sin_basis or self.cos_basis

        angle = 2 * onp.pi * (xm * ugrid + xn * vgrid)

        if self.sin_basis:
            # Phi = sin(xm*u+xn*v)
            # dPhi/dv, -dPhi/du
            dphi.append(_stack(xn * onp.cos(angle), -xm * onp.cos(angle)))

        if self.cos_basis:
            # Phi = cos(xm*u+xn*v)
            dphi.append(-_stack(xn * onp.sin(angle), xm * onp.sin(angle)))

        dphi = onp.concatenate(dphi, axis=0)
        dphi = onp.concatenate((onp.zeros((2, lu, lv, 2)), dphi), axis=0)

        dphi[0, :, :, 0] = onp.ones((lu, lv))
        dphi[1, :, :, 1] = onp.ones((lu, lv))

        return dphi

    def plot_phi_mn(self, phi_mn):
        shape = (self.num_pol, self.num_tor * 2 + 1)
        col = pd.Index(onp.arange(-self.num_tor, (self.num_tor + 1)), name="toroidal")
        ind = pd.Index(onp.arange(1, self.num_pol + 1), name="poloidal")
        ph_sin = pd.DataFrame(
            onp.reshape(phi_mn[self.num_tor + 2 : self.num_tor + 2 + onp.prod(shape)], shape),
            columns=col,
            index=ind,
        )
        ph_cos = pd.DataFrame(
            onp.reshape(phi_mn[self.num_tor * 2 + 2 + onp.prod(shape) :], shape),
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
    def _get_coeffs(self):
        grid = onp.mgrid[1 : (self.num_pol + 2), 1 : (self.num_tor + 2)].reshape((2, -1))
        return grid[0], grid[1], onp.arange(1, (self.num_tor + 2))

    def _get_matrix_from_grid(self, grids):
        ugrid, vgrid = grids  # u -> poloidal, v -> toroidal
        vgrid = vgrid + 0.5 / vgrid.shape[1]
        lu, lv = ugrid.shape
        xm, xn, xn0 = self._get_coeffs()
        xm = xm[:, None, None]
        xn = xn[:, None, None]
        xn0 = xn0[:, None, None]

        dphi = []
        assert self.sin_basis or self.cos_basis

        cosu = onp.cos(2 * onp.pi * xm * ugrid)
        cosv = onp.cos(onp.pi * xn * vgrid)
        sinu = onp.sin(2 * onp.pi * xm * ugrid)
        sinv = onp.sin(onp.pi * xn * vgrid)
        # the current potential is written as:
        # sin(2*onp.pi*u)*sin(2*onp.pi*v) and cos(2*onp.pi*u)*sin(2*onp.pi*v)
        # to ensure proper BC
        if self.sin_basis:
            # current for Phi=sin(2*onp.pi*xn*v)
            cosv0 = onp.cos(onp.pi * xn0 * vgrid)
            dphi.append(_stack(xn0 * cosv0, onp.zeros_like(xn0 * cosv0)))
            # current for Phi=sin(2*onp.pi*xm*u)*sin(onp.pi*xn*v)
            dphi.append(_stack(xn * sinu * cosv / 2, -xm * cosu * sinv))

        if self.cos_basis:
            # current for Phi=cos(2*onp.pi*xm*u)*sin(2*onp.pi*xn*v)
            dphi.append(_stack(xn * cosu * cosv / 2, xm * sinu * sinv))

        dphi = onp.concatenate(dphi, axis=0)
        dphi = onp.concatenate((onp.zeros((2, lu, lv, 2)), dphi), axis=0)

        dphi[0, :, :, 0] = onp.ones((lu, lv))
        dphi[1, :, :, 1] = onp.ones((lu, lv))

        return dphi
