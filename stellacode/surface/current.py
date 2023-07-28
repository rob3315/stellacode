import numpy as onp
from pydantic import BaseModel, Extra
from .abstract_surface import IntegrationParams
import pandas as pd
import typing as tp
from jax.typing import ArrayLike
from stellacode import np


def _stack(a, b):
    return 2 * onp.pi * onp.stack((a, b), axis=-1)


class AbstractCurrent(BaseModel):
    trainable_params: tp.List[str] = []
    current_op: tp.Optional[ArrayLike] = None
    grids: tp.Optional[tp.Tuple[ArrayLike, ArrayLike]] = None

    class Config:
        arbitrary_types_allowed = True

    def get_trainable_params(self):
        return {k: getattr(self, k) for k in self.trainable_params}

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

        dphi = onp.concatenate(dphi, axis=0)
        dphi = onp.concatenate((onp.zeros((2, *grids[0].shape, 2)), dphi), axis=0)

        dphi[0, :, :, 0] = onp.ones(grids[0].shape)
        dphi[1, :, :, 1] = onp.ones(grids[0].shape)

        return dphi

    def set_current_op(self, grids):
        self.current_op = self.get_matrix_from_grid(grids)


class Current(AbstractCurrent):
    num_pol: int
    num_tor: int
    sin_basis: bool = True
    cos_basis: bool = False
    trainable_params: tp.List[str] = []
    phi_mn: tp.Optional[ArrayLike] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.phi_mn is None:
            num_dims = (self.num_tor * 2 + 1) * self.num_pol + self.num_tor
            if self.sin_basis and self.cos_basis:
                num_dims *= 2
            self.phi_mn = onp.zeros(num_dims)

    def get_integration_params(self, factor: float = 4):
        return IntegrationParams(num_points_u=self.num_pol * factor, num_points_v=self.num_tor * factor)

    def _get_coeffs(self):
        grid = onp.mgrid[1 : (self.num_pol + 1), -self.num_tor : (self.num_tor + 1)].reshape((2, -1))
        xm = onp.concatenate((onp.zeros(self.num_tor), grid[0]))
        xn = onp.concatenate((-onp.arange(1, self.num_tor + 1), -grid[1]))

        return xm, xn

    def get_phi_mn(self):
        return self.phi_mn * 1e8

    def get_j_surface(self, phi_mn=None):
        if phi_mn is None:
            phi_mn = self.phi_mn
        return np.einsum("oijk,o->ijk", self.current_op, phi_mn)

    def get_matrix_from_grid(self, grids):
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
    num_pol: int
    num_tor: int
    sin_basis: bool = True
    cos_basis: bool = True
    trainable_params: tp.List[str] = ["phi_mn"]
    phi_mn: tp.Optional[ArrayLike] = None

    def get_coeffs(self):
        grid = onp.mgrid[1 : (self.num_pol + 2), 1 : (self.num_tor + 2)].reshape((2, -1))
        return grid[0], grid[1], onp.arange(1, (self.num_tor + 2))

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

        cosu = onp.cos(2 * onp.pi * xm * ugrid)
        cosv = onp.cos(2 * onp.pi * xn * vgrid)
        sinu = onp.sin(2 * onp.pi * xm * ugrid)
        sinv = onp.sin(2 * onp.pi * xn * vgrid)
        # the current potential is written as:
        # sin(2*onp.pi*u)*sin(2*onp.pi*v) and cos(2*onp.pi*u)*sin(2*onp.pi*v)
        # to ensure proper BC
        if self.sin_basis:
            # current for Phi=sin(2*onp.pi*xn*v)
            cosu0 = onp.cos(2 * onp.pi * xn0 * vgrid)
            dphi.append(_stack(xn0 * cosu0, onp.zeros_like(xn0 * cosu0)))
            # current for Phi=sin(2*onp.pi*xm*u)*sin(2*onp.pi*xn*v)
            dphi.append(_stack(xn * sinu * cosv, -xm * cosu * sinv))

        if self.cos_basis:
            # current for Phi=cos(2*onp.pi*xn*v)
            sinu0 = onp.sin(2 * onp.pi * xn0 * vgrid)
            dphi.append(_stack(-xn0 * sinu0, onp.zeros_like(xn0 * sinu0)))
            # current for Phi=cos(2*onp.pi*xm*u)*sin(2*onp.pi*xn*v)
            dphi.append(_stack(xn * cosu * cosv, xm * sinu * sinv))

        dphi = onp.concatenate(dphi, axis=0)
        dphi = onp.concatenate((onp.zeros((2, lu, lv, 2)), dphi), axis=0)

        dphi[0, :, :, 0] = onp.ones((lu, lv))
        dphi[1, :, :, 1] = onp.ones((lu, lv))

        return dphi
