import typing as tp

import jax
import numpy as onp
import pandas as pd
from jax import Array
from jax.typing import ArrayLike
from pydantic import BaseModel

from stellacode import np

from .abstract_surface import IntegrationParams


def _stack(a, b):
    return 2 * onp.pi * onp.stack((a, b), axis=-1)


class AbstractCurrent(BaseModel):
    """
    Abstract class for currents on surfaces

    Args:
        * num_pol: number of points in the poloidal direction
        * num_tor: number of points in the toroidal direction
        * net_currents: net currents along each direction
        * phi_mn: weights of the current basis functions
        * sin_basis: use the sine basis functions
        * cos_basis: use the cosine basis functions
        * trainable_params: list of trainable parameters
        * scale_phi_mn: scales the weights of the current basis functions
    """

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

    def get_phi(self, uv, phi_mn, max_val_v: float = 1):
        raise NotImplementedError

    def phi_op(self, uv, max_val_v: float = 1):
        return jax.grad(self.get_phi, 1)(uv, np.zeros(len(self.get_phi_mn())), max_val_v)

    def get_phi_on_grid(self, grid, phi_mn, max_val_v: float = 1):
        grid_ = np.reshape(grid, (2, -1))
        _, lu, lv = grid.shape
        phi = jax.vmap(self.get_phi, in_axes=(1, None, None), out_axes=0)
        phi_res = phi(grid_, phi_mn, max_val_v)
        phi = np.reshape(phi_res, (lu, lv))

        return phi

    def get_jac_phi_on_grid(self, grid, phi_mn, max_val_v: float = 1):
        grid_ = np.reshape(grid, (2, -1))
        _, lu, lv = grid.shape

        jac_phi = jax.jacobian(self.get_phi, argnums=0)
        jac_phi_vmap = jax.vmap(jac_phi, in_axes=(1, None, None), out_axes=0)
        jac_phi_res = jac_phi_vmap(grid_, phi_mn, max_val_v)
        jac_phi = np.reshape(jac_phi_res, (lu, lv, 2))

        return jac_phi

    def get_jac_phi_op_on_grid(self, grid, max_val_v: float = 1):
        grid_ = np.reshape(grid, (2, -1))
        _, lu, lv = grid.shape

        jac_phi = jax.jacobian(self.phi_op, argnums=0)
        jac_phi_vmap = jax.vmap(jac_phi, in_axes=(1, None), out_axes=0)
        jac_phi_res = jac_phi_vmap(grid_, max_val_v)
        jac_phi = np.reshape(jac_phi_res, (lu, lv, len(self.get_phi_mn()), 2))

        return jac_phi

    def get_hess_phi_op_on_grid(self, grid, max_val_v: float = 1):
        grid_ = np.reshape(grid, (2, -1))
        _, lu, lv = grid.shape

        hess_phi = jax.hessian(self.phi_op, argnums=0)
        hess_phi_vmap = jax.vmap(hess_phi, in_axes=(1, None), out_axes=0)
        hess_phi_res = hess_phi_vmap(grid_, max_val_v)
        hess_phi = np.reshape(hess_phi_res, (lu, lv, len(self.get_phi_mn()), 2, 2))

        return hess_phi

    def get_hess_phi_on_grid(self, grid, phi_mn, max_val_v: float = 1):
        grid_ = np.reshape(grid, (2, -1))
        _, lu, lv = grid.shape

        hess_phi = jax.hessian(self.get_phi, argnums=0)
        hess_phi_vmap = jax.vmap(hess_phi, in_axes=(1, None, None), out_axes=0)
        hess_phi_res = hess_phi_vmap(grid_, phi_mn, max_val_v)

        return np.reshape(hess_phi_res, (lu, lv, 2, 2))

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

    def set_phi_mn(self, phi_mn: Array):
        self.phi_mn = phi_mn / self.scale_phi_mn

    def get_trainable_params(self):
        return {k: getattr(self, k) for k in self.trainable_params}

    def _get_coeffs(self):
        raise NotImplementedError

    def __call__(self, grids, max_val_v: float = 1, grad: tp.Optional[str] = None):
        # Computing the jacobian to get the matrix of a linear function is very inefficient (slower by ~ an order of
        # magnitude), but it is automatic and in some settings it could be called only once.

        assert grad is None
        grids = np.stack(grids, axis=0)
        grad_phi = self.get_jac_phi_op_on_grid(grids, max_val_v=max_val_v)
        return np.stack((grad_phi[..., 1], -grad_phi[..., 0]), axis=-1)

    def get_grad_current_op(self, grids, max_val_v: float = 1):
        grids = np.stack(grids, axis=0)
        hess_phi = self.get_hess_phi_op_on_grid(grids, max_val_v=max_val_v)
        return np.stack((hess_phi[..., 1, :], -hess_phi[..., 0, :]), axis=-2)


class Current(AbstractCurrent):
    """Current with periodic boundary conditions"""

    def _get_coeffs(self):
        grid = onp.mgrid[1 : (self.num_pol + 1), -self.num_tor : (self.num_tor + 1)].reshape((2, -1))
        xm = onp.concatenate((onp.zeros(self.num_tor), grid[0]))
        xn = onp.concatenate((-onp.arange(1, self.num_tor + 1), -grid[1]))

        return xm, xn

    def get_phi(self, uv, phi_mn, max_val_v: float = 1.0):
        xm, xn = self._get_coeffs()
        v_ = uv[1] / max_val_v
        angle = 2 * onp.pi * (xm * uv[0] + xn * v_)
        phi = phi_mn[0] * v_ - phi_mn[1] * uv[0]
        if self.sin_basis:
            phi += np.sum(phi_mn[2 : 2 + len(xm)] * np.sin(angle))
        if self.cos_basis:
            phi += np.sum(phi_mn[2 + len(xm) :] * np.cos(angle))

        return phi

    def __call__(self, grids, max_val_v: float = 1, grad: tp.Optional[str] = None):
        ugrid, vgrid = grids  # u -> poloidal, v -> toroidal
        vgrid /= max_val_v
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
            if grad == "u":
                dphi.append(2 * onp.pi * xm[..., None] * _stack(-xn * onp.sin(angle) / max_val_v, xm * onp.sin(angle)))
            elif grad == "v":
                dphi.append(
                    2
                    * onp.pi
                    * xn[..., None]
                    / max_val_v
                    * _stack(-xn * onp.sin(angle) / max_val_v, xm * onp.sin(angle))
                )
            elif grad is None:
                dphi.append(_stack(xn * onp.cos(angle) / max_val_v, -xm * onp.cos(angle)))
            else:
                raise NotImplementedError

        if self.cos_basis:
            # Phi = cos(xm*u+xn*v)
            if grad == "u":
                dphi.append(2 * onp.pi * xm[..., None] * _stack(-xn * onp.cos(angle) / max_val_v, xm * onp.cos(angle)))
            elif grad == "v":
                dphi.append(
                    2
                    * onp.pi
                    * xn[..., None]
                    / max_val_v
                    * _stack(-xn * onp.cos(angle) / max_val_v, xm * onp.cos(angle))
                )
            elif grad is None:
                dphi.append(_stack(-xn * onp.sin(angle) / max_val_v, xm * onp.sin(angle)))
            else:
                raise NotImplementedError

        dphi = onp.concatenate(dphi, axis=0)
        dphi = onp.concatenate((onp.zeros((2, lu, lv, 2)), dphi), axis=0)
        if grad is None:
            dphi[0, :, :, 0] = onp.ones((lu, lv)) / max_val_v
            dphi[1, :, :, 1] = onp.ones((lu, lv))

        return dphi

    def get_grad_current_op(self, grids, max_val_v: float = 1):
        """
        Dimensions of returned op are: dimensions: Ncurrent_op x Nu x Nv x N_j_surf x N_grad
        """
        return np.stack(
            (self(grids, max_val_v=max_val_v, grad="u"), self(grids, max_val_v=max_val_v, grad="v")),
            axis=-1,
        )

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
    """
    Current with periodic boundary condition along the poloidal axis and zero boundary
    conditions along the toroidal axis
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        num_dims = 0
        if self.sin_basis:
            num_dims += (self.num_pol + 1) * (self.num_tor + 1) + self.num_tor + 1
        if self.cos_basis:
            num_dims += (self.num_pol + 1) * (self.num_tor + 1)
        self.phi_mn = onp.zeros(num_dims)

    def get_phi(self, uv, phi_mn, max_val_v: float = 1.0):
        xm, xn, xn0 = self._get_coeffs()
        v_ = uv[1] / max_val_v
        phi = phi_mn[0] * v_ - phi_mn[1] * uv[0]

        if self.sin_basis:
            phi += np.sum(phi_mn[2 : 2 + len(xn0)] * np.sin(onp.pi * xn0 * v_))
            phi += np.sum(
                phi_mn[2 + len(xn0) : 2 + len(xn0) + len(xm)]
                * np.sin(2 * onp.pi * xm * uv[0])
                * np.sin(onp.pi * xn * v_)
            )

        if self.cos_basis:
            phi += np.sum(phi_mn[2 + len(xn0) + len(xm) :] * np.cos(2 * onp.pi * xm * uv[0]) * np.sin(onp.pi * xn * v_))

        return phi

    def _get_coeffs(self):
        grid = onp.mgrid[1 : (self.num_pol + 2), 1 : (self.num_tor + 2)].reshape((2, -1))
        return grid[0], grid[1], onp.arange(1, (self.num_tor + 2))

    def __call__(self, grids, max_val_v: float = 1, grad: tp.Optional[str] = None):
        ugrid, vgrid = grids  # u -> poloidal, v -> toroidal
        vgrid = vgrid / max_val_v
        # vgrid = vgrid + 0.5 / vgrid.shape[1]
        assert np.all(vgrid > 0)
        assert np.all(vgrid < 1)

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
        # sin(2*onp.pi*u)*sin(onp.pi*v) and cos(2*onp.pi*u)*sin(onp.pi*v)
        # to ensure proper BC
        if self.sin_basis:
            zero_v0 = np.zeros_like(xn0 * vgrid)

            if grad == "u":
                # current for Phi=sin(2*onp.pi*xn*v)
                dphi.append(_stack(zero_v0, zero_v0))
                # current for Phi=sin(2*onp.pi*xm*u)*sin(onp.pi*xn*v)
                dphi.append(
                    _stack(2 * onp.pi * xm * xn * cosu * cosv / 2 / max_val_v, 2 * onp.pi * xm**2 * sinu * sinv)
                )
            elif grad == "v":
                # current for Phi=sin(2*onp.pi*xn*v)
                sinv0 = onp.sin(onp.pi * xn0 * vgrid)
                dphi.append(_stack(-onp.pi * xn0**2 * sinv0 / max_val_v / 2, zero_v0))
                # current for Phi=sin(2*onp.pi*xm*u)*sin(onp.pi*xn*v)
                dphi.append(_stack(-onp.pi * xn**2 * sinu * sinv / 2 / max_val_v, -onp.pi * xn * xm * cosu * cosv))
            elif grad is None:
                # current for Phi=sin(onp.pi*xn*v)
                cosv0 = onp.cos(onp.pi * xn0 * vgrid)
                dphi.append(_stack(xn0 * cosv0 / max_val_v / 2, onp.zeros_like(xn0 * cosv0)))
                # current for Phi=sin(2*onp.pi*xm*u)*sin(onp.pi*xn*v)
                dphi.append(_stack(xn * sinu * cosv / 2 / max_val_v, -xm * cosu * sinv))

            else:
                raise NotImplementedError

        if self.cos_basis:
            if grad == "u":
                dphi.append(
                    _stack(-2 * onp.pi * xm * xn * sinu * cosv / 2 / max_val_v, 2 * onp.pi * xm**2 * cosu * sinv)
                )
            elif grad == "v":
                dphi.append(_stack(-onp.pi * xn**2 * cosu * sinv / 2 / max_val_v, onp.pi * xn * xm * sinu * cosv))
            elif grad is None:
                # current for Phi=cos(2*onp.pi*xm*u)*sin(onp.pi*xn*v)
                dphi.append(_stack(xn * cosu * cosv / 2 / max_val_v, xm * sinu * sinv))
            else:
                raise NotImplementedError

        dphi = onp.concatenate(dphi, axis=0)
        dphi = onp.concatenate((onp.zeros((2, lu, lv, 2)), dphi), axis=0)
        if grad is None:
            dphi[0, :, :, 0] = onp.ones((lu, lv)) / max_val_v
            dphi[1, :, :, 1] = onp.ones((lu, lv))

        return dphi

    def get_grad_current_op(self, grids, max_val_v: float = 1):
        """
        Dimensions of returned op are: dimensions: Ncurrent_op x Nu x Nv x N_j_surf x N_grad
        """
        return np.stack(
            (self(grids, max_val_v=max_val_v, grad="u"), self(grids, max_val_v=max_val_v, grad="v")),
            axis=-1,
        )
