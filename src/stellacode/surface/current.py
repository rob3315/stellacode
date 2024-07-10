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

    model_config = dict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs):
        """
        Initialize the class.

        This method calculates the number of dimensions based on the basis functions used.
        If both sin and cos basis functions are used, the number of dimensions is multiplied by 2.
        The `phi_mn` parameter is initialized with zeros based on the calculated number of dimensions.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments are passed to the parent class.
        """
        super().__init__(**kwargs)

        # Calculate the number of dimensions
        num_dims = (self.num_tor * 2 + 1) * self.num_pol + self.num_tor

        # If both sin and cos basis functions are used, multiply the number of dimensions by 2
        if self.sin_basis and self.cos_basis:
            num_dims *= 2

        # Initialize `phi_mn` with zeros based on the calculated number of dimensions
        self.phi_mn = onp.zeros(num_dims)

    def get_phi(self, uv, phi_mn, max_val_v: float = 1):
        raise NotImplementedError

    def phi_op(self, uv, max_val_v: float = 1):
        """
        Calculate the gradient of the current potential with respect to the grid points.

        Parameters
        ----------
        uv : ArrayLike
            The grid points in the u-v plane. Shape: (2, N).
        max_val_v : float, optional
            The maximum value of the v-component of the grid points, by default 1.

        Returns
        -------
        ArrayLike
            The gradient of the current potential with respect to the grid points. Shape: (N, 2).
        """
        # Calculate the gradient of the current potential with respect to the grid points.
        # We use the `jax.grad` function to calculate the gradient along the first axis.
        # The `get_phi` function is the target function for the gradient calculation.
        # We pass `uv`, `np.zeros(len(self.get_phi_mn()))` and `max_val_v` as arguments to the `get_phi` function.
        # The `1` argument specifies the axis along which to calculate the gradient.
        return jax.grad(self.get_phi, 1)(uv, np.zeros(len(self.get_phi_mn())), max_val_v)

    def get_phi_on_grid(self, grid, phi_mn, max_val_v: float = 1):
        """
        Calculate the current potential on the grid.

        Parameters
        ----------
        grid : ArrayLike
            The grid points in the u-v plane. Shape: (2, lu, lv).
        phi_mn : ArrayLike
            The coefficients of the current potential.
        max_val_v : float, optional
            The maximum value of the v-component of the grid points, by default 1.

        Returns
        -------
        ArrayLike
            The current potential on the grid. Shape: (lu, lv).
        """
        # Reshape the grid points to a 2D array
        grid_ = np.reshape(grid, (2, -1))
        _, lu, lv = grid.shape

        # Calculate the current potential on the grid using the `jax.vmap` function.
        # We pass `grid_`, `phi_mn` and `max_val_v` as arguments to the `get_phi` function.
        # The `out_axes=0` argument specifies the axis along which to perform the mapping.
        phi = jax.vmap(self.get_phi, in_axes=(1, None, None), out_axes=0)
        phi_res = phi(grid_, phi_mn, max_val_v)

        # Reshape the result to a 2D array with the shape of the grid points
        phi = np.reshape(phi_res, (lu, lv))

        return phi

    def get_jac_phi_on_grid(self, grid, phi_mn, max_val_v: float = 1):
        """
        Calculate the Jacobian of the current potential on the grid.

        Parameters
        ----------
        grid : ArrayLike
            The grid points in the u-v plane. Shape: (2, lu, lv).
        phi_mn : ArrayLike
            The coefficients of the current potential.
        max_val_v : float, optional
            The maximum value of the v-component of the grid points, by default 1.

        Returns
        -------
        ArrayLike
            The Jacobian of the current potential on the grid. Shape: (lu, lv, 2).
        """
        # Reshape the grid points to a 2D array
        grid_ = np.reshape(grid, (2, -1))
        _, lu, lv = grid.shape

        # Calculate the Jacobian of the current potential on the grid using the `jax.vmap` function.
        # We pass `grid_`, `phi_mn` and `max_val_v` as arguments to the `get_phi` function.
        # The `out_axes=0` argument specifies the axis along which to perform the mapping.
        jac_phi = jax.jacobian(self.get_phi, argnums=0)
        jac_phi_vmap = jax.vmap(jac_phi, in_axes=(1, None, None), out_axes=0)
        jac_phi_res = jac_phi_vmap(grid_, phi_mn, max_val_v)

        # Reshape the result to a 2D array with the shape of the grid points
        jac_phi = np.reshape(jac_phi_res, (lu, lv, 2))

        return jac_phi

    def get_jac_phi_op_on_grid(self, grid, max_val_v: float = 1):
        """
        Calculate the Jacobian of the current potential operator on the grid.

        Parameters
        ----------
        grid : ArrayLike
            The grid points in the u-v plane. Shape: (2, lu, lv).
        max_val_v : float, optional
            The maximum value of the v-component of the grid points, by default 1.

        Returns
        -------
        ArrayLike
            The Jacobian of the current potential operator on the grid.
            Shape: (lu, lv, N_phi_mn, 2).

        """
        # Reshape the grid points to a 2D array
        grid_ = np.reshape(grid, (2, -1))
        _, lu, lv = grid.shape

        # Calculate the Jacobian of the current potential operator on the grid
        # using the `jax.vmap` function.
        # We pass `grid_` and `max_val_v` as arguments to the `phi_op` function.
        # The `out_axes=0` argument specifies the axis along which to perform the mapping.
        jac_phi = jax.jacobian(self.phi_op, argnums=0)
        jac_phi_vmap = jax.vmap(jac_phi, in_axes=(1, None), out_axes=0)
        jac_phi_res = jac_phi_vmap(grid_, max_val_v)

        # Reshape the result to a 4D array with the shape of the grid points,
        # number of phi_mn coefficients and the number of partial derivatives.
        jac_phi = np.reshape(jac_phi_res, (lu, lv, len(self.get_phi_mn()), 2))

        return jac_phi

    def get_hess_phi_op_on_grid(self, grid, max_val_v: float = 1):
        """
        Calculate the Hessian of the current potential operator on the grid.

        Parameters
        ----------
        grid : ArrayLike
            The grid points in the u-v plane. Shape: (2, lu, lv).
        max_val_v : float, optional
            The maximum value of the v-component of the grid points, by default 1.

        Returns
        -------
        ArrayLike
            The Hessian of the current potential operator on the grid.
            Shape: (lu, lv, N_phi_mn, 2, 2).
            Where N_phi_mn is the number of phi_mn coefficients.
        """
        # Reshape the grid points to a 2D array
        grid_ = np.reshape(grid, (2, -1))
        _, lu, lv = grid.shape

        # Calculate the Hessian of the current potential operator on the grid
        # using the `jax.hessian` function.
        # We pass `grid_` and `max_val_v` as arguments to the `phi_op` function.
        # The `argnums=0` argument specifies the positional argument to calculate the Hessian with respect to.
        # The `out_axes=0` argument specifies the axis along which to perform the mapping.
        hess_phi = jax.hessian(self.phi_op, argnums=0)
        hess_phi_vmap = jax.vmap(hess_phi, in_axes=(1, None), out_axes=0)
        hess_phi_res = hess_phi_vmap(grid_, max_val_v)

        # Reshape the result to a 4D array with the shape of the grid points,
        # number of phi_mn coefficients and the number of partial derivatives.
        hess_phi = np.reshape(
            hess_phi_res, (lu, lv, len(self.get_phi_mn()), 2, 2))

        return hess_phi

    def get_hess_phi_on_grid(self, grid, phi_mn, max_val_v: float = 1):
        """
        Calculate the Hessian of the current potential on the grid.

        Parameters
        ----------
        grid : ArrayLike
            The grid points in the u-v plane. Shape: (2, lu, lv).
        phi_mn : ArrayLike
            The coefficients of the current potential.
        max_val_v : float, optional
            The maximum value of the v-component of the grid points, by default 1.

        Returns
        -------
        ArrayLike
            The Hessian of the current potential on the grid. Shape: (lu, lv, 2, 2).
        """
        # Reshape the grid points to a 2D array
        grid_ = np.reshape(grid, (2, -1))
        _, lu, lv = grid.shape

        # Calculate the Hessian of the current potential on the grid using the `jax.hessian` function.
        # We pass `grid_`, `phi_mn` and `max_val_v` as arguments to the `get_phi` function.
        # The `argnums=0` argument specifies the positional argument to calculate the Hessian with respect to.
        # The `out_axes=0` argument specifies the axis along which to perform the mapping.
        hess_phi = jax.hessian(self.get_phi, argnums=0)
        hess_phi_vmap = jax.vmap(hess_phi, in_axes=(1, None, None), out_axes=0)
        hess_phi_res = hess_phi_vmap(grid_, phi_mn, max_val_v)

        # Reshape the result to a 3D array with the shape of the grid points and the number of partial derivatives.
        return np.reshape(hess_phi_res, (lu, lv, 2, 2))

    def get_integration_params(self, factor: float = 4):
        return IntegrationParams(num_points_u=self.num_pol * factor, num_points_v=self.num_tor * factor)

    def get_j_surface(self, phi_mn=None):
        """
        Compute the contravariant components of the surface current.

        Parameters
        ----------
        phi_mn : ArrayLike, optional
            Vector of Fourier coefficients of the surface current.
            If None, use self.phi_mn.

        Returns
        -------
        numpy.ndarray
            Contravariant components of the surface current.
            Dimensions: Nu x Nv x N_current_op
        """
        # If phi_mn is not provided, use self.phi_mn
        if phi_mn is None:
            phi_mn = self.phi_mn

        # Compute the contravariant components of the surface current
        #   1) Multiply by the current operator
        #   2) Multiply by the vector of Fourier coefficients
        return np.einsum("oijk,o->ijk", self.current_op, phi_mn)

    def get_phi_mn(self):
        """
        Get the vector of Fourier coefficients of the surface current.

        Returns
        -------
        numpy.ndarray
            Vector of Fourier coefficients of the surface current.
            Dimensions: Nu*Nv*N_current_op

        Notes
        -----
        The vector of Fourier coefficients is scaled by `scale_phi_mn` and concatenated with the net currents
        if `net_currents` is not None.
        """
        # Scale the vector of Fourier coefficients
        phi_mn = self.phi_mn * self.scale_phi_mn

        # Concatenate the net currents with phi_mn if net currents are given
        if self.net_currents is not None:
            phi_mn = np.concatenate((self.net_currents, phi_mn))

        return phi_mn

    def set_phi_mn(self, phi_mn: Array):
        """
        Set the vector of Fourier coefficients of the surface current.

        Parameters
        ----------
        phi_mn : ArrayLike
            Vector of Fourier coefficients of the surface current.
            Dimensions: Nu*Nv*N_current_op

        Notes
        -----
        The vector of Fourier coefficients is scaled by `scale_phi_mn`.
        """
        # Scale the vector of Fourier coefficients
        self.phi_mn = phi_mn / self.scale_phi_mn

    def get_trainable_params(self):
        """
        Return the trainable parameters from the current object.

        Returns
        -------
        dict
            A dictionary of the trainable parameters, where each key is the name
            of the parameter and the corresponding value is the parameter value.
        """
        # Return a dictionary of the trainable parameters
        return {
            k: getattr(self, k)  # Get the value of the parameter
            # Iterate over the trainable parameters
            for k in self.trainable_params
        }

    def _get_coeffs(self):
        raise NotImplementedError

    def __call__(self, grids, max_val_v: float = 1, grad: tp.Optional[str] = None):
        """
        Compute the gradient of the surface current.

        Parameters
        ----------
        grids : tuple of numpy.ndarray
            Tuple of u and v grid points.
            Dimensions: Nu x Nv
        max_val_v : float, optional
            Maximum value of v.
            Default is 1.
        grad : str, optional
            Type of gradient.
            "u" for gradient with respect to u.
            "v" for gradient with respect to v.
            None for both gradients.
            Default is None.

        Returns
        -------
        numpy.ndarray
            Gradient of the surface current.
            Dimensions: Nu x Nv x 2 x N_current_op
        """
        # Computing the jacobian to get the matrix of a linear function is very inefficient (slower by ~ an order of
        # magnitude), but it is automatic and in some settings it could be called only once.

        # Check if the gradient is None
        assert grad is None

        # Stack the grid points
        grids = np.stack(grids, axis=0)

        # Compute the gradient of the surface current
        grad_phi = self.get_jac_phi_op_on_grid(grids, max_val_v=max_val_v)

        # Stack the gradients with respect to u and v
        grad_stacked = np.stack((grad_phi[..., 1], -grad_phi[..., 0]), axis=-1)

        # Transpose the stacked gradients to move the last axis to the second-to-last position
        return np.transpose(grad_stacked, (2, 0, 1, 3))

    def get_grad_current_op(self, grids, max_val_v: float = 1):
        """
        Compute the gradient of the current density operator.

        Parameters
        ----------
        grids : array-like
            List of grids on which to compute the current density operator.
            The grids should be of shape (Nu, Nv).
        max_val_v : float, optional
            Maximum value of the toroidal coordinate.
            Default value is 1.

        Returns
        -------
        op : ndarray
            Array of shape (Ncurrent_op, Nu, Nv, 2, N_grad).
            The gradient of the current density operator.
            The current density operator is computed on the grids (u, v)
            and the values of the operator are stacked along the last
            dimension.
        """
        # Stack the grid points
        grids = np.stack(grids, axis=0)

        # Compute the Hessian of the current potential operator
        hess_phi = self.get_hess_phi_op_on_grid(grids, max_val_v=max_val_v)

        # Stack the second partial derivatives with respect to u and v
        grad_stacked = np.stack(
            (hess_phi[..., 1, :], -hess_phi[..., 0, :]), axis=-2)

        # Transpose the stacked gradients to move the last axis to the second-to-last position
        return np.transpose(grad_stacked, (2, 0, 1, 3, 4))


class Current(AbstractCurrent):
    """Current with periodic boundary conditions"""

    def _get_coeffs(self):
        """
        Get the coefficients for the basis functions.

        Returns
        -------
        xm : ndarray
            Coefficients for the poloidal basis functions.
        xn : ndarray
            Coefficients for the toroidal basis functions.
        """
        # Create a grid of poloidal and toroidal indices
        grid = onp.mgrid[1: (self.num_pol + 1),
                         -self.num_tor: (self.num_tor + 1)].reshape((2, -1))

        # Extract the coefficients for the basis functions
        # Poloidal coefficients
        xm = onp.concatenate((onp.zeros(self.num_tor), grid[0]))
        # Toroidal coefficients
        xn = onp.concatenate((-onp.arange(1, self.num_tor + 1), -grid[1]))

        return xm, xn

    def get_phi(self, uv, phi_mn, max_val_v: float = 1.0):
        """
        Calculate the value of the current at a given point (uv) using the basis functions.

        Parameters
        ----------
        uv : ndarray
            The point (poloidal, toroidal) at which to evaluate the current.
        phi_mn : ndarray
            The coefficients of the basis functions.
        max_val_v : float, optional
            The maximum value of the toroidal coordinate. Default is 1.0.

        Returns
        -------
        float
            The value of the current at the given point.
        """
        # Get the coefficients of the basis functions
        xm, xn = self._get_coeffs()

        # Normalize the toroidal coordinate
        v_ = uv[1] / max_val_v

        # Calculate the angle of the basis functions
        angle = 2 * onp.pi * (xm * uv[0] + xn * v_)

        # Calculate the value of the current using the basis functions
        phi = phi_mn[0] * v_ - phi_mn[1] * uv[0]

        # Add the contributions from the sin basis functions
        if self.sin_basis:
            phi += np.sum(phi_mn[2: 2 + len(xm)] * np.sin(angle))

        # Add the contributions from the cos basis functions
        if self.cos_basis:
            phi += np.sum(phi_mn[2 + len(xm):] * np.cos(angle))

        return phi

    def __call__(self, grids, max_val_v: float = 1, grad: tp.Optional[str] = None):
        """
        Evaluate the current density on the surface.

        Parameters
        ----------
        grids : tuple of ndarrays
            Grids of poloidal and toroidal coordinates.
        max_val_v : float, optional
            Maximum value of the toroidal coordinate.
        grad : str, optional
            Type of gradient to compute.

        Returns
        -------
        ndarray
            Current density on the surface.
        """
        # Unpack the grids
        ugrid, vgrid = grids  # u -> poloidal, v -> toroidal
        vgrid /= max_val_v

        # Extract the shapes
        lu, lv = ugrid.shape

        # Get the coefficients for the basis functions
        xm, xn = self._get_coeffs()
        xm = xm[:, None, None]
        xn = xn[:, None, None]

        # Initialize the list of current contributions
        dphi = []
        assert self.sin_basis or self.cos_basis

        # Compute the angle of the basis functions
        angle = 2 * onp.pi * (xm * ugrid + xn * vgrid)

        # Compute the current contributions for the sin basis functions
        if self.sin_basis:
            if grad == "u":
                dphi.append(2 * onp.pi * xm[..., None] * _stack(-xn *
                            onp.sin(angle) / max_val_v, xm * onp.sin(angle)))
            elif grad == "v":
                dphi.append(
                    2
                    * onp.pi
                    * xn[..., None]
                    / max_val_v
                    * _stack(-xn * onp.sin(angle) / max_val_v, xm * onp.sin(angle))
                )
            elif grad is None:
                dphi.append(_stack(xn * onp.cos(angle) /
                            max_val_v, -xm * onp.cos(angle)))
            else:
                raise NotImplementedError

        # Compute the current contributions for the cos basis functions
        if self.cos_basis:
            if grad == "u":
                dphi.append(2 * onp.pi * xm[..., None] * _stack(-xn *
                            onp.cos(angle) / max_val_v, xm * onp.cos(angle)))
            elif grad == "v":
                dphi.append(
                    2
                    * onp.pi
                    * xn[..., None]
                    / max_val_v
                    * _stack(-xn * onp.cos(angle) / max_val_v, xm * onp.cos(angle))
                )
            elif grad is None:
                dphi.append(_stack(-xn * onp.sin(angle) /
                            max_val_v, xm * onp.sin(angle)))
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
        Compute the gradient of the current density operator.

        Parameters
        ----------
        grids : array-like
            List of grids on which to compute the current density operator.
            The grids should be of shape (Nu, Nv).
        max_val_v : float, optional
            Maximum value of the toroidal coordinate.
            Default value is 1.

        Returns
        -------
        op : ndarray
            Array of shape (Ncurrent_op, Nu, Nv, N_j_surf, N_grad).
            The gradient of the current density operator.
            The current density operator is computed on the grids (u, v)
            and the values of the operator are stacked along the last
            dimension.
        """
        # Compute the gradient of the current density operator
        # along the u direction
        current_u = self(grids, max_val_v=max_val_v, grad="u")

        # Compute the gradient of the current density operator
        # along the v direction
        current_v = self(grids, max_val_v=max_val_v, grad="v")

        # Stack the current contributions along the last dimension
        return np.stack((current_u, current_v), axis=-1)

    def plot_phi_mn(self, phi_mn):
        """
        Plot the Fourier coefficients of the current density.

        Parameters
        ----------
        phi_mn : array-like
            Vector of Fourier coefficients of the current density.

        Returns
        -------
        None
        """

        # Compute the shape of the arrays
        shape = (self.num_pol, self.num_tor * 2 + 1)

        # Define the columns of the DataFrames
        col = pd.Index(
            onp.arange(-self.num_tor, (self.num_tor + 1)), name="toroidal")

        # Define the index of the DataFrames
        ind = pd.Index(onp.arange(1, self.num_pol + 1), name="poloidal")

        # Reshape the sin coefficients and create a DataFrame
        ph_sin = pd.DataFrame(
            onp.reshape(
                phi_mn[self.num_tor + 2: self.num_tor + 2 + onp.prod(shape)], shape),
            columns=col,
            index=ind,
        )

        # Reshape the cos coefficients and create a DataFrame
        ph_cos = pd.DataFrame(
            onp.reshape(phi_mn[self.num_tor * 2 +
                        2 + onp.prod(shape):], shape),
            columns=col,
            index=ind,
        )

        # Import matplotlib and seaborn
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Create a figure with two subplots
        f, axs = plt.subplots(2, 1, figsize=(8, 8))

        # Plot the sin coefficients heatmap
        sns.heatmap(ph_sin, cmap="seismic", center=0, ax=axs[0])
        axs[0].set_title("sin coefficients")

        # Plot the cos coefficients heatmap
        sns.heatmap(ph_cos, cmap="seismic", center=0, ax=axs[1])
        axs[1].set_title("cos coefficients")


class CurrentZeroTorBC(AbstractCurrent):
    """
    Current with periodic boundary condition along the poloidal axis and zero boundary
    conditions along the toroidal axis
    """

    def __init__(self, **kwargs):
        """
        Initialize the class.

        The number of dimensions is calculated based on the basis functions used.
        If both sin and cos basis functions are used, the number of dimensions is
        multiplied by 2.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments are passed to the parent class.
        """
        super().__init__(**kwargs)

        # Calculate the number of dimensions
        num_dims = 0

        # Calculate the number of dimensions for sin basis functions
        if self.sin_basis:
            num_dims += (self.num_pol + 1) * \
                (self.num_tor + 1) + self.num_tor + 1

        # Calculate the number of dimensions for cos basis functions
        if self.cos_basis:
            num_dims += (self.num_pol + 1) * (self.num_tor + 1)

        # Initialize phi_mn with zeros
        self.phi_mn = onp.zeros(num_dims)

    def get_phi(self, uv, phi_mn, max_val_v: float = 1.0):
        """
        Calculate the value of the current at a given point (uv) using the basis functions.

        Parameters
        ----------
        uv : ndarray
            The point (poloidal, toroidal) at which to evaluate the current.
        phi_mn : ndarray
            The coefficients of the basis functions.
        max_val_v : float, optional
            The maximum value of the toroidal coordinate. Default is 1.0.

        Returns
        -------
        float
            The value of the current at the given point.
        """
        # Get the coefficients of the basis functions
        xm, xn, xn0 = self._get_coeffs()

        # Normalize the toroidal coordinate
        v_ = uv[1] / max_val_v

        # Calculate the value of the current using the basis functions
        phi = phi_mn[0] * v_ - phi_mn[1] * uv[0]

        # Add the contributions from the sin basis functions
        if self.sin_basis:
            phi += np.sum(phi_mn[2: 2 + len(xn0)] * np.sin(onp.pi * xn0 * v_))
            phi += np.sum(
                phi_mn[2 + len(xn0): 2 + len(xn0) + len(xm)]
                * np.sin(2 * onp.pi * xm * uv[0])
                * np.sin(onp.pi * xn * v_)
            )

        # Add the contributions from the cos basis functions
        if self.cos_basis:
            phi += np.sum(phi_mn[2 + len(xn0) + len(xm):] *
                          np.cos(2 * onp.pi * xm * uv[0]) * np.sin(onp.pi * xn * v_))

        return phi

    def _get_coeffs(self):
        """
        Get the coefficients for the basis functions.

        Returns
        -------
        xm : ndarray
            Coefficients for the poloidal basis functions.
        xn : ndarray
            Coefficients for the toroidal basis functions.
        xn0 : ndarray
            Coefficients for the toroidal basis functions with n=0.
        """
        # Create a grid of poloidal and toroidal indices
        grid = onp.mgrid[1: (self.num_pol + 2),
                         1: (self.num_tor + 2)].reshape((2, -1))

        # Extract the coefficients for the basis functions
        xm = grid[0]  # Poloidal coefficients
        xn = grid[1]  # Toroidal coefficients
        # Toroidal coefficients with n=0
        xn0 = onp.arange(1, (self.num_tor + 2))

        return xm, xn, xn0

    def __call__(self, grids, max_val_v: float = 1, grad: tp.Optional[str] = None):
        """
        Compute the current density on the surface.

        Parameters
        ----------
        grids : tuple of ndarrays
            Grids of poloidal and toroidal coordinates.
        max_val_v : float, optional
            Maximum value of the toroidal coordinate.
        grad : str, optional
            Type of gradient to compute.

        Returns
        -------
        ndarray
            Current density on the surface.
        """
        # Unpack the grids
        ugrid, vgrid = grids  # u -> poloidal, v -> toroidal
        vgrid = vgrid / max_val_v
        # vgrid = vgrid + 0.5 / vgrid.shape[1]
        assert np.all(vgrid > 0)
        assert np.all(vgrid < 1)

        # Extract the shapes
        lu, lv = ugrid.shape

        # Get the coefficients for the basis functions
        xm, xn, xn0 = self._get_coeffs()
        xm = xm[:, None, None]
        xn = xn[:, None, None]
        xn0 = xn0[:, None, None]

        # Initialize the list of current contributions
        dphi = []
        assert self.sin_basis or self.cos_basis

        # Compute the cosine and sine of the basis functions
        cosu = onp.cos(2 * onp.pi * xm * ugrid)
        cosv = onp.cos(onp.pi * xn * vgrid)
        sinu = onp.sin(2 * onp.pi * xm * ugrid)
        sinv = onp.sin(onp.pi * xn * vgrid)

        # Compute the current contributions for the sin basis functions
        if self.sin_basis:
            zero_v0 = np.zeros_like(xn0 * vgrid)

            if grad == "u":
                dphi.append(_stack(zero_v0, zero_v0))
                dphi.append(
                    _stack(2 * onp.pi * xm * xn * cosu * cosv / 2 / max_val_v,
                           2 * onp.pi * xm**2 * sinu * sinv)
                )
            elif grad == "v":
                sinv0 = onp.sin(onp.pi * xn0 * vgrid)
                dphi.append(_stack(-onp.pi * xn0**2 *
                            sinv0 / max_val_v / 2, zero_v0))
                dphi.append(_stack(-onp.pi * xn**2 * sinu * sinv / 2 / max_val_v,
                                   -onp.pi * xn * xm * cosu * cosv))
            elif grad is None:
                cosv0 = onp.cos(onp.pi * xn0 * vgrid)
                dphi.append(_stack(xn0 * cosv0 / max_val_v /
                            2, onp.zeros_like(xn0 * cosv0)))
                dphi.append(_stack(xn * sinu * cosv / 2 /
                            max_val_v, -xm * cosu * sinv))

            else:
                raise NotImplementedError

        # Compute the current contributions for the cos basis functions
        if self.cos_basis:
            if grad == "u":
                dphi.append(_stack(-2 * onp.pi * xm * xn * sinu * cosv / 2 / max_val_v,
                                   2 * onp.pi * xm**2 * cosu * sinv))
            elif grad == "v":
                dphi.append(_stack(-onp.pi * xn**2 * cosu * sinv / 2 / max_val_v,
                                   onp.pi * xn * xm * sinu * cosv))
            elif grad is None:
                dphi.append(_stack(xn * cosu * cosv / 2 /
                            max_val_v, xm * sinu * sinv))
            else:
                raise NotImplementedError

        # Stack the current contributions
        dphi = onp.concatenate(dphi, axis=0)
        dphi = onp.concatenate((onp.zeros((2, lu, lv, 2)), dphi), axis=0)

        # Compute the gradient of the current density
        if grad is None:
            dphi[0, :, :, 0] = onp.ones((lu, lv)) / max_val_v
            dphi[1, :, :, 1] = onp.ones((lu, lv))

        return dphi

    def get_grad_current_op(self, grids, max_val_v: float = 1):
        """
        Compute the gradient of the current density operator.

        Parameters
        ----------
        grids : array-like
            List of grids on which to compute the current density operator.
            The grids should be of shape (Nu, Nv).
        max_val_v : float, optional
            Maximum value of the toroidal coordinate.
            Default value is 1.

        Returns
        -------
        op : ndarray
            Array of shape (Ncurrent_op, Nu, Nv, N_j_surf, N_grad).
            The gradient of the current density operator.
            The current density operator is computed on the grids (u, v)
            and the values of the operator are stacked along the last
            dimension.

        Notes
        -----
        The dimensions of the returned operator are:
            - Ncurrent_op : number of components of the current density operator.
            - Nu : number of poloidal grid points.
            - Nv : number of toroidal grid points.
            - N_j_surf : number of surface currents.
            - N_grad : number of components of the gradient operator.
        """
        # Compute the gradient of the current density operator
        # along the u direction
        current_u = self(grids, max_val_v=max_val_v, grad="u")

        # Compute the gradient of the current density operator
        # along the v direction
        current_v = self(grids, max_val_v=max_val_v, grad="v")

        # Stack the current contributions along the last dimension
        return np.stack((current_u, current_v), axis=-1)
