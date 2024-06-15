import typing as tp

import numpy as np
from pydantic import BaseModel
from scipy.constants import mu_0
from scipy.io import netcdf_file

from .coordinates import cylindrical_to_cartesian, vmec_to_cylindrical
from .utils import cfourier1D, sfourier1D

from functools import cache, cached_property


def _dot(coeff, cossin):
    """
    Compute the dot product of `coeff` and `cossin` along the first two dimensions.

    Parameters:
        coeff (np.ndarray): Array of coefficients with shape (m, n, ...).
        cossin (np.ndarray): Array of cosine and sine values with shape (m, n, ..., 2).

    Returns:
        np.ndarray: Array of dot products with shape (m, n, ...).
    """
    # Compute the dot product of `coeff` and `cossin` along the first two dimensions
    # using the einsum function from numpy.
    return np.einsum("rm,tzm->rtz", coeff, cossin)


class Grid(BaseModel):
    theta: np.ndarray
    zeta: np.ndarray
    surf_labels: tp.Optional[np.ndarray] = None

    model_config = dict(
        arbitrary_types_allowed=True,
        frozen=True,
    )  # required for caching


class VMECIO:
    """
    Class for reading VMEC output files.

    Attributes:
        file (netcdf_file): The opened VMEC output file.
        grid (Grid): The grid on which the VMEC data is defined.
    """

    def __init__(self, vmec_file: str, grid: Grid):
        """
        Initialize the VMECIO class.

        Parameters:
            vmec_file (str): Path to the VMEC output file.
            grid (Grid): The grid on which the VMEC data is defined.
        """
        # Open the VMEC output file
        self.file = netcdf_file(vmec_file)

        # Store the grid
        self.grid = grid

    def get_var(self, key: str, cast_type=float):
        """
        Get a variable from the VMEC output file.

        Parameters:
            key (str): The name of the variable.
            cast_type (type, optional): The type to cast the variable to.
                Defaults to float.

        Returns:
            np.ndarray: The variable cast to the specified type.
        """
        # Get the variable from the VMEC output file
        var = self.file.variables[key][()]

        # Cast the variable to the specified type and return it
        return var.astype(cast_type)

    @cache
    def get_val(
        self,
        key: str,
        diff: tp.Optional[str] = None,
        nyq: bool = True,
    ) -> float:
        """
        Reconstruct a variable from a fourier coefficients in the VMEC output file.

        Parameters:
            key (str): The name of the variable.
            diff (str, optional):
                Differentiation direction. Can be "theta" or "zeta".
            nyq (bool, optional): Whether to use the Nyquist frequency data.
                Defaults to True.

        Returns:
            float: The value of the variable.
        """
        # Determine the suffix for the variable names
        suffix = "_nyq" if nyq else ""

        # Get the poloidal mode numbers
        xm = self.get_var(f"xm{suffix}", int)
        # Get the toroidal mode numbers
        xn = self.get_var(f"xn{suffix}", int)

        # Calculate the angle based on the grid and xm, xn
        angle = self.grid.theta[..., None] * \
            xm - self.grid.zeta[..., None] * xn

        # Initialize the value
        val = 0.0

        # Check if fourier coeffs of the cosine expansion are present in the VMEC output file
        if f"{key}mnc" in self.file.variables.keys():
            mnc = self.get_var(f"{key}mnc", float)

            # Apply surface labels if present
            if self.grid.surf_labels is not None:
                mnc = mnc[self.grid.surf_labels]

            # Reconstruct the fourier sum
            # Differentiate with respect to theta
            if diff == "theta":
                val -= _dot(mnc * xm[None, :], np.sin(angle))
            # Differentiate with respect to zeta
            elif diff == "zeta":
                val += _dot(mnc * xn[None, :], np.sin(angle))
            # No differentiation
            else:
                val += _dot(mnc, np.cos(angle))

        # Check if fourier coeffs of the sine expansion are present in the VMEC output file
        if f"{key}mns" in self.file.variables.keys():
            mns = self.get_var(f"{key}mns", float)

            # Apply surface labels if present
            if self.grid.surf_labels is not None:
                mns = mns[self.grid.surf_labels]

            # Reconstruct the fourier sum
            # Differentiate with respect to theta
            if diff == "theta":
                val += _dot(mns * xm[None, :], np.cos(angle))
            # Differentiate with respect to zeta
            elif diff == "zeta":
                val -= _dot(mns * xn[None, :], np.cos(angle))
            # No differentiation
            else:
                val += _dot(mns, np.sin(angle))

        return val

    @cache
    def get_val_grad(
        self,
        key: str,
        nyq: bool = True,
    ) -> np.ndarray:
        """
        Compute the gradient of a VMECIO variable.

        Parameters
        ----------
        key : str
            The name of the VMECIO variable.
        nyq : bool, optional
            If True, computes the value on the Nyquist grid.
            If False, computes the value on the original grid.
            Default is True.

        Returns
        -------
        np.ndarray
            The gradient of the VMECIO variable.
            The shape of the array is (Nu, Nv, 2).

        """
        # Set the arguments for the get_val function
        kwargs = dict(key=key, nyq=nyq)

        # Compute the gradient with respect to theta and zeta
        return np.stack(
            (
                # Gradient with respect to theta
                self.get_val(**kwargs, diff="theta"),
                # Gradient with respect to zeta
                self.get_val(**kwargs, diff="zeta"),
            ),
            axis=-1,
        )

    @cache
    def get_b_vmec(
        self,
        covariant: bool = True,
    ) -> np.ndarray:
        """
        Compute the covariant or contravariant magnetic field.

        Parameters
        ----------
        covariant : bool, optional
            If True, computes the covariant magnetic field.
            If False, computes the contravariant magnetic field.
            Default is True.

        Returns
        -------
        np.ndarray
            The magnetic field.
            The shape of the array is (Nsurf, Nu, Nv, 3).

        """
        # Set the prefix for the magnetic field variable
        if covariant:
            prefix = "bsub"
            bs = self.get_val(f"bsubs")
        else:
            prefix = "bsup"
            lt, lz = self.grid.theta.shape
            bs = np.zeros((self.grid.surf_labels.shape[0], lt, lz))

        # Stack the magnetic field components
        return np.stack(
            (
                self.get_val(f"{prefix}u"),  # Bx
                self.get_val(f"{prefix}v"),  # By
                bs,  # Bz
            ),
            axis=-1,
        )

    def get_magnetic_axis(self, nzeta: int = 100):
        """
        Compute the magnetic axis using Fourier series expansion.

        Parameters
        ----------
        nzeta : int, optional
            Number of Fourier modes to use for the expansion.
            Default is 100.

        Returns
        -------
        tuple
            The radial and zonal Fourier coefficients of the magnetic axis.
            The shape of the arrays is (nzeta,).
        """
        # Retrieve the radial and zonal Fourier coefficients of the magnetic axis
        raxis_cc = self.get_var("raxis_cc", float)
        zaxis_cs = self.get_var("zaxis_cs", float)
        xn = self.get_var("xn", int)

        # Compute the magnetic axis using Fourier series expansion
        Raxis = cfourier1D(nzeta, raxis_cc, xn)  # Radial component
        Zaxis = sfourier1D(nzeta, zaxis_cs, xn)  # Zonal component

        return Raxis, Zaxis

    @cached_property
    def rphiz(self):
        """
        Compute the cylindrical coordinates of the grid points.

        Returns
        -------
        ndarray
            Array of shape (Nt, Np, 3) where each element is an array (r, phi, z).
        """
        # Retrieve the radius and z coordinates of the grid points
        radius = self.get_val("r", nyq=False)
        z_ = self.get_val("z", nyq=False)

        # Create the array of phi values
        # This is done by tiling the zeta values along the second axis
        phi = np.tile(self.grid.zeta, (radius.shape[0], 1, 1))

        # Stack the r, phi, and z coordinates along the last axis
        return np.stack((radius, phi, z_), axis=-1)

    @cached_property
    def xyz(self):
        """
        Compute the cartesian coordinates of the grid points.

        Returns
        -------
        ndarray
            Array of shape (Nt, Np, 3) where each element is an array (x, y, z).

        """
        # Retrieve the cylindrical coordinates of the grid points
        rphiz = self.rphiz

        # Convert the cylindrical coordinates to cartesian coordinates
        # using the zeta values along the third axis
        return cylindrical_to_cartesian(rphiz, self.grid.zeta)

    @cached_property
    def grad_rphiz(self):
        """
        Compute the gradient of the cylindrical coordinates.

        Returns
        -------
        ndarray
            Array of shape (Nt, Np, N_coords, 3, 3) where each element is a
            3x3 matrix representing the gradient of r, phi, and z.
        """
        # Retrieve the gradients of the radius and z coordinates
        r_grad = self.get_val_grad("r", nyq=False)
        z_grad = self.get_val_grad("z", nyq=False)

        # Create the array of zeros with the same shape as r_grad
        zeros_array = np.zeros_like(r_grad)

        # Stack the gradients of r, phi, and z along the last axis
        grad_rpz = np.stack((r_grad, zeros_array, z_grad), axis=-1)

        # Transpose the array to move the last two dimensions to the front
        return np.transpose(grad_rpz, (0, 1, 2, 4, 3))

    @cached_property
    def grad_xyz(self):
        """
        Compute the gradient of the cartesian coordinates.

        Returns
        -------
        ndarray
            Array of shape (Nt, Np, N_coords, 3) where each element is an array
            (dx, dy, dz) representing the gradient of x, y, and z.
        """
        # Retrieve the gradient of the cylindrical coordinates
        grad_rphiz = self.grad_rphiz

        # Reshape the cylindrical coordinates to have the same shape as the
        # gradient along the last axis
        rphiz = self.rphiz[..., None]

        # Extract the gradient components
        gr, gz = grad_rphiz[..., 0, :], grad_rphiz[..., 2, :]

        # Extract the cylindrical coordinates
        r_, phi = rphiz[..., 0, :], rphiz[..., 1, :]

        # Compute the gradient of the cartesian coordinates using the chain
        # rule
        return np.stack(
            (
                gr * np.cos(phi) - r_ * np.sin(phi),  # dx/dr
                gr * np.sin(phi) + r_ * np.cos(phi),  # dy/dr
                gz,                                  # dz/dr
            ),
            axis=-2,
        )

    def vmec_to_cylindrical(self, vector_field):
        """
        Apply the transformation from vmec coordinates to cylindrical coordinates

        Parameters
        ----------
        vector_field: ndarray
            The input vector field of shape (Nt, Np, N_coords, 3)

        Returns
        -------
        ndarray
            The transformed vector field of shape (Nt, Np, N_coords, 3)
        """

        # Retrieve the gradient of the cylindrical coordinates
        grad_rphiz = self.grad_rphiz

        # Retrieve the cylindrical coordinates
        rphiz = self.rphiz

        return vmec_to_cylindrical(vector_field, rphiz, grad_rphiz)

    @cached_property
    def b_cylindrical(
        self,
    ):
        """
        Compute the magnetic field in cylindrical coordinates.

        Returns
        -------
        ndarray
            The magnetic field of shape (Nt, Np, N_coords, 3)

        Notes
        -----
        This function computes the magnetic field in cylindrical coordinates
        using the vmec_to_cylindrical transformation.
        """

        # Compute the magnetic field in vmec coordinates
        b_vmec = self.get_b_vmec(covariant=False)

        # Transform the magnetic field to cylindrical coordinates
        return self.vmec_to_cylindrical(b_vmec)

    @cached_property
    def b_cartesian(
        self,
    ):
        """
        Compute the magnetic field in cartesian coordinates.

        Returns
        -------
        ndarray
            The magnetic field of shape (Nt, Np, N_coords, 3)

        Notes
        -----
        This function computes the magnetic field in cartesian coordinates
        using the cylindrical_to_cartesian transformation.
        """

        # Compute the magnetic field in cylindrical coordinates
        b_cylindrical = self.b_cylindrical

        # Transform the magnetic field to cartesian coordinates
        return cylindrical_to_cartesian(b_cylindrical, self.grid.zeta)

    @cached_property
    def j_vmec(
        self,
    ):
        """
        Compute the vmec current in cylindrical coordinates.

        Returns
        -------
        ndarray
            The vmec current of shape (Nt, Np, N_coords, 3)

        Notes
        -----
        This function computes the current in vmec coordinates by stacking the
        vmec current in the radial direction (curru) and in the toroidal
        direction (currv) along the last axis.
        """

        # Get the vmec current in the radial direction (curru)
        ju = self.get_val(f"curru")

        # Get the vmec current in the toroidal direction (currv)
        jv = self.get_val(f"currv")

        # Stack the vmec current in cylindrical coordinates
        return np.stack((ju, jv, np.zeros_like(ju)), axis=-1)

    @cached_property
    def j_cylindrical(
        self,
    ):
        """
        Compute the vmec current in cylindrical coordinates.

        Returns
        -------
        ndarray
            The vmec current of shape (Nt, Np, N_coords, 3)

        Notes
        -----
        This function computes the current in vmec coordinates.
        It then applies the vmec_to_cylindrical transformation
        to map the current to cylindrical coordinates.
        """

        # Compute the vmec current in vmec coordinates
        j_vmec = self.j_vmec

        # Apply the vmec_to_cylindrical transformation to map the current to
        # cylindrical coordinates
        return self.vmec_to_cylindrical(j_vmec)

    @cached_property
    def j_cartesian(
        self,
    ):
        """
        Compute the current in cartesian coordinates.

        Returns
        -------
        ndarray
            The current of shape (Nt, Np, N_coords, 3)

        Notes
        -----
        This function computes the current in vmec coordinates.
        It then applies the cylindrical_to_cartesian transformation
        to map the current to cartesian coordinates using the
        zeta values along the last axis.
        """

        # Compute the current in vmec coordinates
        j_cylindrical = self.j_cylindrical

        # Transform the current to cartesian coordinates using
        # the zeta values along the last axis
        return cylindrical_to_cartesian(j_cylindrical, self.grid.zeta)

    @cached_property
    def nfp(self):
        """
        Retrieve the number of field periods from the VMECIO object.

        Returns
        -------
        int
            The number of field periods.

        Notes
        -----
        This function uses the `get_var` method to retrieve the value of the
        'nfp' variable from the VMECIO object.

        """
        # Get the number of field periods from the VMECIO object.
        # The 'nfp' variable is expected to be an integer.
        return self.get_var("nfp", int)

    @cached_property
    def curpol(self):
        """
        Calculate the current per field period (curpol).

        Returns
        -------
        float
            The current per field period.

        Notes
        -----
        The current per field period is calculated as:
        curpol = 2 * pi / nfp * bsubv(m=0, n=0)
        where bsubv is the extrapolation to the last full mesh point of
        bsubvmnc.
        """
        # Retrieve the bsubvmnc variable from the VMECIO object.
        # Transpose the array to have shape (n toroidal, n poloidal)
        bsubvmnc = self.get_var("bsubvmnc", float).T

        # Calculate bsubv(m=0, n=0) by extrapolating from the last full mesh point
        bsubv00 = 1.5 * bsubvmnc[0, -1] - 0.5 * bsubvmnc[0, -2]

        # Calculate the current per field period
        curpol = 2 * np.pi / self.nfp * bsubv00

        return curpol

    def scale_bnorm(self, b_norm, factor: tp.Optional[float] = 1.0):
        """
        Scale the normalized magnetic field by undoing the scaling applied
        by BNORM.

        Parameters
        ----------
        b_norm : float or array-like
            The normalized magnetic field.
        factor : float, optional
            A scaling factor to apply to the scaled magnetic field.

        Returns
        -------
        float or array-like
            The scaled magnetic field.

        Notes
        -----
        BNORM scales B_n by curpol=(2*pi/nfp)*bsubv(m=0,n=0)
        where bsubv is the extrapolation to the last full mesh point of
        bsubvmnc. This function undoes this scaling by multiplying the
        normalized magnetic field by curpol.
        """
        # Undo the scaling applied by BNORM
        return b_norm * self.curpol * factor

    @cached_property
    def net_poloidal_current(self):
        """
        Calculate the net poloidal current.

        This function calculates the net poloidal current by extrapolating the
        covariant components of B (bvco) to the last full mesh point. It then
        integrates the magnetic field along the toroidal surface using the
        integral_zeta function.

        Returns
        -------
        float
            The net poloidal current.

        Notes
        -----
        In principle, the net poloidal current should be calculated as:
        2pi/mu_0*integral_zeta(B(theta=interior, radius=last_radius))
        However, the integral_zeta function is not available in the VMEC data.
        Therefore, we use a simplified estimate based on the last two points of
        the bvco array.
        """

        # Retrieve the bvco variable from the VMECIO object.
        # Transpose the array to have shape (n toroidal, n poloidal)
        bvco = self.get_var("bvco", float).T

        # Calculate the net poloidal current by extrapolating the covariant
        # components of B to the last full mesh point.
        # (1.5 * bvco[-1] - 0.5 * bvco[-2]) represents the integral of B
        # along the toroidal surface.
        return 2 * np.pi / mu_0 * (1.5 * bvco[-1] - 0.5 * bvco[-2])

    @cached_property
    def net_poloidal_current2(self):
        """
        Calculate the net poloidal current using the last full mesh point.

        This function calculates the net poloidal current by extrapolating the
        covariant components of B (bvco) to the last full mesh point. It then
        integrates the magnetic field along the toroidal surface using the
        integral_zeta function.

        Returns
        -------
        float
            The net poloidal current.

        Notes
        -----
        This function undoes the scaling applied by BNORM and calculates the
        net poloidal current based on the last full mesh point.
        """

        # Calculate the covariant components of B at the last full mesh point.
        # Transpose the array to have shape (n theta, n zeta)
        bvco = self.b_cartesian[-1].T

        # Calculate the gradient of theta and zeta.
        # The shape of the array is (n theta, n zeta, 2)
        xyz_dv = self.grad_xyz[-1, :, :, :, 1] * (2 * np.pi)

        # Multiply the covariant components of B with the gradient of theta.
        # The shape of the array is (n theta, n zeta)
        b_theta = np.sum((bvco * xyz_dv)[:, :, 0], axis=-1)

        # Calculate the net poloidal current by integrating the magnetic field
        # along the toroidal surface.
        # The shape of the array is (n poloidal)
        net_current = np.mean(np.sum(b_theta, axis=0), axis=0)

        # Undo the scaling applied by BNORM
        return net_current / mu_0

    @classmethod
    def from_grid(
        cls,
        file_path,
        ntheta: int = 10,
        nzeta: int = 10,
        n_surf_label: tp.Optional[int] = None,
        surface_label: tp.Optional[int] = None,
    ):
        """
        Create a VMECIO object from a grid defined by ntheta and nzeta.

        Parameters
        ----------
        file_path : str
            Path to the VMEC .nc file.
        ntheta : int, optional
            Number of theta points in the grid. Default is 10.
        nzeta : int, optional
            Number of zeta points in the grid. Default is 10.
        n_surf_label : int, optional
            Number of surface labels to load. If None, loads all surfaces.
            Default is None.
        surface_label : int, optional
            Surface label to load. If None, loads all surfaces. Default is None.

        Returns
        -------
        VMECIO
            VMECIO object with the grid defined by ntheta and nzeta.
        """

        # Open the netCDF file
        file_ = netcdf_file(file_path)

        # Create the theta and zeta arrays
        theta_ = np.linspace(0, 2 * np.pi, num=ntheta)
        zeta_ = np.linspace(0, 2 * np.pi, num=nzeta)
        zeta, theta = np.meshgrid(zeta_, theta_)

        # Get the number of surfaces
        ns = file_.variables["ns"][()].astype(int)

        # Determine the surface labels to load
        if surface_label is None:
            if n_surf_label is None:
                n_surf_label = ns
            surf_labels = np.linspace(
                0, ns - 1, num=n_surf_label, endpoint=True).round().astype(int)
        else:
            surf_labels = np.array([surface_label])

        # Create and return the VMECIO object
        return cls(file_path, grid=Grid(theta=theta, zeta=zeta, surf_labels=surf_labels))
