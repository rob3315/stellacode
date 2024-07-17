import typing as tp

import matplotlib.pyplot as plt
from jax import nn
from jax.typing import ArrayLike

from stellacode import np

from .abstract_surface import AbstractSurfaceFactory
from .utils import cartesian_to_cylindrical, cartesian_to_toroidal, fourier_transform, fourier_transform_derivative, unwrap_u


class CylindricalSurface(AbstractSurfaceFactory):
    """
    Cylindrical surface

    Args:
        * ncp: number of periods
        * delta_angle: minimum angle between two cylinders
        * fourier_coeffs: fourier coefficents of the cross section
        * points: points defining the cross section
        * axis_angle: rotates the surface by the given angle along the toroidal axis
        * radius: mean radius of the cylinders
        * scale_length: The cylinder is scaled by the scale_length factor along the cylinder height
        * distance: distance between the center of the cylinder and the coordinate center
        * make_joints: The cylinder is cut at an angle such that all rotated cylinders are joined
        * trainable_params: list of trainable parameters
    """

    ncp: int
    delta_angle: float = 0.0
    fourier_coeffs: tp.Optional[ArrayLike] = np.zeros((1, 2))
    points: tp.Optional[ArrayLike] = None  # np.ones(10)
    axis_angle: float = 0.0  # rotates the surface by the given angle
    radius: float = 1.0  # radius of the cylinders
    scale_length: float = 1.0  # The cylinder is scaled by the scale_length factor
    distance: float = 3  # distance between the center of the cylinder and the coordinate center
    make_joints: bool = True
    shear_strength: float = 0.0
    trainable_params: tp.List[str] = [
        "fourier_coeffs",
        "axis_angle",
        "radius",
        "distance",
    ]

    def get_xyz(self, uv):
        """
        Calculate the cartesian coordinates (x, y, z) of a point on the surface given its cylindrical coordinates (u, v).

        Parameters
        ----------
        uv : ndarray
            Array of shape (N, 2) containing the cylindrical coordinates (u, v) of the points.

        Returns
        -------
        ndarray
            Array of shape (N, 3) containing the corresponding cartesian coordinates (x, y, z).
        """
        # Get the polar and length variables
        u_ = 2 * np.pi * uv[0]  # poloidal variable
        v_ = uv[1] - 0.5 + 0.5 / \
            self.integration_par.num_points_v  # length variable

        # Get the orthogonal and cylindrical axes
        axis_orth, cyl_axis = self._get_axes()

        # Set the direction of the z-axis
        z_dir = np.array([0.0, 0.0, 1.0])

        # Check if fourier coefficients or points are defined
        assert self.fourier_coeffs is not None or self.points is not None

        # Calculate the radius at each angle phi (u_)
        if self.points is not None:
            _radius = (
                np.interp(
                    u_, np.linspace(0, 1, len(self.points), endpoint=False), nn.softmax(
                        self.points) * len(self.points)
                )
                * self.radius
            )
            _inner_radius = (
                np.interp(
                    np.pi, np.linspace(0, 1, len(self.points), endpoint=False), nn.softmax(
                        self.points) * len(self.points)
                )
                * self.radius
            )
        else:
            _radius = (fourier_transform(
                self.fourier_coeffs, u_)+1) * self.radius
            _inner_radius = (fourier_transform(
                self.fourier_coeffs, np.pi)+1) * self.radius

        # Cross section at each angle phi (u_)
        circle = _radius * (axis_orth * np.cos(u_) + z_dir * np.sin(u_))

        # Calculate the cylinder height
        if self.make_joints:
            dist = self.distance + np.cos(u_) * _radius
        else:
            dist = self.distance - _inner_radius

        _length = dist * 2 * np.tan(np.pi / self.ncp)
        _length *= self.scale_length

        # Apply shear effect if strength is not zero
        if self.shear_strength != 0.0:
            axis_a = self._get_axis_angle()
            shear = (
                self.shear_strength
                * v_
                * np.array([np.sin(axis_a + 2 * np.pi * v_), -np.cos(axis_a + 2 * np.pi * v_), 0.0])
            )
        else:
            shear = 0.0

        # Shift along the cylinder and return the cartesian coordinates
        return cyl_axis * v_ * _length + circle + self.distance * axis_orth + shear

    def get_uv_unwrapped(self, uv):
        u_ = 2 * np.pi * uv[0]  # poloidal variable (phi angle unwrapped)
        v_ = uv[1] - 0.5 + 0.5 / \
            self.integration_par.num_points_v  # length variable

        # Calculate the radius at angle phi (u_)
        if self.points is not None:
            radius = (
                np.interp(
                    u_, np.linspace(0, 1, len(self.points), endpoint=False), nn.softmax(
                        self.points) * len(self.points)
                )
                * self.radius
            )
        else:
            radius = (fourier_transform(
                self.fourier_coeffs, u_) + 1) * self.radius

        # Compute unwrapped u(phi)
        if np.sum(np.diff(radius.val)) == 0:
            # If the radius is NOT defined by a fourier series
            unwrapped_u = u_ * radius
        else:
            # If the radius is defined by a fourier series
            unwrapped_u = unwrap_u(lambda phi: (fourier_transform(self.fourier_coeffs, phi) + 1) * self.radius,
                                   lambda phi: (fourier_transform_derivative(self.fourier_coeffs, phi)) * self.radius, u_)

        # Calculate the axial length
        dist_edge = self.distance - self.radius
        length = 2 * dist_edge * np.tan(np.pi / self.ncp)
        length *= self.scale_length

        # Adjust length if make_joints is True
        if self.make_joints:
            p_dist = self.distance + np.cos(u_) * radius
            length = length * p_dist / (self.distance - self.radius)

        # Shear effect if applied
        if self.shear_strength != 0.0:
            axis_a = self._get_axis_angle()
            shear_x = self.shear_strength * v_ * \
                np.sin(axis_a + 2 * np.pi * v_)
            shear_y = self.shear_strength * v_ * - \
                np.cos(axis_a + 2 * np.pi * v_)
        else:
            shear_x = 0.0
            shear_y = 0.0

        # Since we are creating a 2D unwrapped surface, the output should be 2D points
        # u_ maps the circumferential position, while v_ maps the axial position
        # return np.array([u_ * radius,v_ * length]) + np.array([shear_x, shear_y])
        return np.array([unwrapped_u, v_ * length]) + np.array([shear_x, shear_y])

    def _get_axis_angle(self):
        """
        Compute the angle of the cylindrical axis.

        Returns
        -------
        float
            The angle of the cylindrical axis in radians.

        Notes
        -----
        This function computes the angle of the cylindrical axis by combining the
        initial angle of the axis, the angle per cylinder, and the specific angle.
        The angle of the axis is defined as the angle between the cylindrical axis and
        the x-axis in the cartesian coordinate system.
        """
        return np.pi / 2 + np.pi / self.ncp + self.axis_angle

    def _get_axes(self):
        """
        Compute the cylindrical and orthogonal axes of the cylindrical coordinate system.

        Returns
        -------
        tuple
            The cylindrical and orthogonal axes of the cylindrical coordinate system.
            The cylindrical axis is represented by cyl_axis.
            The orthogonal axis is represented by axis_orth.
        """
        axis_a = self._get_axis_angle()
        axis_orth = np.array([np.sin(axis_a), -np.cos(axis_a), 0.0])
        cyl_axis = np.array([np.cos(axis_a), np.sin(axis_a), 0.0])
        return axis_orth, cyl_axis

    def _to_cyl_frame_mat(self):
        """
        Compute the matrix of the cylindrical coordinate system.

        Returns
        -------
        ndarray
            The matrix of the cylindrical coordinate system.
            Each row represents a basis vector.
            The first column is the cross product of the cylindrical axis and the orthogonal axis.
            The second column is the orthogonal axis.
            The third column is the cylindrical axis.

        Notes
        -----
        This function computes the matrix of the cylindrical coordinate system.
        It first computes the cylindrical and orthogonal axes using the _get_axes method.
        Then it computes the cross product of the cylindrical axis and the orthogonal axis.
        The resulting matrix is a 3D matrix, where each row represents a basis vector.
        The first column is the cross product of the cylindrical axis and the orthogonal axis.
        The second column is the orthogonal axis.
        The third column is the cylindrical axis.
        """
        # Compute the cylindrical and orthogonal axes
        axis_orth, cyl_axis = self._get_axes()

        # Compute the cross product of the cylindrical axis and the orthogonal axis
        cross_product = np.cross(cyl_axis, axis_orth)

        # Stack the basis vectors to form the matrix of the cylindrical coordinate system
        return np.stack((cross_product, axis_orth, cyl_axis), axis=1)

    def cartesian_to_toroidal(self):
        return cartesian_to_toroidal(xyz=self.xyz, tore_radius=self.distance, height=0.0)

    def to_cylindrical(self, xyz):
        """
        Transform 3D Cartesian coordinates to cylindrical coordinates.

        Parameters
        ----------
        xyz : ndarray
            The 3D Cartesian coordinates.

        Returns
        -------
        ndarray
            The cylindrical coordinates.
        """
        # Rotate the coordinates to the cylindrical frame
        # by subtracting the distance along the cylindrical axis times the second
        # column of the rotation matrix
        rot_mat = self._to_cyl_frame_mat()
        xyz_in_frame = np.einsum(
            "ab, ija->ijb", rot_mat, xyz - self.distance * rot_mat[:, 1])

        # Convert the coordinates to cylindrical using the
        # cartesian_to_cylindrical function
        return cartesian_to_cylindrical(xyz_in_frame)

    def plot_cross_section(self, ax=None, num: int = 1, **kwargs):
        """
        Plot cross sections of the cylindrical surface.

        Args:
            ax (Axes, optional): Matplotlib axes object to plot on.
            num (int, optional): Number of cross-sections to plot.
            **kwargs: Additional arguments to pass to the Matplotlib plot function.

        Returns:
            Axes: Matplotlib axes object.
        """
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        u_ = 2 * np.pi * np.linspace(0, 1, 100, endpoint=True)
        _radius = [(fourier_transform(self.fourier_coeffs, u_val) + 1)
                   * self.radius for u_val in u_]
        ax.plot(u_, _radius, **kwargs)
        return ax

    def plot_cross_sections(
        self,
        num_cyl: tp.Optional[int] = None,
        num: int = 5,
        convex_envelope: bool = False,
        concave_envelope: bool = False,
        scale_envelope: float = 1.0,
        ax=None,
    ):
        """
        Plot cross sections of the cylindrical surface.

        Args:
            ax (Axes, optional): Matplotlib axes object to plot on.
            num (int, optional): Number of cross-sections to plot.
            **kwargs: Additional arguments to pass to the Matplotlib plot function.

        Returns:
            Axes: Matplotlib axes object.
        """
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

        # Define u and v coordinates
        u = np.linspace(0, 1, 100, endpoint=True)
        v = np.linspace(0, 1, num, endpoint=True)
        ugrid, vgrid = np.meshgrid(u, v, indexing="ij")

        # Create surface instance and calculate xyz coordinates
        surf = self()
        xyz = self.get_xyz_on_grid(np.stack((ugrid, vgrid)))

        # Calculate radial and theta coordinates
        rtheta = surf._get_rtheta(xyz=xyz, num_cyl=num_cyl)

        # Plot cross-sections
        for i in range(num):
            rphi = np.concatenate((rtheta[:, i, :], rtheta[:, i, :]), axis=0)
            ax.plot(
                rphi[:, 1],
                rphi[:, 0],
                c=[0, 0] + [float((i + 1) / (num + 1))],
            )

        return ax.get_figure(), ax


class VerticalCylinder(AbstractSurfaceFactory):
    """
    Vertical cylindrical surface

    Args:
        * radius: radius of the cylinder
        * height: height of the cylinder

    """

    radius: float = 1.0  # radius of the cylinder
    height: float = 1.0  # height of the cylinder
    trainable_params: tp.List[str] = [
        "radius",
        "height",
    ]

    def get_xyz(self, uv):
        u_ = 2 * np.pi * uv[0]  # poloidal variable
        v_ = uv[1] + 0.5 / self.integration_par.num_points_v  # length variable

        return np.array([np.cos(u_) * self.radius, np.sin(u_) * self.radius, v_ * self.height])
