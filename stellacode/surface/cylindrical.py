import typing as tp

from jax.typing import ArrayLike

from stellacode import np

from .abstract_surface import AbstractSurfaceFactory
from .utils import cartesian_to_toroidal, fourier_transform, cartesian_to_shifted_cylindrical, cartesian_to_cylindrical
import matplotlib.pyplot as plt


class CylindricalSurface(AbstractSurfaceFactory):
    fourier_coeffs: ArrayLike = np.zeros((1, 2))
    axis_angle: float = 0.0  # rotates the surface by the given angle
    radius: float = 1.0  # radius of the cylinders
    scale_length: float = 1.0  # The cylinder is scaled by the scale_length factor
    distance: float = 3  # distance between the center of the cylinder and the coordinate center
    make_joints: bool = True
    trainable_params: tp.List[str] = [
        "fourier_coeffs",
        "axis_angle",
        "radius",
        "distance",
    ]

    def get_xyz(self, uv):
        u_ = 2 * np.pi * uv[0]  # poloidal variable
        v_ = uv[1] - 0.5 + 0.5 / self.integration_par.num_points_v  # length variable
        # rotate along the cylinder base
        axis_orth, cyl_axis = self.get_axes()
        # axis_orth = np.array([np.sin(axis_a), -np.cos(axis_a), 0.0])
        z_dir = np.array([0.0, 0.0, 1.0])
        _radius = (fourier_transform(self.fourier_coeffs, u_) + 1) * self.radius
        circle = _radius * (axis_orth * np.cos(u_) + z_dir * np.sin(u_))

        # elongate along the cylinder height
        dist_edge = self.distance - self.radius
        _length = 2 * dist_edge * np.tan(np.pi / self.num_tor_symmetry)
        _length *= self.scale_length

        if self.make_joints:
            p_dist = self.distance + np.cos(u_) * _radius
            _length = _length * p_dist / (self.distance - self.radius)

        # shift along the cylinder
        return cyl_axis * v_ * _length + circle + self.distance * axis_orth

    def get_axes(self):
        axis_a = np.pi / 2 + np.pi / self.num_tor_symmetry + self.axis_angle
        axis_orth = np.array([np.sin(axis_a), -np.cos(axis_a), 0.0])
        cyl_axis = np.array([np.cos(axis_a), np.sin(axis_a), 0.0])
        return axis_orth, cyl_axis

    def to_cyl_frame_mat(self):
        axis_orth, cyl_axis = self.get_axes()
        return np.stack((np.cross(cyl_axis, axis_orth), axis_orth, cyl_axis), axis=1)

    def cartesian_to_toroidal(self):
        return cartesian_to_toroidal(xyz=self.xyz, tore_radius=self.distance, height=0.0)

    def to_cylindrical(self, xyz):
        rot_mat = self.to_cyl_frame_mat()
        xyz_in_frame = np.einsum("ab, ija->ijb", rot_mat, xyz - self.distance * rot_mat[:, 1])
        return cartesian_to_cylindrical(xyz_in_frame)

    def plot_cross_section(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        u_ = 2 * np.pi * np.linspace(0, 1, 100, endpoint=True)
        _radius = [(fourier_transform(self.fourier_coeffs, u_val) + 1) * self.radius for u_val in u_]
        ax.plot(u_, _radius, **kwargs)
        return ax
