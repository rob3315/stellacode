import typing as tp

from jax.typing import ArrayLike

from stellacode import np

from .abstract_surface import AbstractSurface
from .utils import fourier_transform


class CylindricalSurface(AbstractSurface):
    fourier_coeffs: ArrayLike = np.zeros((1, 2))
    axis_angle: float = 0.0  # rotates the surface by the given angle
    radius: float = 1.0  # radius of the cylinders
    scale_length: float = 1.0  # The cylinder is scaled by the scale_length factor
    distance: float = (
        3  # distance between the center of the cylinder and the coordinate center
    )
    make_joints: bool = True
    trainable_params: tp.List[str] = [
        "fourier_coeffs",
        "axis_angle",
        "radius",
        "scale_length",
        "distance",
    ]

    def get_xyz(self, uv):
        u_ = 2 * np.pi * uv[0]  # poloidal variable
        v_ = (uv[1] - 0.5) * 2  # length variable

        _length = (
            self.scale_length
            * 2
            * (self.distance - self.radius)
            * np.sin(np.pi / self.num_tor_symmetry)
        )

        axis_a = self.axis_angle
        axis_orth = np.array([np.sin(axis_a), -np.cos(axis_a), 0.0])

        z_dir = np.array([0.0, 0.0, 1.0])

        _radius = (fourier_transform(self.fourier_coeffs, u_) + 1) * self.radius


        circle = _radius * (axis_orth * np.cos(u_) + z_dir * np.sin(u_))

        if self.make_joints:
            p_dist = self.distance + np.cos(u_) * _radius
            _length = _length * p_dist / (self.distance - self.radius)

        cyl_axis = np.array(
            [np.cos(axis_a) * v_ * _length, np.sin(axis_a) * v_ * _length, 0.0]
        )

        return cyl_axis + circle + self.distance * axis_orth

    def fit_to_surface(self, surface):
        # Tries to find approximately the smallest surface enclosing the given surface
        # assuming the given surface has get_major_radius and get_minor_radius methods

        major_radius = surface.get_major_radius()
        minor_radius = surface.get_minor_radius()

        wrap_surf = self.copy(
            update=dict(
                radius=minor_radius + major_radius / 3,
                distance=major_radius,
            )
        )
        wrap_surf.compute_surface_attributes(deg=0)
        min_dist = wrap_surf.get_min_distance(surface.P)

        new_surf = wrap_surf.copy(
            update=dict(
                radius=minor_radius + major_radius / 3 - min_dist,
                distance=major_radius,
            )
        )
        new_surf.compute_surface_attributes()

        return new_surf
