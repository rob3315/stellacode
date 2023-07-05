import typing as tp

from jax.typing import ArrayLike

from stellacode import np

from .abstract_surface import AbstractSurface
from .utils import cartesian_to_toroidal, fourier_transform


class ToroidalSurface(AbstractSurface):
    major_radius = 5.0
    minor_radius = 1.0
    fourier_coeffs: ArrayLike = np.zeros((1, 2))

    trainable_params: tp.List[str] = ["major_radius", "minor_radius", "fourier_coeffs"]

    def get_xyz(self, uv):
        u_ = 2 * np.pi * uv[0]  # poloidal angle
        v_ = 2 * np.pi * uv[1]  # toroidal angle

        minor_radius = (fourier_transform(self.fourier_coeffs, u_) + 1) * self.minor_radius

        x = (self.major_radius + minor_radius * np.cos(u_)) * np.cos(v_)
        y = (self.major_radius + minor_radius * np.cos(u_)) * np.sin(v_)
        z = minor_radius * np.sin(u_)
        return np.array([x, y, z])

    def cartesian_to_toroidal(self):
        return cartesian_to_toroidal(xyz=self.P, tore_radius=self.major_radius, height=0.0)
