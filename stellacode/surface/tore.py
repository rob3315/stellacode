import typing as tp

from jax.typing import ArrayLike

from stellacode import np

from .abstract_surface import AbstractSurface
from .utils import cartesian_to_toroidal


class ToroidalSurface(AbstractSurface):
    major_radius = 5.0
    minor_radius = 1.0

    trainable_params: tp.List[str] = [
        "major_radius",
        "minor_radius",
    ]

    def get_xyz(self, uv):
        u_ = 2 * np.pi * uv[0]  # poloidal angle
        v_ = 2 * np.pi * uv[1]  # toroidal angle
        x = (self.major_radius + self.minor_radius * np.cos(u_)) * np.cos(v_)
        y = (self.major_radius + self.minor_radius * np.cos(u_)) * np.sin(v_)
        z = self.minor_radius * np.sin(u_)
        return np.array([x, y, z])

    def cartesian_to_toroidal(self):
        return cartesian_to_toroidal(xyz=self.P, tore_radius=self.major_radius, height=0.)
