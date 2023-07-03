from jax.typing import ArrayLike

from stellacode import np
import typing as tp

from .abstract_surface import AbstractSurface
from .utils import cartesian_to_toroidal


class ToroidalSurface(AbstractSurface):
    Np: int
    outer_radius = 5.0
    inner_radius = 1.0

    def get_xyz(self, uv):
        u_ = 2 * np.pi * uv[0]  # poloidal angle
        v_ = 2 * np.pi * uv[1]  # toroidal angle
        x = (self.outer_radius + self.inner_radius * np.cos(u_)) * np.cos(v_)
        y = (self.outer_radius + self.inner_radius * np.cos(u_)) * np.sin(v_)
        z = self.inner_radius * np.sin(u_)
        return np.array([x, y, z])

    def cartesian_to_toroidal(self):
        return cartesian_to_toroidal(xyz=self.P, tore_radius=self.outer_radius, height=0.)
