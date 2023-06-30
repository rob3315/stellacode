from stellacode import np

from .abstract_surface import AbstractSurface
from jax.typing import ArrayLike


class CylindricalSurface(AbstractSurface):
    Np: int
    alpha: float = 0.0
    beta: float = 1.0
    radius_scale: float = 1.0

    def get_xyz(self, uv):
        fourier_coeffs = self.params["fourier_coeffs"]
        fourier_orders = len(fourier_coeffs) // 2
        fourier_num = np.arange(fourier_orders)
        th = 2 * np.pi * uv[0]
        phi = 2 * np.pi / self.Np * uv[1]

        radius = np.tensordot(
            fourier_coeffs,
            np.concatenate((np.cos(th * fourier_num), np.sin(th * fourier_num))),
        )

        x_pos = (
            np.cos(phi)
            / np.sin(phi + self.alpha)
            * (self.radius_scale * np.sin(self.alpha) + np.cos(th) * radius)
        )

        y_pos = np.tan(phi) * x_pos

        z_pos = np.tan(self.beta) * y_pos + np.sin(th) * radius / np.cos(self.beta)
        return np.array((x_pos, y_pos, z_pos))


class CylindricalSurface2(AbstractSurface):
    Np: int

    def get_xyz(self, uv):
        u, v = uv
        alpha = 0.0
        radius = 1.0

        cyl_axis = np.array([np.cos(alpha) * u, np.sin(alpha) * u, 0.0])

        axis_orth = np.array([np.cos(alpha), np.sin(alpha), 0.0])
        z_dir = np.array([0.0, 0.0, 1.0])

        circle = radius * (
            axis_orth * np.cos(2 * np.pi * v) + z_dir * np.sin(2 * np.pi * v)
        )

        return cyl_axis + circle
