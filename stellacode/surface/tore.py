import typing as tp

from jax.typing import ArrayLike
from jax import Array

from stellacode import np

from .abstract_surface import AbstractSurfaceFactory, IntegrationParams
from .utils import cartesian_to_toroidal, fourier_transform


class ToroidalSurface(AbstractSurfaceFactory):
    major_radius: float
    minor_radius: float
    fourier_coeffs: Array
    axis_angle: float
    integration_par: IntegrationParams
    num_tor_symmetry: int

    def __init__(
        self,
        integration_par: IntegrationParams,
        num_tor_symmetry: int,
        major_radius: float = 5.0,
        minor_radius: float = 1.0,
        fourier_coeffs: ArrayLike = np.zeros((4, 2)),
        axis_angle: float = 0.0,
    ):
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.fourier_coeffs = fourier_coeffs
        self.axis_angle = axis_angle
        self.integration_par = integration_par
        self.num_tor_symmetry = num_tor_symmetry

    def get_xyz(self, uv):
        u_ = 2 * np.pi * uv[0]  # poloidal angle
        v_ = 2 * np.pi * uv[1] / self.num_tor_symmetry + self.axis_angle  # toroidal angle

        minor_radius = (fourier_transform(self.fourier_coeffs, u_) + 1) * self.minor_radius

        x = (self.major_radius + minor_radius * np.cos(u_)) * np.cos(v_)
        y = (self.major_radius + minor_radius * np.cos(u_)) * np.sin(v_)
        z = minor_radius * np.sin(u_)
        return np.array([x, y, z])

    def cartesian_to_toroidal(self):
        return cartesian_to_toroidal(xyz=self.xyz, tore_radius=self.major_radius, height=0.0)

    def plot_cross_section(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        rtphi = self.cartesian_to_toroidal()
        rphi = np.concatenate((rtphi[:, :, :], rtphi[:1, :, :]), axis=0)
        ax.plot(rphi[:, 0, 1], rphi[:, 0, 0], **kwargs)
        return ax
