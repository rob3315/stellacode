import typing as tp

from jax import Array
from jax.typing import ArrayLike

from stellacode import np

from .abstract_surface import AbstractSurfaceFactory, IntegrationParams
from .utils import cartesian_to_toroidal, fourier_transform


class ToroidalSurface(AbstractSurfaceFactory):
    """
    Axisymmetric surface factory

    Args:
        * nfp: number of field periods
        * major_radius: major radius of the torus
        * minor radius: average minor radius of the torus
        * fourier_coeffs: Fourier representation of the surface cross section
        * axis_angle: rotate the surface along the toroidal angle by the given axis angle (just there for testing purposes)
    """

    nfp: int
    major_radius: float = 5.0
    minor_radius: float = 1.0
    fourier_coeffs: ArrayLike = np.zeros((4, 2))
    axis_angle: float = 0.0

    trainable_params: tp.List[str] = [
        "major_radius",
        "minor_radius",
        "fourier_coeffs",
    ]

    def get_xyz(self, uv):
        u_ = 2 * np.pi * uv[0]  # poloidal angle
        v_ = 2 * np.pi * uv[1] / self.nfp + self.axis_angle  # toroidal angle

        minor_radius = (fourier_transform(self.fourier_coeffs, u_) + 1) * self.minor_radius

        x = (self.major_radius + minor_radius * np.cos(u_)) * np.cos(v_)
        y = (self.major_radius + minor_radius * np.cos(u_)) * np.sin(v_)
        z = minor_radius * np.sin(u_)
        return np.array([x, y, z])

    def plot_cross_section(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        surf = self()
        rtphi = cartesian_to_toroidal(xyz=surf.xyz, tore_radius=self.major_radius, height=0.0)
        rphi = np.concatenate((rtphi[:, :, :], rtphi[:1, :, :]), axis=0)
        ax.plot(rphi[:, 0, 1], rphi[:, 0, 0], **kwargs)
        return ax