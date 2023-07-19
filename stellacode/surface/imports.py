"""
Imports for the surface module.
"""
from stellacode import np

from .abstract_surface import AbstractSurface, IntegrationParams
from .current import Current
from .fourier import FourierSurface
from .rotated_surface import RotatedSurface


def get_cws(config):
    n_fp = int(config["geometry"]["Np"])
    n_pol_coil = int(config["geometry"]["ntheta_coil"])
    n_tor_coil = int(config["geometry"]["nzeta_coil"])
    path_cws = str(config["geometry"]["path_cws"])
    cws = FourierSurface.from_file(
        path_cws, integration_par=IntegrationParams(num_points_u=n_pol_coil, num_points_v=n_tor_coil), n_fp=n_fp
    )

    cws = RotatedSurface(
        surface=cws,
        current=get_current_potential(config),
        num_tor_symmetry=n_fp,
    )
    return cws


def get_cws_grid(config):
    n_pol_coil = int(config["geometry"]["ntheta_coil"])
    n_tor_coil = int(config["geometry"]["nzeta_coil"])

    return AbstractSurface.get_uvgrid(n_pol_coil, n_tor_coil)


def get_current_potential(config):
    mpol_coil = int(config["geometry"]["mpol_coil"])
    ntor_coil = int(config["geometry"]["ntor_coil"])
    return Current(num_pol=mpol_coil, num_tor=ntor_coil)


def get_plasma_surface(config):
    n_pol_plasma = int(config["geometry"]["ntheta_plasma"])
    n_tor_plasma = int(config["geometry"]["nzeta_plasma"])
    path_plasma = str(config["geometry"]["path_plasma"])

    plasma = FourierSurface.from_file(
        path_plasma, integration_par=IntegrationParams(num_points_u=n_pol_plasma, num_points_v=n_tor_plasma),
        n_fp=int(config["geometry"]["Np"])
    )
    return plasma


def plot_cws_and_plasma(cws, plasma):
    """Plots two surfaces.

    :param cws: coil winding surface
    :type cws: Surface

    :param plasma: plasma surface
    :type plasma Surface

    :return: None
    :rtype: NoneType

    """
    from mayavi import mlab
    from numpy import linspace, zeros

    mlab.mesh(*cws.expand_for_plot_part(), representation="wireframe", colormap="Wistia")
    mlab.mesh(*plasma.expand_for_plot_part(), representation="surface", colormap="Spectral")
    mlab.plot3d(linspace(0, 5, 100), zeros(100), zeros(100), color=(1, 0, 0))
    mlab.plot3d(zeros(100), linspace(0, 5, 100), zeros(100), color=(0, 1, 0))
    mlab.plot3d(zeros(100), zeros(100), linspace(0, 5, 100), color=(0, 0, 1))
    mlab.show()
