"""
Imports for the surface module.
"""
from stellacode.surface.abstract_surface import AbstractSurface

from .surface_from_file import surface_from_file


def get_cws(config):
    n_fp = int(config["geometry"]["Np"])
    n_pol_coil = int(config["geometry"]["ntheta_coil"])
    n_tor_coil = int(config["geometry"]["nzeta_coil"])
    path_cws = str(config["geometry"]["path_cws"])
    cws = surface_from_file(path_cws, n_fp, n_pol_coil, n_tor_coil)
    return cws


def get_cws_grid(config):
    n_pol_coil = int(config["geometry"]["ntheta_coil"])
    n_tor_coil = int(config["geometry"]["nzeta_coil"])

    return AbstractSurface.get_uvgrid(n_pol_coil, n_tor_coil)


def get_plasma_surface(config):
    n_fp = int(config["geometry"]["Np"])
    n_pol_plasma = int(config["geometry"]["ntheta_plasma"])
    n_tor_plasma = int(config["geometry"]["nzeta_plasma"])
    path_plasma = str(config["geometry"]["path_plasma"])
    plasma = surface_from_file(path_plasma, n_fp, n_pol_plasma, n_tor_plasma)
    return plasma


def get_cws_and_plasma(path_config_file=None, config=None):
    """From a configuration file, returns the CWS and the plasma surface.

    :param path_config_file: path to configuration file, optionnal
    :type path_config_file: str

    :param config: configuration, optionnal
    :type config: configparser.ConfigParser

    :return: two surfaces : CWS and plasma
    :rtype: tuple (Surface, Surface)

    """
    from configparser import ConfigParser

    if config is None:
        print("path_config : {}".format(path_config_file))
        config = ConfigParser()
        config.read(path_config_file)

    n_fp = int(config["geometry"]["Np"])
    n_pol_plasma = int(config["geometry"]["ntheta_plasma"])
    n_pol_coil = int(config["geometry"]["ntheta_coil"])
    n_tor_plasma = int(config["geometry"]["nzeta_plasma"])
    n_tor_coil = int(config["geometry"]["nzeta_coil"])
    path_plasma = str(config["geometry"]["path_plasma"])
    path_cws = str(config["geometry"]["path_cws"])

    cws = surface_from_file(path_cws, n_fp, n_pol_coil, n_tor_coil)
    plasma = surface_from_file(path_plasma, n_fp, n_pol_plasma, n_tor_plasma)

    return cws, plasma


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
