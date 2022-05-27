"""
Some utilitary functions.
"""

from numpy import linspace, zeros
from mayavi import mlab
from configparser import ConfigParser

from .surface_Fourier import Surface_Fourier
from .surface_pwc_fourier import Surface_PWC_Fourier
from .surface_from_file import surface_from_file


def get_cws_and_plasma(path_config_file=None, config=None):
    if config is None:
        print('path_config : {}'.format(path_config_file))
        config = ConfigParser()
        config.read(path_config_file)

    n_fp = int(config['geometry']['Np'])
    n_pol_plasma = int(config['geometry']['ntheta_plasma'])
    n_pol_coil = int(config['geometry']['ntheta_coil'])
    n_tor_plasma = int(config['geometry']['nzeta_plasma'])
    n_tor_coil = int(config['geometry']['nzeta_coil'])
    path_plasma = str(config['geometry']['path_plasma'])
    path_cws = str(config['geometry']['path_cws'])

    cws = surface_from_file(path_cws, n_fp, n_pol_coil, n_tor_coil)
    plasma = surface_from_file(path_plasma, n_fp, n_pol_plasma, n_tor_plasma)

    return cws, plasma


def plot_cws_and_plasma(cws, plasma):
    mlab.mesh(*cws.expand_for_plot_part(),
              representation="wireframe", colormap="Wistia")
    mlab.mesh(*plasma.expand_for_plot_part(),
              representation="surface", colormap="Spectral")
    mlab.plot3d(linspace(0, 3, 100), zeros(
        100), zeros(100), color=(1, 0, 0))
    mlab.plot3d(zeros(100), linspace(0, 3, 100),
                zeros(100), color=(0, 1, 0))
    mlab.plot3d(zeros(100), zeros(100),
                linspace(0, 3, 100), color=(0, 0, 1))
    mlab.show()
