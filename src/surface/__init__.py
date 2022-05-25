"""
implementation of the surfaces
"""

from numpy import linspace, zeros
from mayavi import mlab

from .surface_Fourier import Surface_Fourier
from .surface_pwc_fourier import Surface_PWC_Fourier
from .surface_pwc_spline import Surface_PWC_Spline


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
