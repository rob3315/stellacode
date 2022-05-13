"""
implementation of the surfaces
"""

from mayavi import mlab


def plot_surfaces(list_surfaces):
    for surface in list_surfaces:
        mlab.mesh(*surface.expand_for_plot_part())
    mlab.show()
