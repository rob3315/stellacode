import configparser

import numpy as onp

from stellacode import np
from stellacode.surface.imports import get_plasma_surface
from stellacode.surface.utils import (
    fit_to_surface,
    fourier_coefficients,
    fourier_transform,
)


def test_fourier_series():
    coefs = onp.random.rand(5, 2)

    res = fourier_coefficients(0, 2 * np.pi, 5, lambda val: fourier_transform(coefs, val))[1]

    assert np.max(np.abs(res - coefs)) < 1e-15


def test_convex_hull_fourier_series():
    path_config_file = "test/data/li383/config.ini"
    config = configparser.ConfigParser()
    config.read(path_config_file)
    surf = get_plasma_surface(config)

    tor_surf = surf.get_toroidal_surface_convex_hull(num_coeff=10)
    ax = surf.plot_cross_sections()
    tor_surf.plot_cross_section(ax=ax)
    # import matplotlib.pyplot as plt
    # plt.show()

    # surf.plot(only_one_period=True)
    # tor_surf.plot(only_one_period=True)

    new_surface = fit_to_surface(tor_surf, surf)

    assert new_surface.get_min_distance(surf.xyz) < 3e-2
