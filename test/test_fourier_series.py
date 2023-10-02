import configparser

import numpy as onp
import pytest

from stellacode import np
from stellacode.surface import ToroidalSurface
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


@pytest.mark.parametrize("convex", [True, False])
@pytest.mark.parametrize("num_cyl", [None, 3])
def test_surface_envelope_fourier_series(num_cyl, convex):
    path_config_file = "test/data/li383/config.ini"
    config = configparser.ConfigParser()
    config.read(path_config_file)
    factory = get_plasma_surface(config)
    surf = factory()
    coil_surf = surf.get_surface_envelope(num_coeff=10, num_cyl=num_cyl, convex=convex)
    ax = factory.plot_cross_sections(num_cyl=num_cyl, convex_envelope=True, concave_envelope=True)
    if isinstance(coil_surf, ToroidalSurface):
        coil_surf.plot_cross_section(ax=ax)
    # import matplotlib.pyplot as plt
    # plt.show()

    # surf.plot(only_one_period=True)
    # coil_surf().plot(only_one_period=True)
