import jax
import sys
import configparser

import numpy as onp
import pytest

from stellacode.surface import ToroidalSurface
from stellacode.surface.imports import get_plasma_surface
from stellacode.surface.utils import (
    fourier_coefficients,
    fourier_transform,
)

from os.path import dirname, join, realpath

import matplotlib
matplotlib.use("Agg")

jax.config.update("jax_enable_x64", True)

TEST_FOLDER_PATH = f"{(dirname(realpath(__file__)))}"


def test_fourier_series():
    """
    Test the accuracy of the Fourier series approximation.

    This function generates a random set of coefficients and evaluates the Fourier series
    approximation using the `fourier_coefficients` function. The approximation is then
    compared to the original coefficients using the `assert` statement. The test passes if
    the maximum absolute difference between the approximation and the original coefficients
    is less than 1e-7.

    """
    # Set the random seed for reproducibility
    onp.random.seed(0)

    # Generate random coefficients
    coefs = onp.random.rand(5, 2)

    # Compute the Fourier series approximation
    res = fourier_coefficients(
        0, 2 * onp.pi, 5, lambda val: fourier_transform(coefs, val))[1]

    # Compare the approximation to the original coefficients
    assert onp.max(onp.abs(res - coefs)
                   ) < 1e-7, "Fourier series approximation is inaccurate"


@pytest.mark.parametrize("convex", [True, False])
@pytest.mark.parametrize("num_cyl", [None, 3])
def test_surface_envelope_fourier_series(num_cyl, convex):
    """
    Test the accuracy of the surface envelope computed using the Fourier series approximation.

    This function tests the accuracy of the surface envelope computed using the Fourier series
    approximation. The function generates a surface envelope using the `get_surface_envelope`
    method of the surface factory. The surface envelope is then plotted using the `plot_cross_sections`
    method of the surface factory. The results are checked for accuracy by comparing the surface
    envelope to a toroidal surface if it is a `ToroidalSurface`.

    Parameters
    ----------
    num_cyl : int or None
        The number of cylinders used to compute the surface envelope. If None, the default number
        of cylinders is used.
    convex : bool
        If True, the convex envelope is computed. If False, the concave envelope is computed.

    """
    # Path to the configuration file
    path_config_file = join(
        TEST_FOLDER_PATH, "data", "li383", "config.ini")

    # Read the configuration file
    config = configparser.ConfigParser()
    config.read(path_config_file)

    # Get the surface factory
    factory = get_plasma_surface(config)

    # Generate the surface
    surf = factory()

    # Compute the surface envelope
    coil_surf = surf.get_surface_envelope(
        num_coeff=10, num_cyl=num_cyl, convex=convex, limit=1000)

    # Plot the cross sections
    fig, ax = factory.plot_cross_sections(
        num_cyl=num_cyl, convex_envelope=True, concave_envelope=True)

    # Check if the surface envelope is a toroidal surface
    if isinstance(coil_surf, ToroidalSurface):
        # Plot the cross section of the surface envelope
        coil_surf.plot_cross_section(ax=ax)
