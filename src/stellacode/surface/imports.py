"""
Imports for the surface module.
"""
import os
from os.path import join
from stellacode import np, PROJECT_PATH
from stellacode.definitions import PlasmaConfig
from stellacode.surface.factory_tools import CoilFactory, RotatedSurface, RotateNTimes
from stellacode.tools.vmec import VMECIO

from .abstract_surface import AbstractBaseFactory, IntegrationParams, Surface
from .current import Current
from .factory_tools import Sequential, rotate_coil
from .fourier import FourierSurfaceFactory


def get_cws(config, build_coils: bool = False):
    """
    Create a surface factory for a Fourier surface from a config.

    Args:
        config (dict): Config dictionary containing geometry parameters.
        build_coils (bool, optional): Whether to build coils. Defaults to False.

    Returns:
        AbstractBaseFactory: Surface factory.
    """
    # Get number of field periods
    nfp = int(config["geometry"]["Np"])

    # Get number of poloidal and toroidal coils
    n_pol_coil = int(config["geometry"]["ntheta_coil"])
    n_tor_coil = int(config["geometry"]["nzeta_coil"])

    # Get path to CWS file and replace forward slashes with platform-specific path separator
    path_cws = join(PROJECT_PATH, str(
        config["geometry"]["path_cws"]).replace("/", os.sep))

    # Create Fourier surface factory
    cws = FourierSurfaceFactory.from_file(
        path_cws,
        integration_par=IntegrationParams(
            num_points_u=n_pol_coil, num_points_v=n_tor_coil),
        nfp=nfp
    )

    # Create rotated surface factory
    surface_factory = RotatedSurface(rotate_n=RotateNTimes.from_nfp(nfp))

    # Create coil factory
    coil_factory = CoilFactory(
        current=get_current_potential(config),
        build_coils=build_coils
    )

    # Create sequential surface factory
    seq_surface_factory = Sequential(
        surface_factories=[cws, coil_factory, surface_factory]
    )

    return seq_surface_factory


def get_cws_from_plasma_config(
    plasma_config: PlasmaConfig,
    n_harmonics_current: int,
    mult_coil_points: int = 6,
) -> AbstractBaseFactory:
    """
    Create a Fourier surface factory from a plasma config.

    Args:
        plasma_config: The plasma config.
        n_harmonics_current: The number of harmonics for the current.
        mult_coil_points: The multiplication factor for the number of points on the coil.

    Returns:
        The Fourier surface factory.
    """
    # Check if the path to the CWS data is provided
    assert plasma_config.path_cws is not None

    # Get the number of field periods from the plasma config
    nfp = VMECIO.from_grid(plasma_config.path_plasma).nfp

    # Create a Fourier surface factory from the CWS data
    cws = FourierSurfaceFactory.from_file(
        plasma_config.path_cws,
        integration_par=IntegrationParams(
            num_points_u=n_harmonics_current * mult_coil_points,
            num_points_v=n_harmonics_current * mult_coil_points,
        ),
        nfp=nfp,
    )

    # Create a current object with the specified number of harmonics and the net currents from the plasma config
    current = Current(
        num_pol=n_harmonics_current,
        num_tor=n_harmonics_current,
        net_currents=get_net_current(plasma_config.path_plasma),
    )

    # Create a sequential surface factory that applies the Fourier surface factory, the current rotation, and the
    # coil rotation to the surface
    cws = Sequential(
        surface_factories=[
            cws,
            rotate_coil(
                current=current,
                nfp=cws.nfp,
            ),
        ]
    )

    return cws


def get_cws_grid(config):
    """
    Get the grid for Fourier surface factory.

    Args:
        config (dict): Config dictionary containing geometry parameters.

    Returns:
        Surface: The Fourier surface factory grid.
    """
    # Get number of poloidal and toroidal coils
    n_pol_coil = int(config["geometry"]["ntheta_coil"])
    n_tor_coil = int(config["geometry"]["nzeta_coil"])

    # Return the Fourier surface factory grid
    return Surface.get_uvgrid(n_pol_coil, n_tor_coil)


def get_net_current(plasma_path):
    """
    Compute the net current of a plasma.

    Args:
        plasma_path (str): Path to the plasma grid.

    Returns:
        numpy.ndarray: The net current of the plasma.
            The first element is the net poloidal current,
            and the second element is the net toroidal current.

    Notes:
        The net current is obtained by dividing the net
        poloidal current by the number of field periods (nfp).
    """
    # Load the plasma grid
    vmec = VMECIO.from_grid(plasma_path)

    # Get the number of field periods
    nfp = vmec.nfp

    # Compute the net current
    # The net poloidal current is divided by nfp to get the net current per field period
    return -np.array([vmec.net_poloidal_current / nfp, 0.0])


def get_current_potential(config):
    """
    Get the current parameters from the configuration.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        Current: The current parameters.

    Notes:
        The net current is obtained by dividing the net poloidal current by the number of field periods (nfp).
    """
    # Get number of poloidal and toroidal coils
    mpol_coil = int(config["geometry"]["mpol_coil"])
    ntor_coil = int(config["geometry"]["ntor_coil"])

    # Get the number of field periods
    nfp = int(config["geometry"]["Np"])

    # Compute the net current
    # The net poloidal current is divided by nfp to get the net current per field period
    net_currents = -np.array(
        [
            float(config["other"]["net_poloidal_current_Amperes"]) / nfp,
            float(config["other"]["net_toroidal_current_Amperes"]),
        ]
    )

    # Return the current and potential parameters
    return Current(num_pol=mpol_coil, num_tor=ntor_coil, net_currents=net_currents)


def get_plasma_surface(config):
    """
    Get the plasma surface from the configuration.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        FourierSurfaceFactory: The plasma surface.

    Note:
        The number of poloidal and toroidal points of the plasma surface is obtained from the configuration.
        The path to the plasma surface file is obtained from the configuration.
        The number of field periods is obtained from the configuration.
    """
    # Get the number of poloidal and toroidal points of the plasma surface
    n_pol_plasma = int(config["geometry"]["ntheta_plasma"])
    n_tor_plasma = int(config["geometry"]["nzeta_plasma"])

    # Get the path to the plasma surface file
    path_plasma = join(PROJECT_PATH, str(
        config["geometry"]["path_plasma"]).replace("/", os.sep))

    # Create the plasma surface
    plasma = FourierSurfaceFactory.from_file(
        path_plasma,  # Path to the plasma surface file
        integration_par=IntegrationParams(  # Integration parameters for the Fourier series
            num_points_u=n_pol_plasma,  # Number of poloidal points
            num_points_v=n_tor_plasma,  # Number of toroidal points
        ),
        nfp=int(config["geometry"]["Np"]),  # Number of field periods
    )

    return plasma
