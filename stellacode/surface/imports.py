"""
Imports for the surface module.
"""
from stellacode import np
from stellacode.tools.vmec import VMECIO
import os
from stellacode.surface.factory_tools import RotateNTimes, RotatedSurface, CoilFactory
from .abstract_surface import Surface, IntegrationParams, AbstractBaseFactory
from .current import Current
from .fourier import FourierSurface
from .factory_tools import Sequential, rotate_coil
from stellacode.definitions import PlasmaConfig


def get_cws(config):
    n_fp = int(config["geometry"]["Np"])
    n_pol_coil = int(config["geometry"]["ntheta_coil"])
    n_tor_coil = int(config["geometry"]["nzeta_coil"])
    path_cws = str(config["geometry"]["path_cws"]).replace("/", os.sep)
    cws = FourierSurface.from_file(
        path_cws, integration_par=IntegrationParams(num_points_u=n_pol_coil, num_points_v=n_tor_coil), n_fp=n_fp
    )

    rot_nfp = RotatedSurface(rotate_n=RotateNTimes.from_nfp(n_fp))
    coil_factory = CoilFactory(current=get_current_potential(config))
    surface_factory = Sequential(surface_factories=[cws, coil_factory, rot_nfp])

    return surface_factory


def get_cws_from_plasma_config(
    plasma_config: PlasmaConfig,
    n_harmonics_current: int,
    mult_coil_points: int = 6,
):
    assert plasma_config.path_cws is not None
    num_tor_symmetry = VMECIO.from_grid(plasma_config.path_plasma).nfp
    cws = FourierSurface.from_file(
        plasma_config.path_cws,
        integration_par=IntegrationParams(
            num_points_u=n_harmonics_current * mult_coil_points,
            num_points_v=n_harmonics_current * mult_coil_points,
        ),
        n_fp=num_tor_symmetry,
    )

    current = Current(
        num_pol=n_harmonics_current,
        num_tor=n_harmonics_current,
        net_currents=get_net_current(plasma_config.path_plasma),
    )
    cws = Sequential(
        surface_factories=[
            cws,
            rotate_coil(
                current=current,
                num_tor_symmetry=cws.num_tor_symmetry,
            ),
        ]
    )
    return cws


def get_cws_grid(config):
    n_pol_coil = int(config["geometry"]["ntheta_coil"])
    n_tor_coil = int(config["geometry"]["nzeta_coil"])

    return Surface.get_uvgrid(n_pol_coil, n_tor_coil)


def get_net_current(plasma_path):
    vmec = VMECIO.from_grid(plasma_path)
    num_tor_symmetry = vmec.nfp
    return -np.array([vmec.net_poloidal_current / num_tor_symmetry, 0.0])


def get_current_potential(config):
    mpol_coil = int(config["geometry"]["mpol_coil"])
    ntor_coil = int(config["geometry"]["ntor_coil"])
    num_tor_symmetry = int(config["geometry"]["Np"])
    net_currents = -np.array(
        [
            float(config["other"]["net_poloidal_current_Amperes"]) / num_tor_symmetry,
            float(config["other"]["net_toroidal_current_Amperes"]),
        ]
    )
    return Current(num_pol=mpol_coil, num_tor=ntor_coil, net_currents=net_currents)


def get_plasma_surface(config):
    n_pol_plasma = int(config["geometry"]["ntheta_plasma"])
    n_tor_plasma = int(config["geometry"]["nzeta_plasma"])
    path_plasma = str(config["geometry"]["path_plasma"]).replace("/", os.sep)

    plasma = FourierSurface.from_file(
        path_plasma,
        integration_par=IntegrationParams(num_points_u=n_pol_plasma, num_points_v=n_tor_plasma),
        n_fp=int(config["geometry"]["Np"]),
    )
    return plasma
