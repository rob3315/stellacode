from stellacode.surface import (
    AbstractCurrent,
    Current,
    CurrentZeroTorBC,
    CylindricalSurface,
    FourierSurface,
    IntegrationParams,
    ToroidalSurface,
)
from stellacode.surface.factory_tools import rotate_coil
from stellacode.surface.cylindrical import CylindricalSurface
from stellacode.surface.imports import get_net_current
from stellacode.surface.tore import ToroidalSurface
from stellacode.surface.utils import fit_to_surface
from stellacode.tools.vmec import VMECIO
from stellacode.surface.factory_tools import Sequential
import numpy as np
from stellacode.surface.factory_tools import RotatedSurface, ConcatSurfaces, RotateNTimes
from .coil_surface import CoilFactory


def get_toroidal_surface(
    surf_plasma: FourierSurface,
    plasma_path: str,
    n_harmonics: int = 16,
    factor: int = 6,
    match_surface: bool = False,
    convex: bool = True,
    distance: float = 0.0,
    sin_basis: bool = True,
    cos_basis: bool = True,
):
    net_currents = get_net_current(plasma_path)
    current = Current(
        num_pol=n_harmonics,
        num_tor=n_harmonics,
        sin_basis=sin_basis,
        cos_basis=cos_basis,
        net_currents=net_currents,
    )
    if match_surface:
        tor_surf = surf_plasma.get_surface_envelope(num_coeff=10, convex=convex)
        tor_surf.update_params(minor_radius=tor_surf.minor_radius + distance)
    else:
        minor_radius = surf_plasma.get_minor_radius()
        major_radius = surf_plasma.get_major_radius()
        tor_surf = ToroidalSurface(
            num_tor_symmetry=surf_plasma.num_tor_symmetry,
            major_radius=major_radius,
            minor_radius=minor_radius + distance,
            integration_par=current.get_integration_params(factor=factor),
        )

    return Sequential(
        surface_factories=[
            tor_surf,
            rotate_coil(
                current=current,
                num_tor_symmetry=surf_plasma.num_tor_symmetry,
                num_surf_per_period=1,
                continuous_current_in_period=False,
            ),
        ]
    )


def get_pwc_surface(
    surf_plasma: FourierSurface,
    plasma_path: str = None,
    n_harmonics: int = 16,
    factor: int = 6,
    rotate_diff_current: int = 3,
    make_joints: bool = True,
    common_current_on_each_rot: bool = False,
    axis_angle: float = 0.0,
    distance: float = 0.0,
    sin_basis: bool = True,
    cos_basis: bool = True,
    convex: bool = True,
    match_surface: bool = False,
):
    net_currents = get_net_current(surf_plasma.file_path)
    if common_current_on_each_rot:
        current: AbstractCurrent = Current(
            num_pol=n_harmonics,
            num_tor=n_harmonics,
            sin_basis=sin_basis,
            cos_basis=cos_basis,
            net_currents=net_currents,
        )
    else:
        current = CurrentZeroTorBC(
            num_pol=n_harmonics,
            num_tor=n_harmonics // rotate_diff_current,
            sin_basis=sin_basis,
            cos_basis=cos_basis,
            net_currents=net_currents / rotate_diff_current,
        )
    integration_par = IntegrationParams(
        num_points_u=n_harmonics * factor,
        num_points_v=n_harmonics * factor // rotate_diff_current,
    )
    if match_surface:
        surf_coil = surf_plasma.get_surface_envelope(num_cyl=rotate_diff_current, num_coeff=10, convex=convex)
        surf_coil.update_params(
            radius=surf_coil.radius + distance,
            integration_par=integration_par,
            make_joints=make_joints,
        )
    else:
        surf_coil = CylindricalSurface(
            integration_par=integration_par,
            make_joints=make_joints,
            axis_angle=axis_angle,
            num_tor_symmetry=surf_plasma.num_tor_symmetry * rotate_diff_current,
        )

    surf_coil = Sequential(
        surface_factories=[
            surf_coil,
            rotate_coil(
                current=current,
                num_tor_symmetry=surf_plasma.num_tor_symmetry,
                num_surf_per_period=rotate_diff_current,
                continuous_current_in_period=common_current_on_each_rot,
            ),
        ]
    )
    if not match_surface:
        surf_coil = fit_to_surface(surf_coil, surf_plasma, distance=distance)

    return surf_coil


def get_original_cws(path_cws: str, path_plasma: str, n_harmonics: int = 16, factor: int = 6):
    num_tor_symmetry = VMECIO.from_grid(path_plasma).nfp
    cws = FourierSurface.from_file(
        path_cws,
        integration_par=IntegrationParams(
            num_points_u=n_harmonics * factor,
            num_points_v=n_harmonics * factor,
        ),
        n_fp=num_tor_symmetry,
    )

    cws = Sequential(
        surface_factories=[
            cws,
            rotate_coil(
                current=Current(num_pol=n_harmonics, num_tor=n_harmonics, net_currents=get_net_current(path_plasma)),
                num_tor_symmetry=cws.num_tor_symmetry,
            ),
        ]
    )

    return cws


def get_free_cylinders(
    surf_plasma,
    distance: float = 0.0,
    num_cyl: int = 3,
    n_harmonics: int = 8,
    factor: int = 6,
):
    num_sym_by_cyl = surf_plasma.num_tor_symmetry * num_cyl
    angle = 2 * np.pi / num_sym_by_cyl
    num_points = n_harmonics * factor
    surfaces = []
    for n in range(num_cyl):
        current = CurrentZeroTorBC(
            num_pol=n_harmonics,
            num_tor=n_harmonics,
            sin_basis=True,
            cos_basis=True,
            net_currents=get_net_current(surf_plasma.file_path),
        )
        fourier_coeffs = np.zeros((5, 2))
        minor_radius = surf_plasma.get_minor_radius()
        major_radius = surf_plasma.get_major_radius()
        surface = CylindricalSurface(
            fourier_coeffs=fourier_coeffs,
            integration_par=IntegrationParams(num_points_u=num_points, num_points_v=num_points),
            num_tor_symmetry=num_sym_by_cyl,
            radius=minor_radius + distance,
            distance=major_radius,
        )

        coil_surf = Sequential(
            surface_factories=[
                surface,
                CoilFactory(current=current),
                RotatedSurface(
                    rotate_n=RotateNTimes(angle=angle, max_num=n + 1, min_num=n),
                ),
            ]
        )
        surfaces.append(coil_surf)

    coil_surf = Sequential(
        surface_factories=[
            ConcatSurfaces(surface_factories=surfaces),
            RotatedSurface(
                rotate_n=RotateNTimes(
                    angle=2 * np.pi / surf_plasma.num_tor_symmetry, max_num=surf_plasma.num_tor_symmetry
                )
            ),
        ]
    )
    return coil_surf
