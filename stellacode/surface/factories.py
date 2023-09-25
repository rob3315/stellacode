from stellacode.surface import (
    AbstractCurrent,
    Current,
    CurrentZeroTorBC,
    CylindricalSurface,
    FourierSurface,
    IntegrationParams,
    RotatedCoil,
    ToroidalSurface,
)
from stellacode.surface.cylindrical import CylindricalSurface
from stellacode.surface.imports import get_net_current
from stellacode.surface.tore import ToroidalSurface
from stellacode.surface.utils import fit_to_surface
from stellacode.tools.vmec import VMECIO
from stellacode.surface.rotated_surface import Sequential


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
            RotatedCoil(
                num_tor_symmetry=surf_plasma.num_tor_symmetry,
                rotate_diff_current=1,
                current=current,
            ),
        ]
    )


def get_pwc_surface(
    surf_plasma: FourierSurface,
    plasma_path: str,
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
    net_currents = get_net_current(plasma_path)
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
            net_currents=net_currents,
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
            RotatedCoil(
                num_tor_symmetry=surf_plasma.num_tor_symmetry,
                rotate_diff_current=rotate_diff_current,
                current=current,
                common_current_on_each_rot=common_current_on_each_rot,
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

    cws = RotatedCoil(
        surface=cws,
        current=Current(num_pol=n_harmonics, num_tor=n_harmonics, net_currents=get_net_current(path_plasma)),
        num_tor_symmetry=cws.num_tor_symmetry,
    )
    return cws


import numpy as np

from stellacode.surface.coil_surface import CoilSurface
from stellacode.surface.rotated_surface import RotatedSurface, ConcatSurfaces, RotateNTimes


def get_free_cylinders(
    surf_plasma,
    distance: float = 0.0,
    num_cyl=3,
    n_harmonics: int = 8,
    factor: int = 6,
):
    num_points = n_harmonics * factor
    num_tor_symmetry = VMECIO.from_grid(surf_plasma.file_path).nfp

    num_sym_by_cyl = num_tor_symmetry * num_cyl
    angle = 2 * np.pi / num_sym_by_cyl

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

        surface = CylindricalSurface(
            fourier_coeffs=fourier_coeffs,
            integration_par=IntegrationParams(num_points_u=num_points, num_points_v=num_points),
            num_tor_symmetry=num_sym_by_cyl,
            radius=surf_plasma.get_minor_radius() + distance,
            distance=surf_plasma.get_major_radius(),
        )

        coil_surf = CoilSurface(surface=surface, current=current)
        coil_surf = RotatedSurface(
            surface=coil_surf,
            rotate_n=RotateNTimes(angle=angle, max_num=n + 1, min_num=n),
        )
        surfaces.append(coil_surf)
    coil_surf = ConcatSurfaces(surfaces=surfaces)
    coil_surf = RotatedSurface(
        surface=coil_surf,
        rotate_n=RotateNTimes(angle=2 * np.pi / num_tor_symmetry, max_num=num_tor_symmetry),
    )
    return coil_surf
