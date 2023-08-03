from stellacode.surface import ToroidalSurface, RotatedSurface, CylindricalSurface
from stellacode.surface.cylindrical import CylindricalSurface
from stellacode.surface.tore import ToroidalSurface

from stellacode.surface.imports import get_net_current
from stellacode.surface.utils import fit_to_surface

from stellacode.surface import Current, IntegrationParams, CurrentZeroTorBC, AbstractCurrent
from stellacode.surface import FourierSurface


def get_toroidal_surface(
    surf_plasma: FourierSurface,
    n_harmonics: int = 16,
    factor: int = 6,
    match_surface: bool = False,
    convex: bool = True,
    distance: float = 0.0,
    sin_basis: bool = True,
    cos_basis: bool = True,
):
    net_currents = get_net_current(surf_plasma.file_path)
    current = Current(
        num_pol=n_harmonics,
        num_tor=n_harmonics,
        sin_basis=sin_basis,
        cos_basis=cos_basis,
        net_currents=net_currents,
    )
    if match_surface:
        tor_surf = surf_plasma.get_axisymmetric_surface_envelope(num_coeff=10, convex=convex)
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
    return RotatedSurface(
        surface=tor_surf,
        num_tor_symmetry=surf_plasma.num_tor_symmetry,
        rotate_diff_current=1,
        current=current,
    )


def get_pwc_surface(
    Sp: FourierSurface,
    n_harmonics: int = 16,
    factor: int = 6,
    rotate_diff_current: int = 3,
    make_joints: bool = True,
    common_current_on_each_rot: bool = False,
    axis_angle: float = 0.0,
    distance: float = 0.0,
    sin_basis: bool = True,
    cos_basis: bool = True,
):
    net_currents = get_net_current(Sp.file_path)
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

    S = RotatedSurface(
        surface=CylindricalSurface(
            integration_par=IntegrationParams(
                num_points_u=n_harmonics * factor,
                num_points_v=n_harmonics * factor // rotate_diff_current,
            ),
            num_tor_symmetry=Sp.num_tor_symmetry * rotate_diff_current,
            make_joints=make_joints,
            axis_angle=axis_angle,
        ),
        num_tor_symmetry=Sp.num_tor_symmetry,
        rotate_diff_current=rotate_diff_current,
        current=current,
        common_current_on_each_rot=common_current_on_each_rot,
    )

    S = fit_to_surface(S, Sp, distance=distance)

    return S


def get_original_cws(path_cws: str, path_plasma: str, n_harmonics: int = 16, factor: int = 6):
    assert path_cws[-3::] == ".nc"

    cws = FourierSurface.from_file(
        path_cws,
        integration_par=IntegrationParams(
            num_points_u=n_harmonics * factor,
            num_points_v=n_harmonics * factor,
        ),
    )

    cws = RotatedSurface(
        surface=cws,
        current=Current(num_pol=n_harmonics, num_tor=n_harmonics, net_currents=get_net_current(path_plasma)),
        num_tor_symmetry=cws.num_tor_symmetry,
    )
    return cws
