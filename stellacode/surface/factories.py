from stellacode.surface import (
    AbstractCurrent,
    Current,
    CurrentZeroTorBC,
    CylindricalSurface,
    FourierSurface,
    IntegrationParams,
    ToroidalSurface,
)
from stellacode.surface.abstract_surface import AbstractBaseFactory
from stellacode.surface.factory_tools import rotate_coil
from stellacode.surface.cylindrical import CylindricalSurface
from stellacode.surface.imports import get_net_current
from stellacode.surface.tore import ToroidalSurface
from stellacode.surface.utils import fit_to_surface
from stellacode.tools.vmec import VMECIO
from stellacode.surface.factory_tools import Sequential
import jax.numpy as np
from stellacode.surface.factory_tools import RotatedSurface, ConcatSurfaces, RotateNTimes
from .coil_surface import CoilFactory
from jax.typing import ArrayLike
from jax import Array
import jax
import typing as tp
import matplotlib.pyplot as plt

from .abstract_surface import AbstractBaseFactory
from .coil_surface import CoilFactory


class AbstractToroidalCoils(AbstractBaseFactory):
    def plot_cross_section(self, **kwargs):
        raise NotImplementedError

    def scale_minor_radius(self, scale: float):
        raise NotImplementedError

    def set_base_current_par(self, **kwargs):
        raise NotImplementedError


class WrappedCoil(AbstractToroidalCoils):
    coil_factory: AbstractBaseFactory

    @classmethod
    def from_plasma(
        cls,
        surf_plasma: FourierSurface,
        surf_type: str = "cylindrical",
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
        build_coils: bool = False,
    ):
        if surf_type == "cylindrical":
            return cls(
                coil_factory=get_pwc_surface(
                    surf_plasma=surf_plasma,
                    n_harmonics=n_harmonics,
                    factor=factor,
                    rotate_diff_current=rotate_diff_current,
                    make_joints=make_joints,
                    common_current_on_each_rot=common_current_on_each_rot,
                    axis_angle=axis_angle,
                    distance=distance,
                    sin_basis=sin_basis,
                    cos_basis=cos_basis,
                    convex=convex,
                    match_surface=match_surface,
                    build_coils=build_coils,
                )
            )
        elif surf_type == "toroidal":
            return cls(
                coil_factory=get_toroidal_surface(
                    surf_plasma=surf_plasma,
                    n_harmonics=n_harmonics,
                    factor=factor,
                    match_surface=match_surface,
                    convex=convex,
                    distance=distance,
                    sin_basis=sin_basis,
                    cos_basis=cos_basis,
                    build_coils=build_coils,
                )
            )

        else:
            raise NotImplementedError

    def __call__(self, **kwargs):
        return self.coil_factory(**kwargs)

    def get_trainable_params(self):
        return self.coil_factory.get_trainable_params()

    def update_params(self, **kwargs):
        self.coil_factory.update_params(**kwargs)

    def plot_cross_section(self, **kwargs):
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        self._get_base_surface().plot_cross_section(ax=ax, **kwargs)
        return ax

    def scale_minor_radius(self, scale: float):
        try:
            self._get_base_surface().radius *= scale
        except:
            self._get_base_surface().minor_radius *= scale

    def set_base_current_par(self, **kwargs):
        current = self._get_current()
        for k, v in kwargs.items():
            setattr(current, k, v)

    def set_phi_mn(self, phi_mn: Array):
        current = self._get_current()
        current.set_phi_mn(phi_mn=phi_mn)

    def get_phi_mn(self):
        current = self._get_current()
        return current.get_phi_mn()

    def _get_current(self):
        if isinstance(self.coil_factory.surface_factories[1].surface_factories[0], AbstractCurrent):
            return self.coil_factory.surface_factories[1].surface_factories[0].current
        else:
            return self.coil_factory.surface_factories[1].surface_factories[1].current

    def _get_base_surface(self):
        return self.coil_factory.surface_factories[0]


def get_toroidal_surface(
    surf_plasma: FourierSurface,
    n_harmonics: int = 16,
    factor: int = 6,
    match_surface: bool = False,
    convex: bool = True,
    distance: float = 0.0,
    sin_basis: bool = True,
    cos_basis: bool = True,
    build_coils: bool = False,
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
        tor_surf = surf_plasma.get_surface_envelope(num_coeff=10, convex=convex)
        tor_surf.update_params(minor_radius=tor_surf.minor_radius + distance)
    else:
        minor_radius = surf_plasma.get_minor_radius()
        major_radius = surf_plasma.get_major_radius()
        tor_surf = ToroidalSurface(
            nfp=surf_plasma.nfp,
            major_radius=major_radius,
            minor_radius=minor_radius + distance,
            integration_par=current.get_integration_params(factor=factor),
        )

    return Sequential(
        surface_factories=[
            tor_surf,
            rotate_coil(
                current=current,
                nfp=surf_plasma.nfp,
                num_surf_per_period=1,
                continuous_current_in_period=False,
                build_coils=build_coils,
            ),
        ]
    )


def get_pwc_surface(
    surf_plasma: FourierSurface,
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
    build_coils: bool = False,
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
            nfp=surf_plasma.nfp * rotate_diff_current,
        )

    surf_coil = Sequential(
        surface_factories=[
            surf_coil,
            rotate_coil(
                current=current,
                nfp=surf_plasma.nfp,
                num_surf_per_period=rotate_diff_current,
                continuous_current_in_period=common_current_on_each_rot,
                build_coils=build_coils,
            ),
        ]
    )
    if not match_surface:
        surf_coil = fit_to_surface(surf_coil, surf_plasma, distance=distance)

    return surf_coil


def get_original_cws(path_cws: str, path_plasma: str, n_harmonics: int = 16, factor: int = 6):
    nfp = VMECIO.from_grid(path_plasma).nfp
    cws = FourierSurface.from_file(
        path_cws,
        integration_par=IntegrationParams(
            num_points_u=n_harmonics * factor,
            num_points_v=n_harmonics * factor,
        ),
        n_fp=nfp,
    )

    cws = Sequential(
        surface_factories=[
            cws,
            rotate_coil(
                current=Current(num_pol=n_harmonics, num_tor=n_harmonics, net_currents=get_net_current(path_plasma)),
                nfp=cws.nfp,
            ),
        ]
    )

    return cws


class FreeCylinders(AbstractToroidalCoils):
    net_current: ArrayLike
    tor_currents_w: ArrayLike
    pol_currents_w: ArrayLike
    coil_factory: AbstractBaseFactory
    constrain_tor_current: bool = True

    @classmethod
    def from_plasma(
        cls,
        surf_plasma,
        distance: float = 0.0,
        num_cyl: int = 3,
        n_harmonics_u: int = 8,
        n_harmonics_v: int = 4,
        factor: int = 6,
        constrain_tor_current: bool = True,
    ):
        num_sym_by_cyl = surf_plasma.nfp * num_cyl
        angle = 2 * np.pi / num_sym_by_cyl

        surfaces = []
        for n in range(num_cyl):
            current = CurrentZeroTorBC(
                num_pol=n_harmonics_u, num_tor=n_harmonics_v, sin_basis=True, cos_basis=True, net_currents=np.zeros(2)
            )
            fourier_coeffs = np.zeros((5, 2))
            minor_radius = surf_plasma.get_minor_radius()
            major_radius = surf_plasma.get_major_radius()
            surface = CylindricalSurface(
                fourier_coeffs=fourier_coeffs,
                integration_par=IntegrationParams(
                    num_points_u=n_harmonics_u * factor, num_points_v=n_harmonics_v * factor
                ),
                nfp=num_sym_by_cyl,
                radius=minor_radius + distance,
                distance=major_radius,
                axis_angle=angle * n,
            )

            coil_fac_ = Sequential(
                surface_factories=[
                    surface,
                    CoilFactory(current=current),
                ]
            )
            surfaces.append(coil_fac_)

        coil_factory = Sequential(
            surface_factories=[
                ConcatSurfaces(surface_factories=surfaces),
                RotatedSurface(rotate_n=RotateNTimes(angle=2 * np.pi / surf_plasma.nfp, max_num=surf_plasma.nfp)),
            ]
        )

        return cls(
            coil_factory=coil_factory,
            net_current=get_net_current(surf_plasma.file_path),
            tor_currents_w=np.zeros(num_cyl),
            pol_currents_w=np.zeros(num_cyl),
            constrain_tor_current=constrain_tor_current,
        )

    def __call__(self, **kwargs):
        pol_currents = self.net_current[0] * jax.nn.softmax(self.pol_currents_w)
        if self.constrain_tor_current:
            tor_currents = self.net_current[1] * jax.nn.softmax(self.tor_currents_w)
        else:
            tor_currents = self.tor_currents_w * 1e7

        coils = self.coil_factory.surface_factories[0].surface_factories
        for i in range(len(coils)):
            coils[i].surface_factories[1].current.net_currents = np.array([pol_currents[i], tor_currents[i]])

        return self.coil_factory(**kwargs)

    def get_trainable_params(self):
        return {
            **dict(tor_currents_w=self.tor_currents_w, pol_currents_w=self.pol_currents_w),
            **self.coil_factory.get_trainable_params(),
        }

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            if k in dir(self):
                setattr(self, k, v)
        kwargs = {k: v for k, v in kwargs.items() if k not in ["tor_currents_w", "pol_currents_w"]}
        self.coil_factory.update_params(**kwargs)

    def plot_cross_section(self, **kwargs):
        coils = self.coil_factory.surface_factories[0].surface_factories
        fig, axes = plt.subplots(len(coils), subplot_kw={"projection": "polar"})
        for i in range(len(coils)):
            coils[i].surface_factories[0].plot_cross_section(ax=axes[i], **kwargs)
        return axes

    def scale_minor_radius(self, scale: float):
        coils = self.coil_factory.surface_factories[0].surface_factories
        for i in range(len(coils)):
            coils[i].surface_factories[0].radius *= scale

    def set_base_current_par(self, **kwargs):
        coils = self.coil_factory.surface_factories[0].surface_factories
        for i in range(len(coils)):
            for k, v in kwargs.items():
                current = coils[i].surface_factories[1].current
                setattr(current, k, v)


class StackedToroidalCoils(AbstractBaseFactory):
    coil_factory: ConcatSurfaces
    net_current: ArrayLike
    tor_currents_w: ArrayLike
    pol_currents_w: ArrayLike
    distance_between_coils: float
    constrain_tor_current: bool = True

    @classmethod
    def from_surface(
        cls,
        surf_plasma,
        distance_to_plasma: float,
        distance_between_coils: float,
        n_harmonics: int = 8,
        factor: int = 6,
        constrain_tor_current: bool = True,
    ):
        surf_factory = get_toroidal_surface(
            surf_plasma=surf_plasma,
            plasma_path=surf_plasma.path_plasma,
            n_harmonics=n_harmonics,
            factor=factor,
            distance=distance_to_plasma,
        )

        current = Current(
            num_pol=n_harmonics,
            num_tor=n_harmonics,
            net_currents=np.zeros(2),
        )

        coil_factory = Sequential(
            surface_factories=[
                surf_factory,
                CoilFactory(current=current),
                RotatedSurface(rotate_n=RotateNTimes.from_nfp(surf_plasma.nfp)),
            ]
        )

        return cls(
            coil_factory=ConcatSurfaces(surface_factories=[coil_factory, coil_factory.copy()]),
            net_current=get_net_current(surf_plasma.file_path),
            tor_currents_w=np.zeros(2),
            pol_currents_w=np.zeros(2),
            constrain_tor_current=constrain_tor_current,
            distance_between_coils=distance_between_coils,
        )

    def __call__(self, **kwargs):
        pol_currents = self.net_current[0] * jax.nn.softmax(self.pol_currents_w)
        if self.constrain_tor_current:
            tor_currents = self.net_current[1] * jax.nn.softmax(self.tor_currents_w)
        else:
            tor_currents = self.tor_currents_w * 1e7
        fac = self.coil_factories.surface_factories
        for i in range(len(fac)):
            fac[i].surface_factories[1].current.net_currents = np.array([pol_currents[i], tor_currents[i]])

        fac[1].surface_factories[0].minor_radius = (
            fac[0].surface_factories[0].minor_radius + self.distance_between_coils
        )

        return self.coil_factory(**kwargs)

    def get_trainable_params(self):
        return {
            **dict(tor_currents_w=self.tor_currents_w, pol_currents_w=self.pol_currents_w),
            **self.coil_factory.get_trainable_params(),
        }

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            if k in dir(self):
                setattr(self, k, v)
        kwargs = {k: v for k, v in kwargs.items() if k not in ["tor_currents_w", "pol_currents_w"]}
        self.coil_factory.update_params(**kwargs)
