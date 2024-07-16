import typing as tp

import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from jax import Array
from jax.typing import ArrayLike

from .abstract_surface import IntegrationParams
from .coil_surface import CoilFactory
from .current import AbstractCurrent, Current, CurrentZeroTorBC
from .cylindrical import CylindricalSurface
from .factory_tools import rotate_coil
from .fourier import FourierSurfaceFactory, FourierSurface
from .tore import ToroidalSurface

from .abstract_surface import AbstractBaseFactory
from .cylindrical import CylindricalSurface
from .factory_tools import (
    ConcatSurfaces,
    RotatedSurface,
    RotateNTimes,
    Sequential,
    rotate_coil,
)
from .imports import get_net_current
from .tore import ToroidalSurface
from .utils import fit_to_surface
from stellacode.tools.vmec import VMECIO

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
    """
    A coil factory with a number of cylinders

    Args:
        * coil_factory: Sequential coil factory
        * ncp: number of cylinders
    """
    coil_factory: AbstractBaseFactory
    ncp: int

    @classmethod
    def from_plasma(
        cls,
        surf_plasma: FourierSurfaceFactory,
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
    ) -> "WrappedCoil":
        """
        Create a WrappedCoil object from a plasma surface.

        Args:
            surf_plasma: FourierSurfaceFactory object representing plasma surface.
            surf_type: Type of surface to create. Must be "cylindrical" or "toroidal".
            n_harmonics: Number of Fourier harmonics to use.
            factor: Factor to upscale the number of points in the Fourier harmonics.
            rotate_diff_current: Number of cylinders per field period.
            make_joints: Whether to create joints between surfaces.
            common_current_on_each_rot: Whether to have a common current on each cylinder.
            axis_angle: Angle of rotation around the axis.
            distance: Distance between the plasma surface and the cylinder surface.
            sin_basis: Whether to use sine basis.
            cos_basis: Whether to use cosine basis.
            convex: Whether to use convex surface.
            match_surface: Whether to match the surface of the plasma.
            build_coils: Whether to build coils.

        Returns:
            WrappedCoil object representing the coil factory.
        """
        # Create the coil factory based on the surface type
        if surf_type == "cylindrical":
            coil_factory = get_pwc_surface(
                surf_plasma=surf_plasma,
                n_harmonics=n_harmonics,
                factor=factor,
                sin_basis=sin_basis,
                cos_basis=cos_basis,
                rotate_diff_current=rotate_diff_current,
                make_joints=make_joints,
                common_current_on_each_rot=common_current_on_each_rot,
                distance=distance,
                build_coils=build_coils,
                match_surface=match_surface,
                convex=convex,
                axis_angle=axis_angle,
            )
            ncp = surf_plasma.nfp * rotate_diff_current
        elif surf_type == "toroidal":
            coil_factory = get_toroidal_surface(
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
            ncp = 1
        else:
            raise NotImplementedError(
                "surf_type must be 'cylindrical' or 'toroidal'")

        # Create and return the WrappedCoil object
        return cls(
            coil_factory=coil_factory,
            ncp=ncp,
        )

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

    def _get_coil_factory(self):
        if isinstance(self.coil_factory.surface_factories[1].surface_factories[0], CoilFactory):
            return self.coil_factory.surface_factories[1].surface_factories[0]
        else:
            return self.coil_factory.surface_factories[1].surface_factories[1]

    def _get_current(self):
        return self._get_coil_factory().current

    def _get_base_surface(self):
        return self.coil_factory.surface_factories[0]

    def _get_radius(self):
        surf = self._get_base_surface()
        if isinstance(surf, ToroidalSurface):
            return surf.minor_radius
        else:
            return surf.radius


def get_toroidal_surface(
    surf_plasma: FourierSurfaceFactory,
    n_harmonics: int = 16,
    factor: int = 6,
    match_surface: bool = False,
    convex: bool = True,
    distance: float = 0.0,
    sin_basis: bool = True,
    cos_basis: bool = True,
    build_coils: bool = False,
) -> Sequential:
    """
    Generate a toroidal surface factory.

    Args:
        surf_plasma: FourierSurfaceFactory object representing plasma surface.
        n_harmonics: Number of Fourier harmonics to use.
        factor: Factor to upscale the number of points in the Fourier harmonics.
        match_surface: Whether to match the surface of the plasma.
        convex: Whether to use convex surface.
        distance: Minimal distance between the plasma surface and the toroidal surface.
        sin_basis: Whether to use sine basis.
        cos_basis: Whether to use cosine basis.
        build_coils: Whether to build coils.

    Returns:
        Sequential object representing the toroidal surface factory.
    """
    # Get net currents
    net_currents = get_net_current(surf_plasma.file_path)

    # Define current
    current = Current(
        num_pol=n_harmonics,
        num_tor=n_harmonics,
        sin_basis=sin_basis,
        cos_basis=cos_basis,
        net_currents=net_currents,
    )

    if match_surface:
        # Match surface of plasma
        tor_surf = surf_plasma.get_surface_envelope(
            num_coeff=10, convex=convex, limit=1000)
        tor_surf.update_params(minor_radius=tor_surf.minor_radius + distance)
    else:
        # Create toroidal surface
        minor_radius = surf_plasma.get_minor_radius(vmec=False)
        major_radius = surf_plasma.get_major_radius()
        tor_surf = ToroidalSurface(
            nfp=surf_plasma.nfp,
            major_radius=major_radius,
            minor_radius=minor_radius + distance,
            integration_par=current.get_integration_params(factor=factor),
        )

    # Create rotated coil
    coil_factory = rotate_coil(
        current=current,
        nfp=surf_plasma.nfp,
        num_surf_per_period=1,
        continuous_current_in_period=False,
        build_coils=build_coils,
    )

    # Create sequential surface factory
    return Sequential(
        surface_factories=[
            tor_surf,
            coil_factory,
        ]
    )


def cylinder_from_plasma(
        Sp: FourierSurface,
        integration_par: IntegrationParams,
        num_cyl: int,
        distance: float = 0.0,
        make_joints: bool = True,
        **kwargs,
) -> CylindricalSurface:
    """
    Create a cylindrical surface from a plasma surface.

    Args:
        Sp (FourierSurfaceFactory): Plasma surface factory.
        integration_par (IntegrationParams): Integration parameters.
        num_cyl (int): Number of cylinders per field period.
        distance (float, optional): Distance between the plasma surface and the cylinder surface. Defaults to 0.0.
        make_joints (bool, optional): Whether to create joints between surfaces. Defaults to True.
        **kwargs: Additional arguments for the surface envelope.

    Returns:
        Surface: Cylindrical surface.
    """

    # Create the cylindrical surface from the plasma surface
    # The plasma surface is first converted to a surface envelope
    # which is then used to create the cylindrical surface
    cylinder_surface = Sp.get_surface_envelope(
        num_cyl=num_cyl,
        **kwargs,
    )

    cylinder_surface.update_params(
        radius=cylinder_surface.radius+distance,
        integration_par=integration_par,
        make_joints=make_joints,
    )

    return cylinder_surface


def get_pwc_surface(
    surf_plasma: FourierSurfaceFactory,
    n_harmonics: int = 16,
    factor: int = 6,
    sin_basis: bool = True,
    cos_basis: bool = True,
    rotate_diff_current: int = 3,
    make_joints: bool = True,
    common_current_on_each_rot: bool = False,
    distance: float = 0.0,
    build_coils: bool = False,
    match_surface: bool = False,
    convex: bool = True,
    axis_angle: float = 0.0,
) -> Sequential:
    """
    Generate a PWC coil surface factory.

    Args:
        surf_plasma: FourierSurfaceFactory object representing plasma surface.
        n_harmonics: Number of Fourier harmonics to use for the current.
        factor: Factor to upscale the number of points in the Fourier harmonics.
        sin_basis: Whether to use sine basis for the current.
        cos_basis: Whether to use cosine basis for the current.        
        rotate_diff_current: Number of cylinders per field period.
        make_joints: Whether to create joints between cylinders.
        common_current_on_each_rot: Whether to have a common current on each cylinder.
        distance: Minimal distance between the plasma surface and the cylinders.
        build_coils: Whether to build coils.
        match_surface: Whether to fit to the surface of the plasma.
        convex: Whether to use the convex plasma envelope if match_surface is True.
        axis_angle: Angle of rotation around the magnetic axis if match_surface is False.

    Returns:
        Sequential object representing the PWC coil surface factory.
    """

    # Get net currents
    net_currents = get_net_current(surf_plasma.file_path)

    # Define current
    if common_current_on_each_rot:
        current: AbstractCurrent = Current(
            num_pol=n_harmonics,
            num_tor=n_harmonics,
            sin_basis=sin_basis,
            cos_basis=cos_basis,
            net_currents=net_currents,
        )
        center_vgrid = False
    else:
        current = CurrentZeroTorBC(
            num_pol=n_harmonics,
            num_tor=n_harmonics // rotate_diff_current,
            sin_basis=sin_basis,
            cos_basis=cos_basis,
            net_currents=net_currents / rotate_diff_current,
        )
        center_vgrid = True

    # Define integration parameters
    integration_par = IntegrationParams(
        num_points_u=n_harmonics * factor,
        num_points_v=n_harmonics * factor // rotate_diff_current,
        center_vgrid=center_vgrid,
    )

    # Define surface coil
    if match_surface:
        cyl_surf = cylinder_from_plasma(
            surf_plasma,
            integration_par,
            rotate_diff_current,
            distance,
            make_joints,
            num_coeff=10,
            convex=convex,
            limit=1000,
        )
    else:
        minor_radius = surf_plasma.get_minor_radius(vmec=False)
        major_radius = surf_plasma.get_major_radius()
        cyl_surf = CylindricalSurface(
            integration_par=integration_par,
            make_joints=make_joints,
            axis_angle=axis_angle,
            ncp=surf_plasma.nfp * rotate_diff_current,
            radius=minor_radius + distance,
            distance=major_radius,
        )

    coil_factory = rotate_coil(
        current=current,
        nfp=surf_plasma.nfp,
        num_surf_per_period=rotate_diff_current,
        continuous_current_in_period=common_current_on_each_rot,
        build_coils=build_coils,
    )

    # Define coil sequence
    pwc_surf = Sequential(
        surface_factories=[
            cyl_surf,
            coil_factory,
        ]
    )

    return pwc_surf


def get_original_cws(path_cws: str, path_plasma: str, n_harmonics: int = 16, factor: int = 6):
    nfp = VMECIO.from_grid(path_plasma).nfp
    cws = FourierSurfaceFactory.from_file(
        path_cws,
        integration_par=IntegrationParams(
            num_points_u=n_harmonics * factor,
            num_points_v=n_harmonics * factor,
        ),
        nfp=nfp,
    )

    cws = Sequential(
        surface_factories=[
            cws,
            rotate_coil(
                current=Current(num_pol=n_harmonics, num_tor=n_harmonics,
                                net_currents=get_net_current(path_plasma)),
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
            minor_radius = surf_plasma.get_minor_radius(vmec=False)
            major_radius = surf_plasma.get_major_radius()
            surface = CylindricalSurface(
                fourier_coeffs=fourier_coeffs,
                integration_par=IntegrationParams(
                    num_points_u=n_harmonics_u * factor, num_points_v=n_harmonics_v * factor, center_vgrid=True
                ),
                ncp=num_sym_by_cyl,
                radius=minor_radius + distance,
                distance=major_radius,
                axis_angle=angle * n,
            )

            coil_fac_ = Sequential(
                surface_factories=[
                    surface,
                    CoilFactory(
                        current=current, build_coils=False
                    ),  # only build_coils=True is allopwed because concat surface has no current_op implemented
                ]
            )
            surfaces.append(coil_fac_)

        coil_factory = Sequential(
            surface_factories=[
                ConcatSurfaces(surface_factories=surfaces),
                RotatedSurface(rotate_n=RotateNTimes(
                    angle=2 * np.pi / surf_plasma.nfp, max_num=surf_plasma.nfp)),
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
        pol_currents = self.net_current[0] * \
            jax.nn.softmax(self.pol_currents_w)
        if self.constrain_tor_current:
            tor_currents = self.net_current[1] * \
                jax.nn.softmax(self.tor_currents_w)
        else:
            tor_currents = self.tor_currents_w * 1e7

        coils = self.coil_factory.surface_factories[0].surface_factories
        for i in range(len(coils)):
            coils[i].surface_factories[1].current.net_currents = np.array(
                [pol_currents[i], tor_currents[i]])

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
        kwargs = {k: v for k, v in kwargs.items() if k not in [
            "tor_currents_w", "pol_currents_w"]}
        self.coil_factory.update_params(**kwargs)

    def plot_cross_section(self, **kwargs):
        coils = self.coil_factory.surface_factories[0].surface_factories
        fig, axes = plt.subplots(len(coils), subplot_kw={
                                 "projection": "polar"})
        for i in range(len(coils)):
            coils[i].surface_factories[0].plot_cross_section(
                ax=axes[i], **kwargs)
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

    def _get_base_surfaces(self):
        return [surf.surface_factories[0] for surf in self.coil_factory.surface_factories[0].surface_factories]

    def _get_base_coils(self):
        return [surf.surface_factories[0] for surf in self.coil_factory.surface_factories[0].surface_factories]
