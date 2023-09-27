import collections

from stellacode import np
from stellacode.tools.rotate_n_times import RotateNTimes
from .abstract_surface import AbstractSurfaceFactory, Surface, AbstractBaseFactory, IntegrationParams
from .coil_surface import CoilSurface, CoilFactory

from .utils import cartesian_to_toroidal
import typing as tp
import equinox as eqx
from .current import AbstractCurrent
from pydantic import BaseModel, Extra


class RotatedSurface(AbstractBaseFactory):
    """ 
    Rotate and duplicate a surface N times

    Args:
        * rotate_n: rotator
        * different_current: use different current pattern for each surface 
            Warning:only used for the current operator, 
            it will not apply to the currents.
    """

    rotate_n: RotateNTimes = RotateNTimes(1)
    different_currents: bool = False

    def __call__(self, surface: Surface = Surface(), key=None):
        """ """

        integration_par = IntegrationParams.sum_toroidally([surface.integration_par] * self.rotate_n.n_rot)
        grids = integration_par.get_uvgrid()
        kwargs = dict(
            integration_par=integration_par,
            grids=grids,
        )
        for k in [
            "xyz",
            "jac_xyz",
            "normal",
            "ds",
            "normal_unit",
            "hess_xyz",
            "principle_max",
            "principle_min",
            "j_surface",
        ]:
            stack_dim = None
            if k in dir(surface):
                if k == "j_surface":
                    stack_dim = 1
                val = getattr(surface, k)
                if val is not None:
                    kwargs[k] = self.rotate_n(val, stack_dim=stack_dim)
                print(k, kwargs[k].shape)

            if "current_op" in dir(surface) and getattr(surface, "current_op") is not None:
                if self.different_currents:
                    kwargs["current_op"] = self._get_current_op(surface.current_op)
                else:
                    kwargs["current_op"] = np.concatenate([surface.current_op] * self.rotate_n.n_rot, axis=2)

            for k in ["net_currents", "phi_mn"]:
                if k in dir(surface) and getattr(surface, k) is not None:
                    kwargs[k] = surface.net_currents

        return type(surface)(**kwargs)

    def _get_current_op(self, single_curent_op):
        current_op_ = single_curent_op[2:]

        inner_blocks = collections.deque([current_op_] + [np.zeros_like(current_op_)] * (self.rotate_n.n_rot - 1))
        blocks = []
        for _ in range(len(inner_blocks)):
            blocks.append(np.concatenate(inner_blocks, axis=0))
            inner_blocks.rotate(1)

        # This is a hack because the status of the first two coefficients is
        # special (constant currents not regressed)
        blocks = np.concatenate(blocks, axis=2)
        return np.concatenate((np.concatenate([single_curent_op[:2]] * len(inner_blocks), axis=2), blocks), axis=0)


class ConcatSurfaces(AbstractBaseFactory):
    """ """

    surface_factories: tp.List[AbstractBaseFactory]

    def __call__(self, surface: Surface = Surface(), key=None):
        """ """

        surfaces = []
        for surface_factory in self.surface_factories:
            surfaces.append(surface_factory())

        integration_par = IntegrationParams.sum_toroidally([surf.integration_par for surf in surfaces])
        grids = integration_par.get_uvgrid()
        kwargs = dict(
            integration_par=integration_par,
            grids=grids,
        )
        for k in [
            "xyz",
            "jac_xyz",
            "normal",
            "ds",
            "normal_unit",
            "hess_xyz",
            "principle_max",
            "principle_min",
            "j_surface",
        ]:
            if k in dir(surfaces[0]) and getattr(surfaces[0], k) is not None:
                kwargs[k] = np.concatenate([getattr(surface, k) for surface in surfaces], axis=1)

        return type(surfaces[0])(**kwargs)

    def get_trainable_params(self):
        params = {}
        for i, surface_factory in enumerate(self.surface_factories):
            params = {**params, **{f"{i}.{k}": v for k, v in surface_factory.get_trainable_params().items()}}
        return params

    def update_params(self, **kwargs):
        for i in range(len(self.surface_factories)):
            _kwargs = {".".join(k.split(".")[1:]): v for k, v in kwargs.items() if int(k.split(".")[0]) == i}
            self.surface_factories[i].update_params(**_kwargs)

    def __getattribute__(self, name: str):
        if name in ["integration_par", "grids", "dudv"]:
            return getattr(self.surfaces[0], name)
        else:
            return super().__getattribute__(name)


class Sequential(AbstractBaseFactory):
    """ """

    surface_factories: tp.List[AbstractBaseFactory]

    def __call__(self, surface: Surface = Surface(), key=None):
        """ """

        for surface_factory in self.surface_factories:
            surface = surface_factory(surface)

        return surface

    def get_trainable_params(self):
        params = {}
        for i, surface in enumerate(self.surface_factories):
            params = {**params, **{f"{i}.{k}": v for k, v in surface.get_trainable_params().items()}}
        return params

    def update_params(self, **kwargs):
        for i in range(len(self.surface_factories)):
            _kwargs = {".".join(k.split(".")[1:]): v for k, v in kwargs.items() if int(k.split(".")[0]) == i}
            self.surface_factories[i].update_params(**_kwargs)

        # self.compute_surface_attributes(deg=2)

    def __getattribute__(self, name: str):
        if name in ["integration_par", "grids", "dudv"]:
            return getattr(self.surfaces[0], name)
        else:
            return super().__getattribute__(name)


def rotate_coil(
    current: AbstractCurrent,
    num_tor_symmetry: int,
    num_surf_per_period: int = 1,
    continuous_current_in_period: bool = False,
):
    rot_common_current = RotatedSurface(
        rotate_n=RotateNTimes(angle=2 * np.pi / (num_surf_per_period * num_tor_symmetry), max_num=num_surf_per_period),
        different_currents=not continuous_current_in_period
    )
    rot_nfp = RotatedSurface(rotate_n=RotateNTimes.from_nfp(num_tor_symmetry))
    coil_factory = CoilFactory(current=current)
    if continuous_current_in_period:
        return Sequential(surface_factories=[rot_common_current, coil_factory, rot_nfp])
    else:
        return Sequential(surface_factories=[coil_factory, rot_common_current, rot_nfp])


class RotatedCoil(AbstractBaseFactory):
    """A class used to:
    * represent an abstract surfaces
    * computate of the magnetic field generated by a current carried by the surface.
    * visualize surfaces
    """

    current: AbstractCurrent
    num_tor_symmetry: int = 1
    rotate_diff_current: int = 1
    common_current_on_each_rot: bool = False
    rotate_n: RotateNTimes = RotateNTimes(1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rotate_n = RotateNTimes.from_nfp(self.num_tor_symmetry * self.rotate_diff_current)

    def get_trainable_params(self):
        return self.current.get_trainable_params()

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            if k in type(self.current).__fields__:
                setattr(self.current, k, v)

    @property
    def dudv(self):
        if self.common_current_on_each_rot:
            return self.surface.dudv / self.rotate_diff_current
        else:
            return self.surface.dudv

    def get_num_rotations(self):
        return self.num_tor_symmetry * self.rotate_diff_current

    def _get_curent_op(self, grids):
        if self.common_current_on_each_rot:
            gridu, gridv = grids
            gridu = np.concatenate([gridu] * self.rotate_diff_current, axis=1)
            rd = self.rotate_diff_current
            # this scaling is necessary because the Current class expect the grid to
            # always be from 0 to 1.
            gridv = np.concatenate([(i + gridv) / rd for i in range(rd)], axis=1)
            blocks = self.current._get_matrix_from_grid((gridu, gridv))
            # this is because the current potential derivative vs v is scaled by rd
            # when v is scaled by 1/rd
            blocks[..., -1] *= rd

        else:
            curent_op = self.current._get_matrix_from_grid(grids)
            current_op_ = curent_op[2:]

            inner_blocks = collections.deque(
                [current_op_] + [np.zeros_like(current_op_)] * (self.rotate_diff_current - 1)
            )
            blocks = []
            for _ in range(len(inner_blocks)):
                blocks.append(np.concatenate(inner_blocks, axis=0))
                inner_blocks.rotate(1)

            # This is a hack because the status of the first two coefficients is
            # special (constant currents not regressed)
            blocks = np.concatenate(blocks, axis=2)
            blocks = np.concatenate((np.concatenate([curent_op[:2]] * len(inner_blocks), axis=2), blocks), axis=0)

        return np.concatenate([blocks] * self.num_tor_symmetry, axis=2)

    def __call__(self, surface: Surface):
        """ """

        current_op = self._get_curent_op(grids=surface.grids)

        integration_par = IntegrationParams.sum_toroidally([surface.integration_par] * self.rotate_n.n_rot)
        grids = integration_par.get_uvgrid()
        kwargs = dict(
            integration_par=integration_par,
            grids=grids,
        )
        for k in [
            "xyz",
            "jac_xyz",
            "normal",
            "ds",
            "normal_unit",
            "hess_xyz",
            "principle_max",
            "principle_min",
        ]:
            val = getattr(surface, k)
            if val is not None:
                kwargs[k] = self.rotate_n(val)

        coil_surf = CoilSurface(**kwargs)

        coil_surf.current_op = current_op
        coil_surf.net_currents = self.current.net_currents
        coil_surf.num_tor_symmetry = self.num_tor_symmetry

        coil_surf.phi_mn = self.current.get_phi_mn()
        if self.common_current_on_each_rot:
            coil_surf.integration_par = coil_surf.integration_par.copy(
                update=dict(num_points_v=surface.integration_par.num_points_v * self.rotate_diff_current),
            )
        return coil_surf

    def get_j_surface(self, phi_mn=None):
        # if phi_mn is None:
        #     phi_mn = self.phi_mn
        return np.einsum("oijk,o->ijk", self.current_op, self.current.get_phi_mn())

    def cartesian_to_toroidal(self):
        try:
            major_radius = self.surface.distance
        except:
            major_radius = self.surface.major_radius
        return cartesian_to_toroidal(xyz=self.xyz, tore_radius=major_radius, height=0.0)
