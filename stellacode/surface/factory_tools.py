import collections
import typing as tp

from stellacode import np
from stellacode.tools.rotate_n_times import RotateNTimes

from .abstract_surface import AbstractBaseFactory, IntegrationParams, Surface
from .coil_surface import CoilFactory
from .current import AbstractCurrent


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

    def __call__(self, surface: Surface, **kwargs):
        """ """

        integration_par = IntegrationParams.sum_toroidally([surface.integration_par] * self.rotate_n.n_rot)
        grids = integration_par.get_uvgrid()
        kwargs = dict(
            integration_par=integration_par,
            grids=grids,
        )
        for k in surface.field_keys:
            stack_dim = None
            if k in dir(surface):
                if k == "j_surface":
                    stack_dim = 1
                val = getattr(surface, k)
                if val is not None:
                    kwargs[k] = self.rotate_n(val, stack_dim=stack_dim)

            if "current_op" in dir(surface) and getattr(surface, "current_op") is not None:
                assert "phi_mn" in dir(surface) and getattr(surface, "phi_mn") is not None
                if self.different_currents:
                    kwargs["current_op"] = self._get_current_op(surface.current_op)
                    kwargs["phi_mn"] = self._get_phi_mn(surface.phi_mn)
                else:
                    kwargs["current_op"] = np.concatenate([surface.current_op] * self.rotate_n.n_rot, axis=2)
                    kwargs["phi_mn"] = surface.phi_mn

            for k in ["net_currents"]:
                if k in dir(surface) and getattr(surface, k) is not None:
                    kwargs[k] = getattr(surface, k)

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
        # TODO: treat constant currents just like others see issue #12
        blocks = np.concatenate(blocks, axis=2)
        return np.concatenate((np.concatenate([single_curent_op[:2]] * len(inner_blocks), axis=2), blocks), axis=0)

    def _get_phi_mn(self, phi_mn):
        return np.concatenate([phi_mn[:2]] + [phi_mn[2:]] * self.rotate_n.n_rot)


class ConcatSurfaces(AbstractBaseFactory):
    """
    Apply a list of surface factories and concatenates the resulting surfaces along the toroidal dimensions.
    """

    surface_factories: tp.List[AbstractBaseFactory]

    def __call__(self, surface: tp.Optional[Surface] = None, deg: int = 2, **kwargs):
        surfaces = []
        for surface_factory in self.surface_factories:
            if surface is None:
                surfaces.append(surface_factory(**kwargs))
            else:
                surfaces.append(surface_factory(surface, **kwargs))

        integration_par = IntegrationParams.sum_toroidally([surf.integration_par for surf in surfaces])
        grids = integration_par.get_uvgrid()
        kwargs = dict(
            integration_par=integration_par,
            grids=grids,
        )
        for k in surfaces[0].field_keys:
            if k in dir(surfaces[0]) and getattr(surfaces[0], k) is not None:
                kwargs[k] = np.concatenate([getattr(surface, k) for surface in surfaces], axis=1)

        if "current_op" in dir(surfaces[0]) and getattr(surfaces[0], "current_op") is not None:
            assert all(["current_op" in dir(surf) and getattr(surf, "current_op") is not None for surf in surfaces])
            kwargs["current_op"] = self._get_current_op([surf.current_op for surf in surfaces])
            assert all(["phi_mn" in dir(surf) and getattr(surf, "phi_mn") is not None for surf in surfaces])
            kwargs["phi_mn"] = self._get_phi_mn([surf.phi_mn for surf in surfaces])


        for k in ["net_currents"]:
            if k in dir(surfaces[0]) and getattr(surfaces[0], k) is not None:
                assert all([k in dir(surf) and getattr(surf, k) is not None for surf in surfaces])
                # TODO: We should be able to treat each net_currents as a separate entity.
                kwargs[k] = sum([getattr(surf, k) for surf in surfaces])

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

    def _get_current_op(self, single_curent_ops):
        zeros_blocks = [np.zeros_like(op[2:]) for op in single_curent_ops]
        num_ops = len(single_curent_ops)
        blocks = []
        for i in range(len(single_curent_ops)):
            blocks.append(np.concatenate(zeros_blocks[:i] + [single_curent_ops[i][2:]] + zeros_blocks[i + 1 :], axis=0))

        # This is a hack because the status of the first two coefficients is
        # special (constant currents not regressed)
        # TODO: treat constant currents just like others see issue #12
        blocks = np.concatenate(blocks, axis=2)
        return np.concatenate((np.concatenate([op[:2] for op in single_curent_ops], axis=2), blocks), axis=0)

    def _get_phi_mn(self, phi_mns):
        return np.concatenate([phi_mns[0][:2]] + [phi[2:] for phi in phi_mns])

class Sequential(AbstractBaseFactory):
    """
    Apply a list of surface factories in the given order
    """

    surface_factories: tp.List[AbstractBaseFactory]

    def __call__(self, surface: tp.Optional[Surface] = None, **kwargs):
        """ """

        for surface_factory in self.surface_factories:
            if surface is None:
                surface = surface_factory(**kwargs)
            else:
                surface = surface_factory(surface, **kwargs)

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


def rotate_coil(
    current: AbstractCurrent,
    nfp: int,
    num_surf_per_period: int = 1,
    continuous_current_in_period: bool = False,
    build_coils: bool = False,
):
    rot_common_current = RotatedSurface(
        rotate_n=RotateNTimes(angle=2 * np.pi / (num_surf_per_period * nfp), max_num=num_surf_per_period),
        different_currents=not continuous_current_in_period,
    )
    rot_nfp = RotatedSurface(rotate_n=RotateNTimes.from_nfp(nfp))
    coil_factory = CoilFactory(current=current, build_coils=build_coils)

    if continuous_current_in_period:
        return Sequential(surface_factories=[rot_common_current, coil_factory, rot_nfp])
    else:
        return Sequential(surface_factories=[coil_factory, rot_common_current, rot_nfp])
