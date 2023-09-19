import typing as tp
from typing import Any

from jax.typing import ArrayLike
from pydantic import BaseModel, Extra, Field

from stellacode import mu_0_fac, np
from stellacode.tools import biot_et_savart, biot_et_savart_op
from stellacode.tools.utils import get_min_dist

from .abstract_surface import AbstractSurface
from .current import AbstractCurrent
from stellacode.tools.laplace_force import laplace_force


class CoilSurface(AbstractSurface):
    """A class used to:
    * represent an abstract surfaces
    * computate of the magnetic field generated by a current carried by the surface.
    * visualize surfaces
    """

    surface: AbstractSurface
    current: AbstractCurrent
    current_op: tp.Optional[ArrayLike] = None

    class Config:
        arbitrary_types_allowed = True

    def __getattribute__(self, name: str) -> Any:
        if name in [
            "integration_par",
            "grids",
        ]:
            return getattr(self.surface, name)
        else:
            return super().__getattribute__(name)

    @classmethod
    def from_config(cls, config):
        from .imports import get_current_potential, get_cws_grid

        current = get_current_potential(config)
        surface = get_cws_grid(config)

        return cls(surface=surface, current=current)

    @classmethod
    def from_surface(cls, surface: AbstractSurface, current):
        return cls(**dict(surface), surface=surface, current=current)

    def _set_curent_op(self):
        self.current_op = self.current._get_matrix_from_grid(self.grids)

    def get_trainable_params(self):
        return {**self.current.get_trainable_params(), **self.surface.get_trainable_params()}

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            assert not (k in type(self.surface).__fields__ and k in type(self.current).__fields__)
            if k in type(self.surface).__fields__:
                setattr(self.surface, k, v)
            elif k in type(self.current).__fields__:
                setattr(self.current, k, v)

        self.compute_surface_attributes(deg=2)

    def get_current_basis_dot_prod(self):
        lu, lv = self.ds.shape
        return np.einsum(
            "oija,ijda,ijdk,pijk,ij->op",
            self.current_op,
            self.jac_xyz,
            self.jac_xyz,
            self.current_op,
            1 / self.ds,
            optimize=True,
        ) / (
            lu * lv
        )  # why isn't this dudv?

    def get_j_3D(self, phi_mn=None, scale_by_ds: bool = True):
        # phi_mn is a vector containing the components of the best scalar current potential.
        # The real surface current is given by :
        if phi_mn is None:
            phi_mn = self.current.get_phi_mn()
        if scale_by_ds:
            return np.einsum("oijk,ijdk,ij,o->ijd", self.current_op, self.jac_xyz, 1 / self.ds, phi_mn)
        else:
            return np.einsum("oijk,ijdk,o->ijd", self.current_op, self.jac_xyz, phi_mn)

    def get_grad_s_j_3D(self, phi_mn=None):
        if phi_mn is None:
            phi_mn = self.current.get_phi_mn()

        return np.einsum("oijk,ijdke,ij,o->ijde", self.current_op, self.hess_xyz, 1 / self.ds, phi_mn)

    def get_b_field(
        self,
        xyz_plasma: ArrayLike,
        plasma_normal: tp.Optional[ArrayLike] = None,
    ):
        return mu_0_fac * biot_et_savart(
            xyz_plasma=xyz_plasma,
            xyz_coil=self.xyz,
            j_3d=self.get_j_3D(scale_by_ds=False),
            dudv=self.dudv,
            plasma_normal=plasma_normal,
        )

    def get_b_field_op(
        self, xyz_plasma: ArrayLike, plasma_normal: tp.Optional[ArrayLike] = None, scale_by_mu0: bool = False
    ):
        bs_op = biot_et_savart_op(
            xyz_plasma=xyz_plasma,
            xyz_coil=self.xyz,
            surface_current=self.current_op,
            jac_xyz_coil=self.jac_xyz,
            dudv=self.dudv,
            plasma_normal=plasma_normal,
        )
        if scale_by_mu0:
            bs_op *= mu_0_fac

        return bs_op

    def naive_laplace_force(self, epsilon: float = 1e-3):
        j_3d = self.get_j_3D()

        xyz_ext = self.xyz + epsilon * self.normal_unit
        xyz_int = self.xyz - epsilon * self.normal_unit

        b_avg = self.get_b_field(xyz_ext) + self.get_b_field(xyz_int)

        return 0.5 * np.cross(j_3d, b_avg)

    def laplace_force(self):
        return laplace_force(
            j_3d=self.get_j_3D(),
            xyz=self.xyz,
            normal_unit=self.normal_unit,
            ds=self.ds,
            g_up_map=self.get_g_upper_basis(),
            num_tor_symmetry=self.num_tor_symmetry,
            du=self.du,
            dv=self.dv,
        )

    def imshow_j(self, phi_mn):
        import matplotlib.pyplot as plt

        j_3d = self.get_j_3D(phi_mn)
        norm_j = np.linalg.norm(j_3d, axis=-1)
        plt.imshow(norm_j, cmap="seismic")

    def get_j_surface(self, phi_mn=None):
        """
        It is the contravariant components of the current: J^i
        """
        if phi_mn is None:
            phi_mn = self.current.get_phi_mn()
        return np.einsum("oijk,o->ijk", self.current_op, phi_mn)

    def plot_j_surface(self, phi_mn=None, num_prec: int = 2, ax=None):
        j_surface = self.get_j_3D(phi_mn)
        return self.surface.plot_2d_field(j_surface[:, : self.surface.nbpts[1]], num_prec=num_prec, ax=ax)
