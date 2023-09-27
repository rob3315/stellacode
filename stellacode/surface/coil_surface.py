import typing as tp
from typing import Any

from jax.typing import ArrayLike
from pydantic import BaseModel, Extra, Field

from stellacode import mu_0_fac, np
from stellacode.tools import biot_et_savart, biot_et_savart_op
from stellacode.tools.utils import get_min_dist

from .abstract_surface import Surface, AbstractSurfaceFactory, AbstractBaseFactory
from .current import AbstractCurrent
from stellacode.tools.laplace_force import laplace_force


class CoilFactory(AbstractBaseFactory):
    current: AbstractCurrent

    @classmethod
    def from_config(cls, config):
        from .imports import get_current_potential, get_cws_grid

        current = get_current_potential(config)
        surface = get_cws_grid(config)

        return cls(surface=surface, current=current)

    def get_trainable_params(self):
        return self.current.get_trainable_params()

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            if k in type(self.current).__fields__:
                setattr(self.current, k, v)

    def __call__(self, surface: Surface, key=None):
        """compute surface elements used in the shape optimization up
        to degree deg
        deg is 0,1 or 2"""

        surf = CoilSurface.from_surface(surface)
        surf.current_op = self.current._get_matrix_from_grid(surf.grids, surf.integration_par.max_val_v)

        surf.j_surface = surf.get_j_surface(self.current.get_phi_mn())
        surf.net_currents = self.current.net_currents
        surf.phi_mn = self.current.get_phi_mn()
        return surf


class CoilSurface(Surface):
    """A class used to:
    * represent an abstract surfaces
    * computate of the magnetic field generated by a current carried by the surface.
    * visualize surfaces
    """

    current_op: tp.Optional[ArrayLike] = None
    j_surface: tp.Optional[ArrayLike] = None
    net_currents: tp.Optional[ArrayLike] = None
    num_tor_symmetry: tp.Optional[ArrayLike] = None
    phi_mn: tp.Optional[ArrayLike] = None

    @classmethod
    def from_surface(cls, surface: Surface):
        return cls(**{k: v for k, v in dict(surface).items() if k in cls.__fields__.keys()})

    @property
    def field_keys(self):
        return super().field_keys + ["j_surface"]

    def get_current_basis_dot_prod(self):
        lu, lv = self.ds.shape
        cb = np.einsum("oija,ijda,ijdk,pijk->op", self.current_op, self.jac_xyz, self.jac_xyz, self.current_op)
        js = np.einsum("oija,ijda->oijd", self.current_op, self.jac_xyz)
        np.einsum("oijd,pijd->op", js, js)

        cb = (
            np.einsum(
                "oija,ijda,ijdk,pijk,ij->op",
                self.current_op,
                self.jac_xyz,
                self.jac_xyz,
                self.current_op,
                1 / self.ds,
                optimize=True,
            )
            * self.dudv
        )

        return cb

    def get_j_3D(self, phi_mn=None, scale_by_ds: bool = True):
        # phi_mn is a vector containing the components of the best scalar current potential.
        # The real surface current is given by :
        # if phi_mn is None:
        #     phi_mn = self.current.get_phi_mn()

        # if scale_by_ds:
        #     return np.einsum("oijk,ijdk,ij,o->ijd", self.current_op, self.jac_xyz, 1 / self.ds, phi_mn)
        # else:
        #     return np.einsum("oijk,ijdk,o->ijd", self.current_op, self.jac_xyz, phi_mn)

        if phi_mn is None:
            if self.j_surface is not None:
                j_surface = self.j_surface
            else:
                j_surface = self.get_j_surface()
            if scale_by_ds:
                return np.einsum("ijk,ijdk,ij->ijd", j_surface, self.jac_xyz, 1 / self.ds)
            else:
                return np.einsum("ijk,ijdk->ijd", j_surface, self.jac_xyz)
        else:
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
        bf = mu_0_fac * biot_et_savart(
            xyz_plasma=xyz_plasma,
            xyz_coil=self.xyz,
            j_3d=self.get_j_3D(scale_by_ds=False),
            dudv=self.dudv,
            plasma_normal=plasma_normal,
        )

        return bf

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

    def laplace_force(self, num_tor_symmetry: int, phi_mn=None):
        return laplace_force(
            j_3d=self.get_j_3D(phi_mn=phi_mn),
            xyz=self.xyz,
            normal_unit=self.normal_unit,
            ds=self.ds,
            g_up_map=self.get_g_upper_basis(),
            num_tor_symmetry=num_tor_symmetry,
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
            phi_mn = self.phi_mn
        return np.einsum("oijk,o->ijk", self.current_op, phi_mn)

    def plot_j_surface(self, phi_mn=None, num_prec: int = 2, ax=None):
        j_surface = self.get_j_3D(phi_mn)
        return self.plot_2d_field(j_surface[:, : self.nbpts[1]], num_prec=num_prec, ax=ax)
