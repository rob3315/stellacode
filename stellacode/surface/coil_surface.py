import typing as tp

from jax.typing import ArrayLike

from stellacode import mu_0_fac, np
from stellacode.tools import biot_et_savart, biot_et_savart_op

from .abstract_surface import Surface, AbstractBaseFactory
from .current import AbstractCurrent
from stellacode.tools.laplace_force import laplace_force
import matplotlib.pyplot as plt


class CoilFactory(AbstractBaseFactory):
    """
    Build a coil from a surface and a current class

    Args:
        * current: current class computing a current operator on a given surface grid.
    """

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

    def __call__(self, surface: Surface, **kwargs):
        surf = CoilSurface.from_surface(surface)
        surf.current_op = self.current(surf.grids, surf.integration_par.max_val_v)

        surf.j_surface = surf.get_j_surface(self.current.get_phi_mn())
        surf.net_currents = self.current.net_currents
        surf.phi_mn = self.current.get_phi_mn()
        return surf


class CoilSurface(Surface):
    """Represent a coil

    Args:
        * current_op: tensor for computing the 2D surface current from the current weights
        * j_surface: contravariant components of the current: J^i
        * net_currents: net poloidal and toroidal currents
        * nfp: number of toroidal periods (Note: should be in a specialized class instead of a general coil class)
        * phi_mn: current weights
    """

    current_op: tp.Optional[ArrayLike] = None
    j_surface: tp.Optional[ArrayLike] = None
    net_currents: tp.Optional[ArrayLike] = None
    nfp: tp.Optional[ArrayLike] = None
    phi_mn: tp.Optional[ArrayLike] = None

    @classmethod
    def from_surface(cls, surface: Surface):
        return cls(**{k: v for k, v in dict(surface).items() if k in cls.__fields__.keys()})

    @property
    def field_keys(self):
        return super().field_keys + ["j_surface"]

    def get_current_basis_dot_prod(self):
        """Compute the scalar product matrix of the current basis functions"""
        return (
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

    def get_j_3D(self, phi_mn=None, scale_by_ds: bool = True):
        """Compute the 3D current onto the surface"""


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
        """
        Compute the magnetic field.

        Args:
            * xyz_plasma: position on which the magnetic field is computed.
            * plasma_normal: the returned magnetic field is the scalar product of the magnetic field by this vector.

        """
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
        """
        Compute the magnetic field operator

        Args:
            * xyz_plasma: position on which the magnetic field is computed.
            * plasma_normal: the returned magnetic field is the scalar product of the magnetic field by this vector.
            * scale_by_mu0: the result is multiplied by mu_0

        """
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
        """
        Naive computation of the Laplace force

        Args:
            * epsilon: distance at which the magnetic field is computed. Should be larger than the resolution of
                the grid on the surface otherwise the computation may be very inaccurate.
        """
        j_3d = self.get_j_3D()

        xyz_ext = self.xyz + epsilon * self.normal_unit
        xyz_int = self.xyz - epsilon * self.normal_unit

        b_avg = self.get_b_field(xyz_ext) + self.get_b_field(xyz_int)

        return 0.5 * np.cross(j_3d, b_avg)

    def laplace_force(self, nfp: int, phi_mn=None):
        """
        Return the Laplace force
        """
        return laplace_force(
            j_3d=self.get_j_3D(phi_mn=phi_mn),
            xyz=self.xyz,
            normal_unit=self.normal_unit,
            ds=self.ds,
            g_up_map=self.get_g_upper_basis(),
            nfp=nfp,
            du=self.du,
            dv=self.dv,
        )

    def imshow_j(self, phi_mn):
        """Show the current surface density"""
        j_3d = self.get_j_3D(phi_mn)
        norm_j = np.linalg.norm(j_3d, axis=-1)
        plt.imshow(norm_j, cmap="seismic")

    def get_j_surface(self, phi_mn=None):
        """
        Contravariant components of the current: J^i
        """
        if phi_mn is None:
            phi_mn = self.phi_mn
        return np.einsum("oijk,o->ijk", self.current_op, phi_mn)

    def plot_j_surface(self, phi_mn=None, num_prec: int = 2, ax=None):
        j_surface = self.get_j_3D(phi_mn)
        return self.plot_2d_field(j_surface[:, : self.nbpts[1]], num_prec=num_prec, ax=ax)
