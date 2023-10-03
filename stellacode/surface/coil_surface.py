import typing as tp

from jax.typing import ArrayLike
from jax import Array

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
        * build_coils: if True, returns a coilOperator otherwise a CoilSurface
    """

    current: AbstractCurrent
    build_coils: bool = False

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
        coil_op = CoilOperator.from_surface(
            surface=surface,
            current_op=self.current(surface.grids, surface.integration_par.max_val_v),
            net_currents=self.current.net_currents,
        )

        if self.build_coils:
            return coil_op.get_coil(phi_mn=self.current.get_phi_mn())
        else:
            return coil_op


class CoilOperator(Surface):
    """Represent a coil operator

    Args:
        * current_op: tensor for computing the 2D surface current from the current weights
        * net_currents: net poloidal and toroidal currents
        * nfp: number of toroidal periods (Note: should be in a specialized class instead of a general coil class)
    """

    current_op: ArrayLike
    net_currents: tp.Optional[ArrayLike] = None

    @classmethod
    def from_surface(cls, surface: Surface, current_op: ArrayLike, net_currents: tp.Optional[ArrayLike] = None):
        dict_ = {k: v for k, v in dict(surface).items() if k in cls.__fields__.keys()}
        dict_["current_op"] = current_op
        dict_["net_currents"] = net_currents
        return cls(**dict_)

    def get_coil(self, phi_mn):
        dict_ = {k: v for k, v in dict(self).items() if k != "current_op"}
        dict_["j_surface"] = self.get_j_surface(phi_mn)
        dict_["j_3d"] = self.get_j_3d(phi_mn)

        return CoilSurface(**dict_)

    def get_j_surface(self, phi_mn):
        """
        Contravariant components of the current: J^i
        """
        return np.einsum("oijk,o->ijk", self.current_op, phi_mn)

    def get_j_3d(self, phi_mn, scale_by_ds: bool = True):
        """Compute the 3D current onto the surface"""
        if scale_by_ds:
            return np.einsum("oijk,ijdk,ij,o->ijd", self.current_op, self.jac_xyz, 1 / self.ds, phi_mn)
        else:
            return np.einsum("oijk,ijdk,o->ijd", self.current_op, self.jac_xyz, phi_mn)

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


class CoilSurface(Surface):
    """Represent a coil

    Args:
        * current_op: tensor for computing the 2D surface current from the current weights
        * j_surface: contravariant components of the current: J^i
        * net_currents: net poloidal and toroidal currents
        * nfp: number of toroidal periods (Note: should be in a specialized class instead of a general coil class)
        * phi_mn: current weights
    """

    j_surface: Array
    j_3d: Array
    net_currents: tp.Optional[Array]

    @classmethod
    def from_surface(cls, surface: Surface):
        return cls(**{k: v for k, v in dict(surface).items() if k in cls.__fields__.keys()})

    @property
    def field_keys(self):
        return super().field_keys + ["j_surface", "j_3d"]

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
            j_3d=self.j_3d * self.ds[..., None],
            dudv=self.dudv,
            plasma_normal=plasma_normal,
        )

        return bf

    def naive_laplace_force(self, epsilon: float = 1e-3):
        """
        Naive computation of the Laplace force

        Args:
            * epsilon: distance at which the magnetic field is computed. Should be larger than the resolution of
                the grid on the surface otherwise the computation may be very inaccurate.
        """
        j_3d = self.j_3d

        xyz_ext = self.xyz + epsilon * self.normal_unit
        xyz_int = self.xyz - epsilon * self.normal_unit

        b_avg = self.get_b_field(xyz_ext) + self.get_b_field(xyz_int)

        return 0.5 * np.cross(j_3d, b_avg)

    def laplace_force(self, nfp: int):
        """
        Return the Laplace force
        """
        return laplace_force(
            j_3d=self.j_3d,
            xyz=self.xyz,
            normal_unit=self.normal_unit,
            ds=self.ds,
            g_up_map=self.get_g_upper_basis(),
            nfp=nfp,
            du=self.du,
            dv=self.dv,
        )

    def imshow_j(self):
        """Show the current surface density"""
        plt.imshow(np.linalg.norm(self.j_3d, axis=-1), cmap="seismic")

    def plot_j_surface(self, num_prec: int = 2, ax=None):
        return self.plot_2d_field(self.j_3d[:, : self.nbpts[1]], num_prec=num_prec, ax=ax)
