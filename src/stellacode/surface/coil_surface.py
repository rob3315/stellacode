import typing as tp

import matplotlib.pyplot as plt
from jax import Array
from jax.typing import ArrayLike

from stellacode import mu_0_fac, np
from stellacode.tools import biot_et_savart, biot_et_savart_op
from stellacode.tools.laplace_force import laplace_force

from .abstract_surface import AbstractBaseFactory, Surface, get_inv_ds_grad
from .current import AbstractCurrent
import jax


class CoilFactory(AbstractBaseFactory):
    """
    Build a coil from a surface and a current class

    Args:
        * current: current class computing a current operator on a given surface grid.
        * build_coils: if True, returns a coilOperator otherwise a CoilSurface
        * compute_grad_current_op: compute also the gradient of the current operator
    """

    current: AbstractCurrent
    build_coils: bool = False
    compute_grad_current_op: bool = False

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
            if k in type(self.current).model_fields:
                setattr(self.current, k, v)

    def __call__(self, surface: Surface, **kwargs):
        coil_op = CoilOperator.from_surface(
            surface=surface,
            current_op=self.current(
                surface.grids, surface.integration_par.max_val_v),
            net_currents=self.current.net_currents,
            phi_mn=self.current.get_phi_mn(),
        )
        if self.compute_grad_current_op:
            coil_op.grad_current_op = self.current.get_grad_current_op(
                surface.grids, surface.integration_par.max_val_v)

        if self.build_coils:
            return coil_op.get_coil(phi_mn=self.current.get_phi_mn())
        else:
            return coil_op


class GroovedCoilFactory(AbstractBaseFactory):
    """
    Build a coil from a surface and grooves

    TODO: add boundary element method to this class to compute automatically
    j3d from the result of get_grooves.

    Args:
        * u_ctr_points: poloidal position of the control points of the grooves with dimensions: N_groove + 1 x N_ctr_points
        * v_ctr_points_w: toroidal position weights of the control points of the grooves with dimensions: N_groove x N_ctr_points
        * trainable_params: list of parameters that should be optimized
    """

    u_ctr_points: Array
    v_ctr_points_w: Array
    trainable_params: tp.List[str] = ["v_ctr_points"]

    @classmethod
    def from_params(cls, num_grooves: int, num_ctr_points: int):
        u_pos = np.tile(np.linspace(0, num_ctr_points, endpoint=False)[
                        None], (num_grooves + 1, 1))
        v_pos = np.zeros((num_grooves, num_ctr_points))
        return cls(u_ctr_points=u_pos, v_ctr_points_w=v_pos)

    def get_trainable_params(self):
        return self.trainable_params

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.trainable_params:
                setattr(self, k, v)

    def __call__(self, surface: Surface, **kwargs):
        j_3d = self.get_j_3d(surface.grids)
        j_surface = np.einsum("klab,kla,kl->klb",
                              surface.get_g_upper_basis(), j_3d, surface.ds)
        coil = CoilSurface.from_surface(surface=surface)
        coil.j_surface = j_surface
        coil.j_3d = j_3d

        return coil

    def get_j_3d(self, grids):
        """Compute j_3d on surface grid points from the groove parameters using the Boundary Element Method"""
        raise NotImplementedError

    def get_grooves(self, u):
        """
        Return a list of values for v corresponding to the position of grooves on the surface.
        There may be advantages (and disadvantages) in using cubic interpolation, which is implemented for jax here:
        https://jax-cosmo.readthedocs.io/en/latest/_modules/jax_cosmo/scipy/interpolate.html

        Note that the first 'groove' is defined by v=0 and the last groove is defined by v=1.
        """
        v_ctr_points = np.cumsum(jax.nn.softmax(self.v_ctr_points_w), axis=0)
        v_ctr_points = np.concatenate(
            (np.zeros_like(v_ctr_points[:1]), v_ctr_points), axis=0)
        return jax.vmap(np.interp, in_axes=(None, 0, 0))(u, self.u_ctr_points, v_ctr_points)


class CoilOperator(Surface):
    """Represent a coil operator

    Args:
        * current_op: tensor for computing the 2D surface current from the current weights
        * net_currents: net poloidal and toroidal currents
        * nfp: number of toroidal periods (Note: should be in a specialized class instead of a general coil class)
    """

    current_op: ArrayLike
    grad_current_op: tp.Optional[
        ArrayLike
    ] = None  # Dimensions of returned op are: dimensions: Ncurrent_op x Nu x Nv x N_j_surf x N_grad
    net_currents: tp.Optional[ArrayLike] = None
    phi_mn: tp.Optional[ArrayLike] = None

    @classmethod
    def from_surface(
        cls,
        surface: Surface,
        current_op: ArrayLike,
        net_currents: tp.Optional[ArrayLike] = None,
        phi_mn: tp.Optional[ArrayLike] = None,
    ):
        dict_ = {k: v for k, v in dict(
            surface).items() if k in cls.model_fields.keys()}
        dict_["current_op"] = current_op
        dict_["net_currents"] = net_currents
        dict_["phi_mn"] = phi_mn
        return cls(**dict_)

    def get_coil(self, phi_mn: tp.Optional[ArrayLike] = None):
        dict_ = {k: v for k, v in dict(self).items() if k not in [
            "current_op", "grad_current_op"]}
        if self.jac_xyz is not None:
            dict_["j_surface"] = self.get_j_surface(phi_mn)
            dict_["j_3d"] = self.get_j_3d(phi_mn)

            if self.grad_current_op is not None:
                dict_["grad_j_surface"] = self.get_grad_j_surface(phi_mn)
                dict_["grad_j_3d"] = self.get_grad_j_3d(phi_mn)

        return CoilSurface(**dict_)

    def get_j_surface(self, phi_mn: tp.Optional[ArrayLike] = None):
        """
        Contravariant components of the current: J^i
        """
        if phi_mn is None:
            phi_mn = self.phi_mn
        return np.einsum("oijk,o->ijk", self.current_op, phi_mn)

    def get_grad_j_surface(self, phi_mn: tp.Optional[ArrayLike] = None):
        """
        Contravariant components of the current: J^i
        """
        if phi_mn is None:
            phi_mn = self.phi_mn
        return np.einsum("oijkl,o->ijkl", self.grad_current_op, phi_mn)

    def get_j_3d(self, phi_mn: tp.Optional[ArrayLike] = None, scale_by_ds: bool = True):
        """Compute the 3D current onto the surface"""
        if phi_mn is None:
            phi_mn = self.phi_mn
        if scale_by_ds:
            return np.einsum("oijk,ijdk,ij,o->ijd", self.current_op, self.jac_xyz, 1 / self.ds, phi_mn)
        else:
            return np.einsum("oijk,ijdk,o->ijd", self.current_op, self.jac_xyz, phi_mn)

    def get_grad_j_3d(self, phi_mn: tp.Optional[ArrayLike] = None):
        """Compute the gradient of the 3D current versus u and v"""
        if phi_mn is None:
            phi_mn = self.phi_mn
        grad_j_surface = self.get_grad_j_surface(phi_mn)

        # Compute the gradient of 1/ds
        invds_grad = get_inv_ds_grad(self)

        fac1 = np.einsum("ijkl,ijdk,ij->ijdl", grad_j_surface,
                         self.jac_xyz, 1 / self.ds)
        fac2 = np.einsum("oijk,ijdkl,ij,o->ijdl", self.current_op,
                         self.hess_xyz, 1 / self.ds, phi_mn)
        fac3 = np.einsum("oijk,ijdk,ijl,o->ijdl", self.current_op,
                         self.jac_xyz, invds_grad, phi_mn)
        return fac1 + fac2 + fac3

    def get_b_field_op(
        self, xyz_plasma: ArrayLike, plasma_normal: tp.Optional[ArrayLike] = None, use_mu_0_factor: bool = True
    ):
        """
        Compute the magnetic field operator

        Args:
            * xyz_plasma: Surface on which the magnetic field is computed
            * plasma_normal: The plasma normal vector, against which is computed the normal magnetic field
            * use_mu_0_factor: Whether to multiply the result by mu_0/4pi

        """
        bs_op = biot_et_savart_op(
            xyz_plasma=xyz_plasma,
            xyz_coil=self.xyz,
            surface_current=self.current_op,
            jac_xyz_coil=self.jac_xyz,
            dudv=self.dudv,
            plasma_normal=plasma_normal,
        )

        if use_mu_0_factor:
            return bs_op*mu_0_fac
        else:
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

    def plot_j_surface(self, num_prec: int = 2, ax=None):
        return self.plot_2d_field(self.get_j_3d()[:, : self.nbpts[1]], num_prec=num_prec, ax=ax)


class CoilSurface(Surface):
    """Represent a coil

    Args:
        * current_op: tensor for computing the 2D surface current from the current weights
        * j_surface: contravariant components of the current: J^i
        * net_currents: net poloidal and toroidal currents
        * nfp: number of toroidal periods (Note: should be in a specialized class instead of a general coil class)
        * phi_mn: current weights
    """

    j_surface: tp.Optional[Array] = None
    j_3d: tp.Optional[Array] = None
    net_currents: tp.Optional[Array] = None
    grad_j_surface: tp.Optional[Array] = None
    grad_j_3d: tp.Optional[Array] = None

    @classmethod
    def from_surface(cls, surface: Surface):
        return cls(**{k: v for k, v in dict(surface).items() if k in cls.model_fields.keys()})

    @property
    def field_keys(self):
        return super().field_keys + ["j_surface", "j_3d"]

    def get_b_field(
        self,
        xyz_plasma: ArrayLike,
        plasma_normal: tp.Optional[ArrayLike] = None,
        use_mu_0_factor: bool = True
    ):
        """
        Compute the magnetic field.

        Args:
            * xyz_plasma: Surface on which the magnetic field is computed
            * plasma_normal: The plasma normal vector, against which is computed the normal magnetic field
            * use_mu_0_factor: Whether to multiply the result by mu_0/4pi

        """
        bf = biot_et_savart(
            xyz_plasma=xyz_plasma,
            xyz_coil=self.xyz,
            j_3d=self.j_3d * self.ds[..., None],
            dudv=self.dudv,
            plasma_normal=plasma_normal,
        )
        if use_mu_0_factor:
            return bf*mu_0_fac
        else:
            return bf

    def get_j_surface(self):
        return self.j_surface

    def naive_laplace_force(self, epsilon: float = 1.0):
        """
        Naive computation of the Laplace force

        Args:
            * epsilon: distance at which the magnetic field is computed in unit of inter points
                grid distance. Should be larger than 1 otherwise the computation may be very inaccurate.
        """
        j_3d = self.j_3d
        dist = np.min(np.linalg.norm(self.xyz[1:] - self.xyz[:-1], axis=-1))

        xyz_ext = self.xyz + epsilon * dist * self.normal_unit
        xyz_int = self.xyz - epsilon * dist * self.normal_unit

        b_avg = self.get_b_field(xyz_ext) + self.get_b_field(xyz_int)

        return 0.5 * np.cross(j_3d, b_avg)

    def laplace_force(
        self,
        cut_coils: tp.Optional[tp.List[int]] = None,
        num_tor_pts: int = 100000,
        end_u: int = 1000000,
        end_v: int = 1000000,
    ):
        g_up = self.get_g_upper_basis()
        if cut_coils is None:
            return laplace_force(
                j_3d_f=self.j_3d[:, :num_tor_pts],
                xyz_f=self.xyz[:, :num_tor_pts],
                j_3d_b=self.j_3d,
                xyz_b=self.xyz,
                normal_unit_b=self.normal_unit,
                ds_b=self.ds,
                g_up_map_b=g_up,
                du=self.du,
                dv=self.dv,
                end_u=end_u,
                end_v=end_v,
            )
        else:
            lap_forces = []
            begin = 0
            _cut_coils = cut_coils + [1000000]
            for end in _cut_coils:
                lap_forces.append(
                    laplace_force(
                        j_3d_f=self.j_3d[:, :num_tor_pts],
                        xyz_f=self.xyz[:, :num_tor_pts],
                        j_3d_b=self.j_3d[:, begin:end],
                        xyz_b=self.xyz[:, begin:end],
                        normal_unit_b=self.normal_unit[:, begin:end],
                        ds_b=self.ds[:, begin:end],
                        g_up_map_b=g_up[:, begin:end],
                        du=self.du,
                        dv=self.dv,
                    )
                )
                begin = end
        return sum(lap_forces)

    def imshow_j(self):
        """Show the current surface density"""
        plt.imshow(np.linalg.norm(self.j_3d, axis=-1), cmap="seismic")

    def plot_j_surface(self, num_prec: int = 2, ax=None):
        return self.plot_2d_field(self.j_3d[:, : self.nbpts[1]], num_prec=num_prec, ax=ax)